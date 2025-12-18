from aug_mpc.training_algs.sac.sactor_critic_algo import SActorCriticAlgoBase

import torch 
import torch.nn as nn
import torch.nn.functional as F

import os

import time

class SAC(SActorCriticAlgoBase):

    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    remote_db=remote_db,
                    seed=seed)

        # target entropy scheduling (per-action values)
        self._entropy_disc_start = -0.1
        self._entropy_disc_end = -2.0
        self._entropy_cont_start = -0.8
        self._entropy_cont_end = -2.5
        # metric (tracking error) range mapped to the above targets
        self._entropy_metric_high = 0.5
        self._entropy_metric_low = 0.03

        self._this_child_path = os.path.abspath(__file__) # overrides parent

    def _sum_log_probs(self, log_prob_vec, ref_tensor, idxs):
        if idxs is None or idxs.numel() == 0:
            return torch.zeros_like(ref_tensor)
        return log_prob_vec.index_select(1, idxs.to(log_prob_vec.device)).sum(1, keepdim=True)

    def _split_log_prob_components(self, log_prob_tuple):
        log_prob_sum, log_prob_vec = log_prob_tuple
        disc_idxs = getattr(self, "_disc_idxs", torch.tensor([], dtype=torch.long, device=log_prob_vec.device))
        cont_idxs = getattr(self, "_cont_idxs", torch.tensor([], dtype=torch.long, device=log_prob_vec.device))
        if disc_idxs.device != log_prob_vec.device:
            disc_idxs = disc_idxs.to(log_prob_vec.device)
        if cont_idxs.device != log_prob_vec.device:
            cont_idxs = cont_idxs.to(log_prob_vec.device)
        log_pi_disc = self._sum_log_probs(log_prob_vec, log_prob_sum, disc_idxs)
        log_pi_cont = self._sum_log_probs(log_prob_vec, log_prob_sum, cont_idxs)
        return log_prob_sum, log_prob_vec, log_pi_disc, log_pi_cont

    def _alpha_tensor(self, which: str, ref_tensor: torch.Tensor):
        if self._autotune:
            if which == "disc":
                return self._log_alpha_disc.exp().detach()
            return self._log_alpha_cont.exp().detach()
        value = self._alpha_disc if which == "disc" else self._alpha_cont
        return ref_tensor.new_tensor(value)
    
    def _collect_transition(self):
        
        # experience collection
        self._switch_training_mode(train=False)

        obs = self._env.get_obs(clone=True) # also accounts for resets when envs are 
        # either terminated or truncated. CRUCIAL: we need to clone, 
        # otherwise obs is a view and will be overridden in the call to step
        # with next_obs!!!
        if self._vec_transition_counter > self._warmstart_vectimesteps or \
            self._resume: # collect actions from policy always if resume, or after warmstart end
            actions, _, _ = self._agent.get_action(x=obs)
            actions = actions.detach()
            if self._n_expl_envs>0 and self._time_to_randomize_actions():
                # this is synchronized across envs, so it's important
                # env takes care of removing temp. correlation between eps (e.g. randomizing
                # eps timelines)
                self._perturb_some_actions(actions=actions)
                
        else: # collect random actions during warmstart if not resume
            actions = self._sample_random_actions()
        
        # perform a step of the (vectorized) env and retrieve trajectory
        env_step_ok = self._env.step(actions)
        
        # add experience to replay buffer
        self._add_experience(obs=obs,
                actions=self._env.get_actions(clone=False, normalized=True), # actions returned to agent space
                rewards=self._env.get_rewards(clone=False), # no need to clone 
                next_obs=self._env.get_next_obs(clone=False), # data is copied anyway
                next_terminal=self._env.get_terminations(clone=False)) 

        return env_step_ok

    def _collect_eval_transition(self):
        
        # experience collection
        self._switch_training_mode(train=False)

        obs = self._env.get_obs(clone=True) # also accounts for resets when envs are 
        # either terminated or truncated. CRUCIAL: we need to clone, 
        # otherwise obs is be a view and will be overridden in the call to step
        # with next_obs!!!

        if not self._override_agent_actions:
            actions, _, mean = self._agent.get_action(x=obs)
            actions = actions.detach()
            
            if self._det_eval: # use mean instead of stochastic policy
                actions[:, :] = mean.detach()

            if self._allow_expl_during_eval:
                if self._n_expl_envs>0:  
                    if self._time_to_randomize_actions():
                        self._perturb_some_actions(actions=actions)

        else:

            self._actions_override.synch_all(read=True,retry=True) # read from CPU
            # write on GPU
            if self._use_gpu:
                self._actions_override.synch_mirror(from_gpu=False,non_blocking=True)
            actions=self._actions_override.get_torch_mirror(gpu=self._use_gpu)

        # perform a step of the (vectorized) env and retrieve trajectory
        env_step_ok = self._env.step(actions)

        if self._load_qf:
            # get qf value for state and action using average of 
            # q networks
            qf1_v=self._agent.get_qf1_val(x=obs,a=actions)
            qf2_v=self._agent.get_qf2_val(x=obs,a=actions)
            qf_v=(qf1_v+qf2_v)/2 # use average
            qf_vals=self._qf_vals.get_torch_mirror(gpu=False)
            qf_vals[:, :]=qf_v.cpu()
            self._qf_vals.synch_all(read=False,retry=False)

            # target qf
            next_obs=self._env.get_next_obs(clone=False)
            next_action, next_log_tuple, _ = self._agent.get_action(next_obs)
            next_log_pi_sum, next_log_pi_vec, next_log_pi_disc, next_log_pi_cont = self._split_log_prob_components(next_log_tuple)
            qf1_v_next=self._agent.get_qf1_val(x=next_obs,a=next_action)
            qf2_v_next=self._agent.get_qf2_val(x=next_obs,a=next_action)
            alpha_disc = self._alpha_tensor("disc", next_log_pi_sum)
            alpha_cont = self._alpha_tensor("cont", next_log_pi_sum)
            entropy_penalty = alpha_disc * next_log_pi_disc + alpha_cont * next_log_pi_cont
            min_qf_next_target = torch.min(qf1_v_next, qf2_v_next) - entropy_penalty
            rew_now=self._env.get_rewards(clone=False)
            reached_terminal_state=self._env.get_terminations(clone=False).to(torch.float32)
            qf_trgt_v=rew_now+(1 - reached_terminal_state)*self._discount_factor*min_qf_next_target
            qf_trgt=self._qf_trgt.get_torch_mirror(gpu=False)
            qf_trgt[:, :]=qf_trgt_v
            self._qf_trgt.synch_all(read=False,retry=False)

        return env_step_ok
    
    def _time_to_randomize_actions(self):
        its_time=(self._vec_transition_counter % self._noise_freq_vec == 0 or \
                self._pert_counter>0)
        return its_time
    
    def _update_policy(self):
        
        # training phase
        if self._vec_transition_counter > self._warmstart_vectimesteps:
            
            self._switch_training_mode(train=True)

            obs,actions,next_obs,rewards,next_terminal = self._sample(size=self._batch_size) # sample
            # experience from replay buffer

            if self._use_rnd:
                # rnd input
                torch.cat(tensors=(obs, actions), dim=1, out=self._rnd_input)
                # add exploration bonus
                raw_bonus_batch=self._rnd_net.get_raw_bonus(self._rnd_input)
                
                with torch.no_grad():
                    # compute intrinsic reward BEFORE updating RND predictor
                    rewards=self._novelty_scaler.process_bonuses(raw_bonus_batch=raw_bonus_batch,
                        raw_reward_batch=rewards.view(-1, 1),
                        return_avg_raw_exp_bonus=None,
                        return_avg_proc_exp_bonus=None,
                        return_all_proc_exp_bonus=self._proc_exp_bonus_all,
                        return_all_norm_exp_bonus=None,
                        return_all_raw_exp_bonus=self._raw_exp_bonus_all)
                    
                if self._update_counter % self._rnd_freq == 0:
                    # train rnd predictor
                    rnd_loss = torch.mean(raw_bonus_batch)
                    self._rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    self._rnd_optimizer.step()
                    
                    self._rnd_loss[self._log_it_counter, 0] = rnd_loss.item()
                    
                    self._n_rnd_updates[self._log_it_counter]+=1

                    if self._debug:
                        # bonus stats
                        self._expl_bonus_raw_avrg[self._log_it_counter, 0] = self._raw_exp_bonus_all.mean().item()
                        self._expl_bonus_raw_std[self._log_it_counter, 0] = self._raw_exp_bonus_all.std().item()
                        self._expl_bonus_proc_avrg[self._log_it_counter, 0] = self._proc_exp_bonus_all.mean().item()
                        self._expl_bonus_proc_std[self._log_it_counter, 0] = self._proc_exp_bonus_all.std().item()

            with torch.no_grad():
                next_action, next_log_tuple, _ = self._agent.get_action(next_obs)
                next_log_pi_sum, next_log_pi_vec, next_log_pi_disc, next_log_pi_cont = self._split_log_prob_components(next_log_tuple)
                qf1_next_target = self._agent.get_qf1t_val(next_obs, next_action)
                qf2_next_target = self._agent.get_qf2t_val(next_obs, next_action)
                alpha_disc = self._alpha_tensor("disc", next_log_pi_sum)
                alpha_cont = self._alpha_tensor("cont", next_log_pi_sum)
                entropy_penalty = alpha_disc * next_log_pi_disc + alpha_cont * next_log_pi_cont
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - entropy_penalty
                next_q_value = rewards.flatten() + (1 - next_terminal.flatten()) * self._discount_factor * (min_qf_next_target).view(-1)
            
            qf1_a_values = self._agent.get_qf1_val(obs, actions).view(-1)
            qf2_a_values = self._agent.get_qf2_val(obs, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            self._qf_optimizer.zero_grad()
            qf_loss.backward()
            self._qf_optimizer.step()
            self._n_qfun_updates[self._log_it_counter]+=1

            if self._update_counter % self._policy_freq == 0:  # TD 3 Delayed update support
                # policy update
                alpha_loss_disc_val = None
                alpha_loss_cont_val = None
                for i in range(self._policy_freq): # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi_tuple, _ = self._agent.get_action(obs)
                    log_pi_sum, log_pi_vec, log_pi_disc, log_pi_cont = self._split_log_prob_components(log_pi_tuple)
                    alpha_disc = self._alpha_tensor("disc", log_pi_sum)
                    alpha_cont = self._alpha_tensor("cont", log_pi_sum)
                    qf1_pi = self._agent.get_qf1_val(obs, pi)
                    qf2_pi = self._agent.get_qf2_val(obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    entropy_penalty = alpha_disc * log_pi_disc + alpha_cont * log_pi_cont
                    actor_loss = (entropy_penalty - min_qf_pi).mean()
                    self._actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self._actor_optimizer.step()
                    if self._autotune:
                        with torch.no_grad():
                            _, log_pi_tuple, _ = self._agent.get_action(obs)
                        log_pi_sum, log_pi_vec, log_pi_disc, log_pi_cont = self._split_log_prob_components(log_pi_tuple)
                        alpha_loss_disc = (-(self._log_alpha_disc.exp()) * (log_pi_disc + self._target_entropy_disc)).mean()
                        alpha_loss_cont = (-(self._log_alpha_cont.exp()) * (log_pi_cont + self._target_entropy_cont)).mean()
                        self._a_optimizer_disc.zero_grad()
                        alpha_loss_disc.backward()
                        self._a_optimizer_disc.step()
                        self._a_optimizer_cont.zero_grad()
                        alpha_loss_cont.backward()
                        self._a_optimizer_cont.step()
                        alpha_loss_disc_val = alpha_loss_disc.item()
                        alpha_loss_cont_val = alpha_loss_cont.item()
                        self._alpha_disc = self._log_alpha_disc.exp().item()
                        self._alpha_cont = self._log_alpha_cont.exp().item()
                        self._alpha = 0.5*(self._alpha_disc + self._alpha_cont)
                    self._n_policy_updates[self._log_it_counter]+=1
                
                if self._debug:
                    # just log last policy update info
                    self._actor_loss[self._log_it_counter, 0] = actor_loss.item()
                    policy_entropy=-log_pi_sum
                    self._policy_entropy_mean[self._log_it_counter, 0] = policy_entropy.mean().item()
                    self._policy_entropy_std[self._log_it_counter, 0] = policy_entropy.std().item()
                    self._policy_entropy_max[self._log_it_counter, 0] = policy_entropy.max().item()
                    self._policy_entropy_min[self._log_it_counter, 0] = policy_entropy.min().item()
                    if self._disc_idxs.numel() > 0:
                        policy_entropy_disc = -log_pi_disc
                        self._policy_entropy_disc_mean[self._log_it_counter, 0] = policy_entropy_disc.mean().item()
                        self._policy_entropy_disc_std[self._log_it_counter, 0] = policy_entropy_disc.std().item()
                        self._policy_entropy_disc_max[self._log_it_counter, 0] = policy_entropy_disc.max().item()
                        self._policy_entropy_disc_min[self._log_it_counter, 0] = policy_entropy_disc.min().item()
                    else:
                        nan = torch.nan
                        self._policy_entropy_disc_mean[self._log_it_counter, 0] = nan
                        self._policy_entropy_disc_std[self._log_it_counter, 0] = nan
                        self._policy_entropy_disc_max[self._log_it_counter, 0] = nan
                        self._policy_entropy_disc_min[self._log_it_counter, 0] = nan
                    if self._cont_idxs.numel() > 0:
                        policy_entropy_cont = -log_pi_cont
                        self._policy_entropy_cont_mean[self._log_it_counter, 0] = policy_entropy_cont.mean().item()
                        self._policy_entropy_cont_std[self._log_it_counter, 0] = policy_entropy_cont.std().item()
                        self._policy_entropy_cont_max[self._log_it_counter, 0] = policy_entropy_cont.max().item()
                        self._policy_entropy_cont_min[self._log_it_counter, 0] = policy_entropy_cont.min().item()
                    else:
                        nan = torch.nan
                        self._policy_entropy_cont_mean[self._log_it_counter, 0] = nan
                        self._policy_entropy_cont_std[self._log_it_counter, 0] = nan
                        self._policy_entropy_cont_max[self._log_it_counter, 0] = nan
                        self._policy_entropy_cont_min[self._log_it_counter, 0] = nan

                    self._alphas[self._log_it_counter, 0] = self._alpha
                    self._alphas_disc[self._log_it_counter, 0] = self._alpha_disc
                    self._alphas_cont[self._log_it_counter, 0] = self._alpha_cont
                    if self._autotune:
                        if alpha_loss_disc_val is not None:
                            self._alpha_loss_disc[self._log_it_counter, 0] = alpha_loss_disc_val
                        if alpha_loss_cont_val is not None:
                            self._alpha_loss_cont[self._log_it_counter, 0] = alpha_loss_cont_val
                        self._alpha_loss[self._log_it_counter, 0] = 0.5*((alpha_loss_disc_val or 0.0)+(alpha_loss_cont_val or 0.0))

            # update the target networks
            if self._update_counter % self._trgt_net_freq == 0:
                for param, target_param in zip(self._agent.qf1.parameters(), self._agent.qf1_target.parameters()):
                    target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
                for param, target_param in zip(self._agent.qf2.parameters(), self._agent.qf2_target.parameters()):
                    target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
                self._n_tqfun_updates[self._log_it_counter]+=1

            if self._debug:
                # DEBUG INFO
        
                # current q estimates on training batch
                self._qf1_vals_mean[self._log_it_counter, 0] = qf1_a_values.mean().item()
                self._qf1_vals_std[self._log_it_counter, 0] = qf1_a_values.std().item()
                self._qf1_vals_max[self._log_it_counter, 0] = qf1_a_values.max().item()
                self._qf1_vals_min[self._log_it_counter, 0] = qf1_a_values.min().item()
                self._qf2_vals_mean[self._log_it_counter, 0] = qf2_a_values.mean().item()
                self._qf2_vals_std[self._log_it_counter, 0] = qf2_a_values.std().item()
                self._qf2_vals_max[self._log_it_counter, 0] = qf2_a_values.max().item()
                self._qf2_vals_min[self._log_it_counter, 0] = qf2_a_values.min().item()
                self._min_qft_vals_mean[self._log_it_counter, 0] = min_qf_next_target.mean().item()
                self._min_qft_vals_std[self._log_it_counter, 0] = min_qf_next_target.std().item()
        
                # q losses (~bellman error)
                self._qf1_loss[self._log_it_counter, 0] = qf1_loss.item()
                self._qf2_loss[self._log_it_counter, 0] = qf2_loss.item()
                    

    def _update_validation_losses(self):
        
        if self._debug and (self._vec_transition_counter > self._warmstart_vectimesteps):
            # wait for training to have started (if in debug)

            obs,actions,next_obs,rewards,next_terminal = self._sample_validation() # sample
            # experience from validation buffer

            with torch.no_grad():
                
                # critics loss
                next_action, next_log_tuple, _ = self._agent.get_action(next_obs)
                next_log_pi_sum, next_log_pi_vec, next_log_pi_disc, next_log_pi_cont = self._split_log_prob_components(next_log_tuple)
                qf1_next_target = self._agent.get_qf1t_val(next_obs, next_action)
                qf2_next_target = self._agent.get_qf2t_val(next_obs, next_action)
                alpha_disc = self._alpha_tensor("disc", next_log_pi_sum)
                alpha_cont = self._alpha_tensor("cont", next_log_pi_sum)
                entropy_penalty = alpha_disc * next_log_pi_disc + alpha_cont * next_log_pi_cont
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - entropy_penalty
                next_q_value = rewards.flatten() + (1 - next_terminal.flatten()) * self._discount_factor * (min_qf_next_target).view(-1)
                
                qf1_a_values = self._agent.get_qf1_val(obs, actions).view(-1)
                qf2_a_values = self._agent.get_qf2_val(obs, actions).view(-1)
                qf1_loss_eval = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss_eval = F.mse_loss(qf2_a_values, next_q_value)

                # actor loss
                pi, log_pi_tuple, _ = self._agent.get_action(obs)
                log_pi_sum, log_pi_vec, log_pi_disc, log_pi_cont = self._split_log_prob_components(log_pi_tuple)
                alpha_disc = self._alpha_tensor("disc", log_pi_sum)
                alpha_cont = self._alpha_tensor("cont", log_pi_sum)
                qf1_pi = self._agent.get_qf1_val(obs, pi)
                qf2_pi = self._agent.get_qf2_val(obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss_eval = ((alpha_disc * log_pi_disc + alpha_cont * log_pi_cont) - min_qf_pi).mean()
                
                # write db data
                self._qf1_loss_validation[self._log_it_counter, 0] = qf1_loss_eval.item()
                self._qf2_loss_validation[self._log_it_counter, 0] = qf2_loss_eval.item()
                self._actor_loss_validation[self._log_it_counter, 0] = actor_loss_eval.item()
                if self._autotune: # also compute alpha loss
                    alpha_loss_disc_eval = (-(self._log_alpha_disc.exp()) * (log_pi_disc + self._target_entropy_disc)).mean()
                    alpha_loss_cont_eval = (-(self._log_alpha_cont.exp()) * (log_pi_cont + self._target_entropy_cont)).mean()
                    self._alpha_loss_validation[self._log_it_counter, 0] = 0.5*(alpha_loss_disc_eval.item()+alpha_loss_cont_eval.item())
                    self._alpha_loss_disc_validation[self._log_it_counter, 0] = alpha_loss_disc_eval.item()
                    self._alpha_loss_cont_validation[self._log_it_counter, 0] = alpha_loss_cont_eval.item()
                
                # compute an index of overfit to training data
                self._update_overfit_idx(loss=(self._qf1_loss[self._log_it_counter, 0]+self._qf2_loss[self._log_it_counter, 0])/2.0, 
                        val_loss=(self._qf1_loss_validation[self._log_it_counter, 0]+self._qf2_loss_validation[self._log_it_counter, 0])/2.0)
                self._overfit_index[self._log_it_counter, 0] = self._overfit_idx                

    def _get_performance_metric(self):
        tracking_err = None
        if "TrackingError" in self._env.custom_db_data:
            # custom db stores tracking error components; take x component average over envs
            track_data = self._env.custom_db_data["TrackingError"].get_avrg_over_envs(env_selector=self._db_env_selector)
            tracking_err = track_data[0, 0].item()

        if tracking_err is None:
            tracking_err = self._episodic_reward_metrics.get_tot_rew_avrg_over_envs(env_selector=
                                                        self._db_env_selector).item()

        self._update_target_entropy_from_metric(tracking_err)
        
        return tracking_err

    def _update_target_entropy_from_metric(self, metric: float):
        if not self._anneal_entropy:
            return
        # map metric into [0, 1] progress
        metric_clamped = max(min(metric, self._entropy_metric_high), self._entropy_metric_low)
        denom = max(self._entropy_metric_high - self._entropy_metric_low, 1e-6)
        progress = (self._entropy_metric_high - metric_clamped) / denom

        # interpolate targets
        trgt_disc = self._entropy_disc_start + progress*(self._entropy_disc_end - self._entropy_disc_start)
        trgt_cont = self._entropy_cont_start + progress*(self._entropy_cont_end - self._entropy_cont_start)

        # update all dependent target entropy values
        self._trgt_avrg_entropy_per_action_disc = trgt_disc
        self._trgt_avrg_entropy_per_action_cont = trgt_cont
        self._target_entropy_disc = float(self._disc_idxs.numel()) * float(trgt_disc)
        self._target_entropy_cont = float(self._cont_idxs.numel()) * float(trgt_cont)
        self._target_entropy = self._target_entropy_disc + self._target_entropy_cont
        self._trgt_avrg_entropy_per_action = self._target_entropy / float(max(self._actions_dim, 1))
