from lrhc_control.training_algs.ppo.actor_critic_algo import ActorCriticAlgoBase

import torch 
import torch.nn as nn

import os

import time

class PPO(ActorCriticAlgoBase):

    # "accurate" implementation of PPO. W.r.t cleanrl's implementation, 
    # this class correctly handles truncations during bootstrap and it does so 
    # trading memory efficiency and some computational overhead with code readability 
    # (obs and next obs are stored in separate tensors)
     
    def __init__(self,
            env, 
            debug = False,
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    seed=seed)

        self._this_child_path = os.path.abspath(__file__) # overrides parent

    def _play(self):

        # collect data from current policy over a number of timesteps
        for transition in range(self._rollout_timesteps):

            self._obs[transition] = self._env.get_obs() # also accounts for resets when envs are 
            # either terminated or truncated

            # sample actions from latest policy (actor) and state value from latest value function (critic)
            with torch.no_grad(): # no need for gradient computation
                action, logprob, _ = self._agent.get_action(self._obs[transition], only_mean=self._eval) # when evaluating, use only mean
                self._values[transition] = self._agent.get_value(self._obs[transition]).reshape(-1, 1)
            
            # perform a step of the (vectorized) env and retrieve trajectory
            env_step_ok = self._env.step(action)

            self._actions[transition] = action
            self._logprobs[transition] = logprob.reshape(-1, 1)
            self._next_obs[transition] = self._env.get_next_obs() # state after env step (get_next_obs
            # holds the current obs even in case of resets. It includes both terminal and 
            # truncation states. It is the "true" state.)
            with torch.no_grad(): # no need for gradient computation
                self._next_values[transition] = self._agent.get_value(self._next_obs[transition]).reshape(-1, 1)
            self._rewards[transition] = self._env.get_rewards()
            self._next_terminations[transition] = self._env.get_terminations()
            self._next_dones[transition] = torch.logical_or(self._env.get_terminations(), self._env.get_truncations())
            
            if not env_step_ok:
                return False
        
        return True
    
    def _compute_returns(self):

        # bootstrap: compute advantages and returns
        with torch.no_grad():
            self._advantages.zero_() # reset advantages
            lastgaelam = 0
            for t in reversed(range(self._rollout_timesteps)):
                # loop over state transitions
                
                # temporal difference error computation (if next obs is a terminal state, no rewards are available in
                # the future)
                nextnonterminal = 1.0 - self._next_terminations[t]
                actual_reward_discounted = self._rewards[t] + self._discount_factor * self._next_values[t] * nextnonterminal
                td_error = actual_reward_discounted - self._values[t] # meas. - est. reward

                # compute advantages using the Generalized Advantage Estimation (GAE) 
                # GAE estimation needs successive transitions, so we need to stop when the trajectory is either 
                # truncated or terminated (that's why nextnondone is used)
                # note that longer trajectories reduce the bias in the GAE estimator
                nextnondone = 1.0 - self._next_dones[t]
                self._advantages[t] = lastgaelam = td_error + self._discount_factor * self._gae_lambda * nextnondone * lastgaelam
            #   cumulative rewards from each time step to the end of the episode
            self._returns[:, :] = self._advantages + self._values

    def _improve_policy(self):

        # flatten batches before policy update
        batched_obs = self._obs.reshape((-1, self._env.obs_dim()))
        batched_logprobs = self._logprobs.reshape(-1)
        batched_actions = self._actions.reshape((-1, self._env.actions_dim()))
        batched_advantages = self._advantages.reshape(-1)
        batched_returns = self._returns.reshape(-1)
        batched_values = self._values.reshape(-1)

        # optimize policy and value network
        clipfracs = []
        for epoch in range(self._update_epochs):
            shuffled_batch_indxs = torch.randperm(self._batch_size) # randomizing 
            # indexes for removing correlations
            for start in range(0, self._batch_size, self._minibatch_size):
                end = start + self._minibatch_size
                minibatch_inds = shuffled_batch_indxs[start:end]

                _, newlogprob, entropy, newvalue = self._agent.get_action_and_value(
                                                                    batched_obs[minibatch_inds], 
                                                                    batched_actions[minibatch_inds])
                logratio = newlogprob - batched_logprobs[minibatch_inds]
                ratio = logratio.exp() # ratio between the probability of taking an action
                # under the current policy and the probability of taking the same action under the policy 

                with torch.no_grad():
                    # calculate approximate KL divergence http://joschu.net/blog/kl-approx.html
                    # The KL (Kullback-Leibler) divergence is a measure of how one probability 
                    # distribution diverges from a second, expected probability distribution
                    # in PPO, this is used as a regularization term in the objective function
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

                minibatch_advantages = batched_advantages[minibatch_inds]
                if self._norm_adv: # normalizing advantages if required
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self._clip_vloss:
                    v_loss_unclipped = (newvalue - batched_returns[minibatch_inds]) ** 2
                    v_clipped = batched_values[minibatch_inds] + torch.clamp(
                        newvalue - batched_values[minibatch_inds],
                        -self._clip_coef,
                        self._clip_coef,
                    )
                    v_loss_clipped = (v_clipped - batched_returns[minibatch_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - batched_returns[minibatch_inds]) ** 2).mean()
                
                # entropy loss
                entropy_loss = entropy.mean()

                # total loss
                loss = pg_loss - self._entropy_coeff * entropy_loss + self._val_f_coeff * v_loss

                # update policy using this minibatch
                self._optimizer.zero_grad() # reset gradients
                loss.backward() # compute backward pass
                nn.utils.clip_grad_norm_(self._agent.parameters(), 
                                    self._max_grad_norm) # clip loss gradient
                self._optimizer.step() # update actor's (policy) parameters
            
            if self._target_kl is not None and approx_kl > self._target_kl:
                break
            
        # ppo-iteration db data
        y_pred, y_true = batched_values.cpu(), batched_returns.cpu()
        var_y = torch.var(y_true)
        explained_var = torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
        self._tot_loss[self._it_counter, 0] = loss.item()
        self._value_loss[self._it_counter, 0] = self._val_f_coeff * v_loss.item()
        self._policy_loss[self._it_counter, 0] = pg_loss.item()
        self._entropy_loss[self._it_counter, 0] = - self._entropy_coeff * entropy_loss.item()
        self._old_approx_kl[self._it_counter, 0] = old_approx_kl.item()
        self._approx_kl[self._it_counter, 0] = approx_kl.item()
        self._clipfrac[self._it_counter, 0] = torch.mean(torch.tensor(clipfracs))
        self._explained_variance[self._it_counter, 0] = explained_var
        self._policy_update_db_data_dict.update({"losses/tot_loss": self._tot_loss[self._it_counter, 0],
                                        "losses/value_loss": self._value_loss[self._it_counter, 0],
                                        "losses/policy_loss": self._policy_loss[self._it_counter, 0],
                                        "losses/entropy": self._entropy_loss[self._it_counter, 0],
                                        "losses/old_approx_kl": self._old_approx_kl[self._it_counter, 0],
                                        "losses/approx_kl": self._approx_kl[self._it_counter, 0],
                                        "losses/clipfrac": self._clipfrac[self._it_counter, 0],
                                        "losses/explained_variance": self._explained_variance[self._it_counter, 0]}) 
