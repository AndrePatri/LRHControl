from lrhc_control.training_algs.sac.sactor_critic_algo import SActorCriticAlgoBase

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

        self._this_child_path = os.path.abspath(__file__) # overrides parent
    
    def _collect_transition(self):
        
        # experience collection
        self._switch_training_mode(train=False)

        obs = self._env.get_obs() # also accounts for resets when envs are 
        # either terminated or truncated
        if self._vec_transition_counter > self._warmstart_vectimesteps or \
            self._eval:
            actions, _, _ = self._agent.actor.get_action(x=obs)
            actions = actions.detach()
        else:
            actions = self._sample_random_actions()
                
        # perform a step of the (vectorized) env and retrieve trajectory
        env_step_ok = self._env.step(actions)
        
        if not self._eval:
            self._add_experience(obs=obs,
                    actions=actions,
                    rewards=self._env.get_rewards(),
                    next_obs=self._env.get_next_obs(),
                    terminations=self._env.get_terminations(), 
                    truncations=self._env.get_truncations()) # add experience
            # to rollout buffer

        return env_step_ok
        
    def _update_policy(self):
        
        # training phase
        if self._vec_transition_counter > self._warmstart_vectimesteps:
                
            self._switch_training_mode(train=True)

            obs,next_obs,actions,rewards,_,_,next_done = self._sample() # sample
            # experience from replay buffer
                
            with torch.no_grad():
                next_action, next_log_p, _ = self._agent.actor.get_action(next_obs)
                qf1_next_target = self._agent.qf1_target(next_obs, next_action)
                qf2_next_target = self._agent.qf2_target(next_obs, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_log_p
                next_q_value = rewards.flatten() + (1 - next_done.flatten()) * self._discount_factor * (min_qf_next_target).view(-1)

            qf1_a_values = self._agent.qf1(obs, actions).view(-1)
            qf2_a_values = self._agent.qf2(obs, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            self._qf_optimizer.zero_grad()
            qf_loss.backward()
            self._qf_optimizer.step()

            if self._vec_transition_counter % self._policy_freq == 0:  # TD 3 Delayed update support
                # policy update
                for i in range(self._policy_freq): # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self._agent.actor.get_action(obs)
                    qf1_pi = self._agent.qf1(obs, pi)
                    qf2_pi = self._agent.qf2(obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self._alpha * log_pi) - min_qf_pi).mean()
                    self._actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self._actor_optimizer.step()
                    if self._autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self._agent.actor.get_action(obs)
                        alpha_loss = (-self._log_alpha.exp() * (log_pi + self._target_entropy)).mean()
                        self._a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self._a_optimizer.step()
                        self._alpha = self._log_alpha.exp().item()

            # update the target networks
            if self._vec_transition_counter % self._trgt_net_freq == 0:
                for param, target_param in zip(self._agent.qf1.parameters(), self._agent.qf1_target.parameters()):
                    target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
                for param, target_param in zip(self._agent.qf2.parameters(), self._agent.qf2_target.parameters()):
                    target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
            
