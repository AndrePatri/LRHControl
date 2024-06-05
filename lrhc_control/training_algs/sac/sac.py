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
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    seed=seed)

        self._this_child_path = os.path.abspath(__file__) # overrides parent

    def eval(self):

        pass

    def learn(self):
        
        if not self._setup_done:
            self._should_have_called_setup()
        
        self._episodic_reward_getter.reset() # necessary, we don't want to accumulate 
        # debug rewards from previous rollouts

        for transition in range(self._total_timesteps):
            
            self._switch_training_mode(train=False)

            obs = self._env.get_obs() # also accounts for resets when envs are 
            # either terminated or truncated
            if transition > self._warmstart_timesteps:
                actions, _, _ = self._agent.get_action(x=obs)
                actions = actions.detach()
            else:
                actions = self._sample_random_actions()
                
            # perform a step of the (vectorized) env and retrieve trajectory
            env_step_ok = self._env.step(actions)
            
            self._add_experience(obs=obs,
                    actions=actions,
                    rewards=self._env.get_rewards(),
                    next_obs=self._env.get_next_obs(),
                    terminations=self._env.get_terminations(), 
                    truncations=self._env.get_truncations())

            print(f"transition {transition} done")
            if not env_step_ok:
                return False
            
            if transition > self._warmstart_timesteps:
                
                self._switch_training_mode(train=True)

                obs,next_obs,actions,rewards,_,_,next_done = self._sample()
                
                with torch.no_grad():
                    next_action, next_log_p, _ = self._agent.get_action(next_obs)
                    qf1_next_target = self._qf1_target(next_obs, next_action)
                    qf2_next_target = self._qf2_target(next_obs, next_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_log_p
                    next_q_value = rewards.flatten() + (1 - next_done.flatten()) * self._discount_factor * (min_qf_next_target).view(-1)

                qf1_a_values = self._qf1(obs, actions).view(-1)
                qf2_a_values = self._qf2(obs, actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self._qf_optimizer.zero_grad()
                qf_loss.backward()
                self._qf_optimizer.step()

                print("qf step done")

                if transition % self._policy_freq == 0:  # TD 3 Delayed update support
                    for i in range(self._policy_freq): # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self._agent.get_action(obs)
                        qf1_pi = self._qf1(obs, pi)
                        qf2_pi = self._qf2(obs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self._alpha * log_pi) - min_qf_pi).mean()
                        print("ijijdicidmimimii")
                        self._actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self._actor_optimizer.step()
                        print("polocy step done")
                        if self._autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self._agent.get_action(obs)
                            alpha_loss = (-self._log_alpha.exp() * (log_pi + self._target_entropy)).mean()

                            self._a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self._a_optimizer.step()
                            self._alpha = self._log_alpha.exp().item()

                # update the target networks
                if transition % self._trgt_net_freq == 0:
                    for param, target_param in zip(self._qf1.parameters(), self._qf1_target.parameters()):
                        target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
                    for param, target_param in zip(self._qf2.parameters(), self._qf2_target.parameters()):
                        target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
                    print("updated target network")

        self._post_step()
