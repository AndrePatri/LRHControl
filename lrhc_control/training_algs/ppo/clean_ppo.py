from lrhc_control.training_algs.ppo.actor_critic_algo import ActorCriticAlgoBase

import torch 
import torch.nn as nn

import os

import time

class CleanPPO(ActorCriticAlgoBase):

    # clearl implementation of PPO. It's characterisic is the tradeoff between code simplicity and
    # correct handling of termination/truncation conditions in the bootstrap process. Specifically, bootstrap
    # is stopped when EITHER termination or truncation occur, which is not theoretically correct (should only happen when
    # terminating). However, not stopping bootstrap at trucations introduces some complications in the code which may not be 
    # justified. 
     
    def __init__(self,
            env, 
            debug = False,
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    seed=seed)

        self._this_child_path = os.path.abspath(__file__) # overrides parent

    def _play(self,
        n_timesteps: int):

        # collect data from current policy over a number of timesteps
        for step in range(n_timesteps):
            
            self._dones[step] = torch.logical_or(self._env.get_terminations(), 
                                        self._env.get_truncations()) # note: this is not
            # correct in theory -> truncations should not be treated as terminations. But introducing
            # this error (underestimates values of truncation states) makes code cleaner (clearnrl does this)

            self._obs[step] = self._env.get_obs()

            # sample actions from latest policy (actor) and state value from latest value function (critic)
            with torch.no_grad(): # no need for gradients computation
                action, logprob, _, value = self._agent.get_action_and_value(self._obs[step])
                self._values[step] = value.reshape(-1, 1)
            self._actions[step] = action.reshape(-1, 1)
            self._logprobs[step] = logprob.reshape(-1, 1)
            
            # perform a step of the (vectorized) env and retrieve 
            env_step_ok = self._env.step(action)
            
            if not env_step_ok:
                return False
            # retrieve new observations, rewards and termination/truncation states
            self._rewards[step] = self._env.get_rewards()
        
        return True
    
    def _compute_returns(self):

        # bootstrap: compute advantages and returns
        with torch.no_grad():
            
            self._advantages.zero_() # reset advantages
            lastgaelam = 0

            for t in reversed(range(self._env_timesteps)):
                if t == self._env_timesteps - 1:
                    # handling last transition in env batch
                    last_done = torch.logical_or(self._env.get_terminations(), 
                                        self._env.get_truncations()).to(self._dtype)
                    nextnonterminal = 1.0 - last_done
                    nextvalues = self._agent.get_value(self._env.get_next_obs()).reshape(-1, 1)
                else:
                    nextnonterminal = 1.0 - self._dones[t + 1]
                    nextvalues = self._values[t + 1]
                # temporal difference error computation
                actual_reward_discounted = self._rewards[t] + self._discount_factor * nextvalues * nextnonterminal
                td_error = actual_reward_discounted - self._values[t] # meas. - est. reward

                # compute advantages using the Generalized Advantage Estimation (GAE) 
                self._advantages[t] = lastgaelam = td_error + self._discount_factor * self._gae_lambda * nextnonterminal * lastgaelam

            # estimated cumulative rewards from each time step to the end of the episode
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
            
                # nan_mask = torch.isnan(batched_obs[minibatch_inds])
                # max_value = torch.max(batched_obs[minibatch_inds]).item()
                # min_value = torch.min(batched_obs[minibatch_inds]).item()
                # nan_mask2 = torch.isnan(batched_actions[minibatch_inds])
                # max_value2 = torch.max(batched_actions[minibatch_inds]).item()
                # min_value2 = torch.min(batched_actions[minibatch_inds]).item()
                # print("Epoch")
                # print(epoch)
                # print("obs nans")
                # print(torch.sum(nan_mask).item())
                # print("obs max/min")
                # print(max_value)
                # print(min_value)
                # print("actions nans")
                # print(torch.sum(nan_mask2).item())
                # print("actions max/min")
                # print(max_value2)
                # print(min_value2)

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

                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

                minibatch_advantages = batched_advantages[minibatch_inds]
                if self._norm_adv: # normalizing advantages if requires
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