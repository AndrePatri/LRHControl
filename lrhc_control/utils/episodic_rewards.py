from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

import torch

from typing import List

class EpisodicRewards():

    def __init__(self,
            reward_tensor: torch.Tensor,
            reward_names: List[str] = None):

        self._n_envs = reward_tensor.shape[0]
        self._n_rewards = reward_tensor.shape[1]

        self._episodic_returns = None
        self._current_ep_idx = None
                            
        self.reset()

        self._reward_names = reward_names
        if reward_names is not None:
            if not len(reward_names) == self._n_rewards:
                exception = f"Provided reward names length {len(reward_names)} does not match {self._n_rewards}!!"
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)
        else:
            self._reward_names = []
            for i in range(self._n_rewards):
                self._reward_names.append(f"reward_n{i}")
    
    def reset(self):
        
        # undiscounted returns of each env, during a single episode
        self._episodic_returns = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu") # we don't need it on GPU
        
        # avrg reward of each env, during a single episode, over the numper of transitions
        self._episodic_rewards_avrg = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        
        # avrg reward of each env, over all the played episoded between calls to this class
        # reset() method
        self._rollout_avrg_rewards = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        self._rollout_avrg_rewards_avrg = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")

        self._current_ep_idx = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")
        
        self._steps_counter = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")

    def update(self, 
        rewards: torch.Tensor,
        ep_finished: torch.Tensor):

        if (not rewards.shape[0] == self._n_envs) or \
            (not rewards.shape[1] == self._n_rewards):
            exception = f"Provided rewards tensor shape {rewards.shape[0]}, {rewards.shape[1]}" + \
                f" does not match {self._n_envs}, {self._n_rewards}!!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
        
        if (not ep_finished.shape[0] == self._n_envs) or \
            (not ep_finished.shape[1] == 1):
            exception = f"Provided ep_finished boolean tensor shape {ep_finished.shape[0]}, {ep_finished.shape[1]}" + \
                f" does not match {self._n_envs}, {1}!!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        self._episodic_returns[:, :] = self._episodic_returns + rewards

        self._episodic_rewards_avrg[:, :] = self._episodic_returns[:, :] / self._steps_counter[:, :] # average by n timesteps

        self._rollout_avrg_rewards[ep_finished.flatten(), :] = self._rollout_avrg_rewards[ep_finished.flatten(), :] + self._episodic_rewards_avrg[ep_finished.flatten(), :]
        self._rollout_avrg_rewards_avrg[:, :] = self._rollout_avrg_rewards[:, :] / self._current_ep_idx[:, :]

        self._episodic_returns[ep_finished.flatten(), :] = 0 # if finished, reset (undiscounted) ep rewards

        # increment counters
        self._current_ep_idx[ep_finished.flatten(), 0] = self._current_ep_idx[ep_finished.flatten(), 0] + 1 # an episode has been played
        self._steps_counter[~ep_finished.flatten(), :] +=1
        self._steps_counter[ep_finished.flatten(), :] = 1 # reset counters

    def reward_names(self):
        return self._reward_names
    
    def step_idx(self):
        return self._steps_counter
    
    def get_rollout_avrg_reward(self, average=True):
    
        if average: # normalize of the number of steps
            return self._rollout_avrg_rewards_avrg
        else:
            return self._rollout_avrg_rewards

    def get_rollout_reward_env_avrg(self,
                average: bool = True):
        return torch.sum(self.get_rollout_avrg_reward(average=average), dim=0, keepdim=True)/self._n_envs
    
    def get_total_reward(self,
                average: bool = True):
        if average:
            return torch.sum(self._rollout_avrg_rewards_avrg, dim=1, keepdim=True)
        else:
            return torch.sum(self._rollout_avrg_rewards, dim=1, keepdim=True)
            
    def get_total_reward_env_avrg(self, 
            average: bool = True):
        return torch.sum(self.get_total_reward(average=average), dim=0, keepdim=True)/self._n_envs

    def get_n_played_episodes(self):
        return torch.sum(self._current_ep_idx).item()


                             
        