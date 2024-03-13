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

        self._episodic_rewards = None
        self._current_ep_idx = None

        self.reset()

        self.reward_names = reward_names
        if reward_names is not None:
            if not len(reward_names) == self._n_rewards:
                exception = f"Provided reward names length {len(reward_names)} does not match {self._n_rewards}!!"
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)
        else:
            self.reward_names = []
            for i in range(self._n_rewards):
                self.reward_names.append(f"reward_n{i}")
    
    def reset(self):

        self._episodic_rewards = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu") # we don't need it on GPU

        self._current_ep_idx = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")
        
        self._avrg_episodic_reward = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        
        self._tot_avrg_episodic_reward = torch.full(size=(self._n_envs, 1), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        
    def update(self, 
        step_reward: torch.Tensor,
        is_done: torch.Tensor):

        if (not step_reward.shape[0] == self._n_envs) or \
            (not step_reward.shape[1] == self._n_rewards):
            exception = f"Provided step_reward tensor shape {step_reward.shape[0]}, {step_reward.shape[1]}" + \
                f" does not match {self._n_envs}, {self._n_rewards}!!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
        
        if (not is_done.shape[0] == self._n_envs) or \
            (not is_done.shape[1] == 1):
            exception = f"Provided is_done boolean tensor shape {is_done.shape[0]}, {is_done.shape[1]}" + \
                f" does not match {self._n_envs}, {1}!!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
        
        self._episodic_rewards[:, :] = self._episodic_rewards[:, :] + step_reward[:, :]

        self._current_ep_idx[is_done.flatten(), 0] = self._current_ep_idx[is_done.flatten(), 0] + 1
    
    def get(self):
    
        # average reward over the performed episodes for each env
        self._avrg_episodic_reward = self._episodic_rewards / self._current_ep_idx
        return self._avrg_episodic_reward

    def get_total(self):
        
        # total average reward over the performed episodes for each env
        self._tot_avrg_episodic_reward[: , 0] = torch.sum(self._episodic_rewards / self._current_ep_idx, dim=1)
        return self._tot_avrg_episodic_reward


                             
        