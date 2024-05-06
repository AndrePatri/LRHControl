from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

import torch

from typing import List

class EpisodicData():

    # class for helping log db dta from episodes over 
    # vectorized envs

    def __init__(self,
            name: str,
            data_tensor: torch.Tensor,
            data_names: List[str] = None):

        self._name = name

        self._n_envs = data_tensor.shape[0]
        self._data_size = data_tensor.shape[1]
                            
        self.reset()

        self._data_names = data_names
        if data_names is not None:
            if not len(data_names) == self._data_size:
                exception = f"Provided data names length {len(data_names)} does not match {self._data_size}!!"
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)
        else:
            self._data_names = []
            for i in range(self._data_size):
                self._data_names.append(f"data_n{i}")
    
    def name(self):
        return self._name
    
    def reset(self):
        
        # undiscounted sum of each env, during a single episode
        self._episodic_sum = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu") # we don't need it on GPU
        
        # avrg data of each env, during a single episode, over the number of transitions
        self._episodic_avrg = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        
        # avrg data of each env, over all the ALREADY played episodes.
        self._rollout_sum = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        # avrg over n of episodes (including the current one)
        self._rollout_sum_avrg = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        # current episode index
        self._current_ep_idx = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")
        # current ste counter (within this episode)
        self._steps_counter = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")

    def update(self, 
        new_data: torch.Tensor,
        ep_finished: torch.Tensor):

        if (not new_data.shape[0] == self._n_envs) or \
            (not new_data.shape[1] == self._data_size):
            exception = f"Provided new_data tensor shape {new_data.shape[0]}, {new_data.shape[1]}" + \
                f" does not match {self._n_envs}, {self._data_size}!!"
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

        self._episodic_sum[:, :] = self._episodic_sum + new_data # sum over the current episode
        self._episodic_avrg[:, :] = self._episodic_sum[:, :] / self._steps_counter[:, :] # average bover the played timesteps
        
        self._rollout_sum_avrg[:, :] = (self._rollout_sum + self._episodic_avrg) / self._current_ep_idx[:, :] # average sum over episodes (including current)
        self._rollout_sum[ep_finished.flatten(), :] += self._episodic_avrg[ep_finished.flatten(), :] # sum over ALREADY played episodes

        self._episodic_sum[ep_finished.flatten(), :] = 0 # if finished, reset (undiscounted) ep data

        # increment counters
        self._current_ep_idx[ep_finished.flatten(), 0] = self._current_ep_idx[ep_finished.flatten(), 0] + 1 # an episode has been played
        self._steps_counter[~ep_finished.flatten(), :] +=1 # step performed
        self._steps_counter[ep_finished.flatten(), :] = 1 # reset step counters

    def data_names(self):
        return self._data_names
    
    def step_idx(self):
        return self._steps_counter
    
    def get_rollout_stat(self):
        return self._rollout_sum_avrg

    def get_rollout_stat_env_avrg(self):
        return torch.sum(self.get_rollout_stat(), dim=0, keepdim=True)/self._n_envs
    
    def get_rollout_stat_comp(self):
        return torch.sum(self._rollout_sum_avrg, dim=1, keepdim=True)
            
    def get_rollout_stat_comp_env_avrg(self):
        return torch.sum(self.get_rollout_stat_comp(), dim=0, keepdim=True)/self._n_envs

    def get_n_played_episodes(self):
        return torch.sum(self._current_ep_idx).item()

