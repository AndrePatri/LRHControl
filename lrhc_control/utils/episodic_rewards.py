from lrhc_control.utils.episodic_data import EpisodicData
import torch
from typing import List

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

class EpisodicRewards(EpisodicData):

    def __init__(self,
            reward_tensor: torch.Tensor,
            reward_names: List[str] = None,
            max_episode_length: int = 1,
            ep_vec_freq: int = None):

        # the maximum ep length
        super().__init__(data_tensor=reward_tensor, data_names=reward_names, name="SubRewards",
                ep_vec_freq=ep_vec_freq)
        self.set_constant_data_scaling(scaling=max_episode_length)
    
    def set_constant_data_scaling(self, scaling: int):
        # overrides parent

        scaling = torch.full((self._n_envs, 1),
                    fill_value=scaling,
                    dtype=torch.int32,device="cpu") # reward metrics are scaled using
        super().set_constant_data_scaling(enable=True,scaling=scaling)
    
    def enable_timestep_scaling(self):
        super().set_constant_data_scaling(enable=False)

    def update(self, 
        rewards: torch.Tensor,
        ep_finished: torch.Tensor):

        super().update(new_data=rewards, ep_finished=ep_finished)

    def reward_names(self):
        return self._data_names

class EpisodicRewardsOld():

    def __init__(self,
            reward_tensor: torch.Tensor,
            reward_names: List[str] = None):

        self._n_envs = reward_tensor.shape[0]
        self._n_rewards = reward_tensor.shape[1]

        self._episodic_returns = None
        self._current_ep_idx = None

        self._steps_counter = 0

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

        self._episodic_returns = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu") # we don't need it on GPU

        self._current_ep_idx = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")
        
        self._avrg_episodic_return = torch.full(size=(self._n_envs, self._n_rewards), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")
        
        self._tot_avrg_episodic_return = torch.full(size=(self._n_envs, 1), 
                                    fill_value=0.0,
                                    dtype=torch.float32, device="cpu")

        self._steps_counter = 0

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
        
        self._episodic_returns[:, :] = self._episodic_returns[:, :] + step_reward[:, :]

        self._current_ep_idx[is_done.flatten(), 0] = self._current_ep_idx[is_done.flatten(), 0] + 1

        self._steps_counter +=1

    def reward_names(self):

        return self._reward_names
    
    def step_counters(self):

        return self._steps_counter
    
    def get_sub_avrg_over_eps(self, 
            average: bool = True):
    
        # average reward over the performed episodes for each env
        self._avrg_episodic_return = self._episodic_returns / self._current_ep_idx

        if average: # normalize of the number of steps
            return self._avrg_episodic_return / self._steps_counter
        else:
            return self._avrg_episodic_return

    def get_sub_env_avrg_over_eps(self,
                average: bool = True):

        return torch.sum(self.get(average=average), dim=0, keepdim=True)/self._n_envs
    
    def get_total_reward(self, 
            average: bool = True):
        
        # total average reward over the performed episodes for each env
        self._tot_avrg_episodic_return[: , 0] = torch.sum(self._episodic_returns / self._current_ep_idx, dim=1, keepdim=True)

        if average: # normalize of the number of steps
            return self._tot_avrg_episodic_return / self._steps_counter
        else:

            return self._tot_avrg_episodic_return
            
    def get_total_reward_env_avrg(self, 
            average: bool = True):

        return torch.sum(self.get_total(normaaveragelize=average), dim=0, keepdim=True)/self._n_envs

    def get_n_played_episodes(self):

        return torch.sum(self._current_ep_idx).item()

if __name__ == "__main__":  

    n_envs = 1
    data_dim = 3
    max_ep_length = 1
    ep_finished = torch.full((n_envs, 1),fill_value=0,dtype=torch.bool,device="cpu")
    new_data = torch.full((n_envs, data_dim),fill_value=0,dtype=torch.float32,device="cpu")
    data_names = ["okokok", "sdcsdc", "cdcsdcplpl"]
    reward_data = EpisodicRewards(reward_tensor=new_data,
                    reward_names=data_names,
                    max_episode_length=max_ep_length)
    reward_data.reset()

    ep_finished[:, :] = False
    new_data[0, 0] = 1
    new_data[0, 1] = 2
    new_data[0, 2] = 3

    reward_data.update(rewards=new_data,
                ep_finished=ep_finished)

    ep_finished[:, :] = False
    new_data+=1 

    reward_data.update(rewards=new_data,
                ep_finished=ep_finished)
    
    ep_finished[:, :] = False
    new_data+=1 

    reward_data.update(rewards=new_data,
                ep_finished=ep_finished)

    print("get_rollout_stat:")
    print(reward_data.get_sub_avrg_over_eps())

    print("get_rollout_stat_env_avrg:")
    print(reward_data.get_sub_env_avrg_over_eps())

    print("get_rollout_stat_comp:")
    print(reward_data.get_avrg_over_eps())

    print("get_rollout_stat_comp_env_avrg:")
    print(reward_data.get_tot_avrg())