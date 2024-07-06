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
            data_names: List[str] = None, 
            debug: bool = False):

        self._name = name

        self._debug = debug
        
        self._use_constant_scaling = False # whether to use constant 
        # scaling over episodes (this is useful to log meaningful reward data). If not 
        # enabled, metrics are actually averaged over the episode's timesteps, meaning that 
        # no difference between long or short episodes can be seen
        self._scaling = None

        self._n_envs = data_tensor.shape[0]
        self._data_size = data_tensor.shape[1]
                            
        self._init_data()

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
    
    def set_constant_data_scaling(self,
                enable: bool = True,
                scaling: torch.Tensor = None):
        
        if scaling is not None:

            if (not scaling.shape[0] == self._n_envs) or \
            (not scaling.shape[1] == 1):
                exception = f"Provided scaling tensor shape {scaling.shape[0]}, {scaling.shape[1]}" + \
                    f" does not match {self._n_envs}, {1}!!"
                Journal.log(self.__class__.__name__ + f"[{self._name}]",
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)

            self._scaling[:, :] = scaling
        
        self._use_constant_scaling = enable

    def name(self):
        return self._name
    
    def _init_data(self):
        
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

        self._scale_now = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")

        # just used if use_constant_scaling
        self._scaling = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu") # default to scaling 1
        
    def reset(self):
        # reset all data
        self._episodic_sum.zero_()
        self._episodic_avrg.zero_()
        self._rollout_sum.zero_()
        self._rollout_sum_avrg.zero_()
        self._current_ep_idx.fill_(1)
        self._steps_counter.fill_(1)
        self._scale_now.fill_(1)

    def update(self, 
        new_data: torch.Tensor,
        ep_finished: torch.Tensor): # rewards scaled over episode length

        if (not new_data.shape[0] == self._n_envs) or \
            (not new_data.shape[1] == self._data_size):
            exception = f"Provided new_data tensor shape {new_data.shape[0]}, {new_data.shape[1]}" + \
                f" does not match {self._n_envs}, {self._data_size}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
        
        if (not ep_finished.shape[0] == self._n_envs) or \
            (not ep_finished.shape[1] == 1):
            exception = f"Provided ep_finished boolean tensor shape {ep_finished.shape[0]}, {ep_finished.shape[1]}" + \
                f" does not match {self._n_envs}, {1}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        if self._debug and (not torch.isfinite(new_data).all().item()):
            print(new_data)
            exception = f"Found non finite elements in provided data!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
        
        if not self._use_constant_scaling:
            self._scale_now[:, :] = self._steps_counter # use current n of timesteps as scale 
        else:
            self._scale_now[:, :] = self._scaling # constant scaling

        self._episodic_sum[:, :] = self._episodic_sum + new_data # sum over the current episode
        self._episodic_avrg[:, :] = self._episodic_sum[:, :] / self._scale_now[:, :] # average bover the played timesteps
        
        self._rollout_sum_avrg[:, :] = (self._rollout_sum + self._episodic_avrg) / self._current_ep_idx[:, :] # average sum over episodes (including current)
        self._rollout_sum[ep_finished.flatten(), :] += self._episodic_avrg[ep_finished.flatten(), :] # sum over ALREADY played episodes

        self._episodic_sum[ep_finished.flatten(), :] = 0 # if finished, reset (undiscounted) ep data

        # increment counters
        self._current_ep_idx[ep_finished.flatten(), 0] = self._current_ep_idx[ep_finished.flatten(), 0] + 1 # an episode has been played
        self._steps_counter[~ep_finished.flatten(), :] +=1 # step performed
        self._steps_counter[ep_finished.flatten(), :] = 1 # reset step counters

    def data_names(self):
        return self._data_names
    
    def step_counters(self):
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
    
if __name__ == "__main__":  

    n_envs = 1
    data_dim = 3
    ep_finished = torch.full((n_envs, 1),fill_value=0,dtype=torch.bool,device="cpu")
    new_data = torch.full((n_envs, data_dim),fill_value=0,dtype=torch.float32,device="cpu")
    data_scaling = torch.full((n_envs, 1),fill_value=1,dtype=torch.int32,device="cpu")
    data_names = ["okokok", "sdcsdc", "cdcsdcplpl"]
    test_data = EpisodicData("TestData",
                    data_tensor=new_data,
                    data_names=data_names,
                    debug=True)
    
    # with constant scaling
    print("###### CONSTANT SCALING #######")

    test_data.set_constant_data_scaling(enable=True,
                scaling=data_scaling)
    test_data.reset()
    ep_finished[:, :] = False
    new_data[0, 0] = 1
    new_data[0, 1] = 2
    new_data[0, 2] = 3

    test_data.update(new_data=new_data,
                ep_finished=ep_finished)

    ep_finished[:, :] = False
    new_data+=1 

    test_data.update(new_data=new_data,
                ep_finished=ep_finished)
    
    ep_finished[:, :] = False
    new_data+=1 

    test_data.update(new_data=new_data,
                ep_finished=ep_finished)

    
    print("get_rollout_stat:")
    print(test_data.get_rollout_stat())

    print("get_rollout_stat_env_avrg:")
    print(test_data.get_rollout_stat_env_avrg())

    print("get_rollout_stat_comp:")
    print(test_data.get_rollout_stat_comp())

    print("get_rollout_stat_comp_env_avrg:")
    print(test_data.get_rollout_stat_comp_env_avrg())

    # with adaptive scaling
    print("###### ADAPTIVE SCALING #######")
    test_data.set_constant_data_scaling(enable=False,
                scaling=None)
    test_data.reset()
    ep_finished[:, :] = False
    new_data[0, 0] = 1
    new_data[0, 1] = 2
    new_data[0, 2] = 3

    test_data.update(new_data=new_data,
                ep_finished=ep_finished)

    ep_finished[:, :] = False
    new_data+=1 

    test_data.update(new_data=new_data,
                ep_finished=ep_finished)
    
    ep_finished[:, :] = False
    new_data+=1 

    test_data.update(new_data=new_data,
                ep_finished=ep_finished)

    
    print("get_rollout_stat:")
    print(test_data.get_rollout_stat())

    print("get_rollout_stat_env_avrg:")
    print(test_data.get_rollout_stat_env_avrg())

    print("get_rollout_stat_comp:")
    print(test_data.get_rollout_stat_comp())

    print("get_rollout_stat_comp_env_avrg:")
    print(test_data.get_rollout_stat_comp_env_avrg())

    print("get_rollout_stat_comp_env_avrg:")
    print(test_data.get_rollout_stat_comp_env_avrg())
