from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

import torch

from typing import List

class MemBuffer():
    
    # memory buffer for computing runtime std and mean 
    # synchronous means that there is a unique counter for the current
    # position in the buffer
    def __init__(self,
            name: str,
            data_tensor: torch.Tensor,
            data_names: List[str] = None, 
            debug: bool = False,
            horizon: int = 2,
            dtype: torch.dtype = torch.float32,
            use_gpu: bool = False):
        
        self._name = name

        self._dtype = dtype
        self._torch_device = "cuda" if use_gpu else "cpu"

        self._horizon=horizon # number of samples to store
        self._membf_pos=0 #position in mem buff at which new samples with be added
      
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
                self._data_names.append(f"{self._name}_n{i}")

    def _init_data(self):
        
        self._mem_buff=None
        # initialize a memory buffer with a given horizon
        self._mem_buff=torch.full(size=(self._n_envs, self._data_size,self._horizon), 
                fill_value=0.0,
                dtype=self._dtype, 
                device=self._torch_device)
        self._running_mean=torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, 
                device=self._torch_device)
        self._running_std=torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, 
                device=self._torch_device)
        
        self._membf_pos=0
    
    def reset_all(self,
           init_data:torch.Tensor=None):
        if init_data is None: # reset to 0
            self._mem_buff.zero_()
        else:
            # fill all buffer with init provided by data
            if self._debug:
                self._check_new_data(new_data=init_data)
            self._mem_buff[:, :, :]=init_data.unsqueeze(2)
        self._membf_pos=0
        self._running_mean.zero_()
        self._running_std.fill_(0.0)

    def reset(self,
        to_be_reset: torch.Tensor,
        init_data:torch.Tensor=None):

        if init_data is None: # reset to 0
            self._mem_buff[to_be_reset, :, :]=0
        else:
            # fill all buffer with init provided by data
            if self._debug:
                self._check_new_data(new_data=init_data)
            self._mem_buff[to_be_reset, :, :]=init_data[to_be_reset, :].unsqueeze(2)
        # _membf_pos kept at last one
        self._running_mean[to_be_reset, :]=0
        self._running_std[to_be_reset, :]=0.0
        
    def _check_new_data(self,new_data):
        self._check_sizes(new_data=new_data)
        self._check_finite(new_data=new_data)

    def _check_sizes(self,new_data):
        if (not new_data.shape[0] == self._n_envs) or \
            (not new_data.shape[1] == self._data_size):
            exception = f"Provided new_data tensor shape {new_data.shape[0]}, {new_data.shape[1]}" + \
                f" does not match {self._n_envs}, {self._data_size}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
    
    def _check_finite(self,new_data):
        if (not torch.isfinite(new_data).all().item()):
            print(new_data)
            exception = f"Found non finite elements in provided data!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
            
    def update(self, 
        new_data: torch.Tensor):

        if self._debug:
            self._check_new_data(new_data=new_data)

        self._mem_buff[:,:,self._membf_pos]=new_data
        self._running_mean[:, :]=torch.mean(self._mem_buff,dim=2)
        self._running_std[:, :]=torch.std(self._mem_buff,dim=2)

        self._membf_pos+=1
        if self._membf_pos==self.horizon():
            self._membf_pos=0         

    def data_names(self):
        return self._data_names
    
    def horizon(self):
        return self._horizon
    
    def get(self,idx:int=None):
        if idx is None: # always get last 
            return self._mem_buff[:,:,self._membf_pos-1]
        else: # return data ad horizon idx, where 0 means latest
            # and self._horizon-1 mean oldest
            if (not idx>=0) and (idx<self._horizon):
                exception = f"Idx {idx} exceeds horizon length {self._horizon} (0-based)!"
                Journal.log(self.__class__.__name__ + f"[{self._name}]",
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)
            self._horizon-1
            return self._mem_buff[:,:,self._membf_pos-1-idx]
    
    def get_bf(self,clone:bool=False):
        if clone:
            return self._mem_buff.clone()
        else:
            return self._mem_buff
        
    def std(self,clone:bool=False):
        if clone:
            return self._running_std.clone()
        else:
            return self._running_std
    
    def mean(self,clone:bool=False):
        if clone:
            return self._running_mean.clone()
        else:
            return self._running_mean

    def pos(self):
        return self._membf_pos

class EpisodicData():

    # class for helping log db dta from episodes over 
    # vectorized envs

    def __init__(self,
            name: str,
            data_tensor: torch.Tensor,
            data_names: List[str] = None, 
            debug: bool = False,
            dtype: torch.dtype = torch.float32,
            ep_freq: int = None):

        self._keep_track=True

        self._ep_freq=ep_freq

        self._name = name

        self._dtype = dtype

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
        self._current_ep_sum = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=self._dtype, device="cpu") # we don't need it on GPU
        # avrg data of each env, during a single episode, over the number of transitions
        self._current_ep_sum_scaled = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=self._dtype, device="cpu")
        
        # avrg data of each env, over all the ALREADY played episodes.
        self._tot_sum_up_to_now = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=self._dtype, device="cpu")
        # avrg over n of episodes (including the current one)
        self._average_over_eps = torch.full(size=(self._n_envs, self._data_size), 
                                    fill_value=0.0,
                                    dtype=self._dtype, device="cpu")
        if self._ep_freq is not None:
            self._average_over_eps_last = torch.full_like(self._average_over_eps,
                                                fill_value=0.0)
        # current episode index
        self._n_played_eps = torch.full(size=(self._n_envs, 1), 
                                    fill_value=0,
                                    dtype=torch.int32, device="cpu")
        # current ste counter (within this episode)
        self._steps_counter = torch.full(size=(self._n_envs, 1), 
                                    fill_value=0,
                                    dtype=torch.int32, device="cpu")

        self._scale_now = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu")

        # just used if use_constant_scaling
        self._scaling = torch.full(size=(self._n_envs, 1), 
                                    fill_value=1,
                                    dtype=torch.int32, device="cpu") # default to scaling 1

    def reset(self,
            keep_track: bool = None):

        if keep_track is not None:
            if not keep_track:
                self._current_ep_sum.zero_()
                self._steps_counter.zero_()
        else:
            if not self._keep_track: # if not, we propagate ep sum and steps 
                # from before this reset call 
                self._current_ep_sum.zero_()
                self._steps_counter.zero_()
            
        self._current_ep_sum_scaled.zero_()
        self._tot_sum_up_to_now.zero_()
        self._average_over_eps.zero_()
        self._n_played_eps.zero_()

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
            self._scale_now[:, :] = self._steps_counter+1 # use current n of timesteps as scale 
        else:
            self._scale_now[:, :] = self._scaling # constant scaling

        self._current_ep_sum[:, :] = self._current_ep_sum + new_data # sum over the current episode
        self._current_ep_sum_scaled[:, :] = self._current_ep_sum[:, :] / self._scale_now[:, :] # average bover the played timesteps
        
        self._tot_sum_up_to_now[ep_finished.flatten(), :] += self._current_ep_sum_scaled[ep_finished.flatten(), :]

        self._n_played_eps[ep_finished.flatten(), 0] += 1 # an episode has been played
        self._average_over_eps[ep_finished.flatten(), :] = \
            (self._tot_sum_up_to_now[ep_finished.flatten(), :]) / \
                self._n_played_eps[ep_finished.flatten(), :] 
        
        self._current_ep_sum[ep_finished.flatten(), :] = 0 # if finished, reset current sum

        # increment counters
        self._steps_counter[~ep_finished.flatten(), :] +=1 # step performed
        self._steps_counter[ep_finished.flatten(), :] =0 # reset step counters

        if self._ep_freq is not None:
            # automatic reset when self._ep_freq episodes have been played
            if self.get_n_played_episodes()>=self._ep_freq:
                self._average_over_eps_last[:, :]=\
                    self._average_over_eps
                self.reset()        

    def data_names(self):
        return self._data_names
    
    def step_counters(self):
        return self._steps_counter
    
    def get_sub_avrg_over_eps(self):
        if self._ep_freq is not None:
            return self._average_over_eps_last
        else:
            return self._average_over_eps

    def get_sub_env_avrg_over_eps(self):
        return torch.sum(self.get_sub_avrg_over_eps(), dim=0, keepdim=True)/self._n_envs
    
    def get_avrg_over_eps(self):
        return torch.sum(self.get_sub_avrg_over_eps(), dim=1, keepdim=True)
            
    def get_tot_avrg(self):
        return torch.sum(self.get_avrg_over_eps(), dim=0, keepdim=True)/self._n_envs

    def get_n_played_episodes(self):
        return torch.sum(self._n_played_eps).item()
    
if __name__ == "__main__":  

    n_envs = 4
    data_dim = 3
    ep_finished = torch.full((n_envs, 1),fill_value=0,dtype=torch.bool,device="cpu")
    new_data = torch.full((n_envs, data_dim),fill_value=0,dtype=torch.float32,device="cpu")
    data_scaling = torch.full((n_envs, 1),fill_value=1,dtype=torch.int32,device="cpu")
    data_names = ["okokok", "sdcsdc", "cdcsdcplpl"]
    test_data = EpisodicData("TestData",
                data_tensor=new_data,
                data_names=data_names,
                debug=True,
                ep_freq=3)
    
    # # with constant scaling
    # print("###### CONSTANT SCALING #######")

    test_data.set_constant_data_scaling(enable=True,
                scaling=data_scaling)
    test_data.reset()
    ep_finished[:, :] = False
    new_data[:, 0] = 1
    new_data[:, 1] = 2
    new_data[:, 2] = 3

    for i in range(10):
        if i == 9:
            ep_finished[:, :] = True
        test_data.update(new_data=new_data,
                    ep_finished=ep_finished)
    
    # ep_finished[:, :] = False
    # for i in range(5):
    #     # if i == 4:
    #         # ep_finished[:, :] = True
    #     test_data.update(new_data=new_data,
    #                 ep_finished=ep_finished)

    # test_data.reset(keep_track=False)
    
    # for i in range(5):
    #     if i == 4:
    #         ep_finished[:, :] = True
    #     test_data.update(new_data=new_data,
    #                 ep_finished=ep_finished)
        
    print("get_rollout_stat:")
    print(test_data.get_sub_avrg_over_eps())

    print("get_rollout_stat_env_avrg:")
    print(test_data.get_sub_env_avrg_over_eps())

    print("get_rollout_stat_comp:")
    print(test_data.get_avrg_over_eps())

    print("get_rollout_stat_comp_env_avrg:")
    print(test_data.get_tot_avrg())

    # # with adaptive scaling
    # print("###### ADAPTIVE SCALING #######")
    # test_data.set_constant_data_scaling(enable=False,
    #             scaling=None)
    # test_data.reset()
    # ep_finished[:, :] = False
    # new_data[0, 0] = 1
    # new_data[0, 1] = 2
    # new_data[0, 2] = 3

    # test_data.update(new_data=new_data,
    #             ep_finished=ep_finished)

    # ep_finished[:, :] = False
    # new_data+=1 

    # test_data.update(new_data=new_data,
    #             ep_finished=ep_finished)
    
    # ep_finished[:, :] = False
    # new_data+=1 

    # test_data.update(new_data=new_data,
    #             ep_finished=ep_finished)

    
    # print("get_rollout_stat:")
    # print(test_data.get_sub_avrg_over_eps())

    # print("get_rollout_stat_env_avrg:")
    # print(test_data.get_sub_env_avrg_over_eps())

    # print("get_rollout_stat_comp:")
    # print(test_data.get_avrg_over_eps())

    # print("get_rollout_stat_comp_env_avrg:")
    # print(test_data.get_tot_avrg())

    # print("get_rollout_stat_comp_env_avrg:")
    # print(test_data.get_tot_avrg())

    # n_envs = 3
    # data_dim = 4
    # new_data = torch.full((n_envs, data_dim),fill_value=0,dtype=torch.float32,device="cuda")
    # to_be_reset = torch.full((n_envs, 1),fill_value=False,dtype=torch.bool,device="cuda")
    # data_names = ["okokok", "sdcsdc", "cdcsdcplpl","sacasca"]
    # new_data.fill_(1.0)
    # stds = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Example standard deviations for each column

    # mem_buffer=MemBuffer(name="MemBProva",
    #         data_tensor=new_data,
    #         data_names=data_names,
    #         horizon=100000,
    #         dtype=torch.float32,
    #         use_gpu=True)
    
    # mem_buffer.reset(to_be_reset=to_be_reset.flatten())
    # # mem_buffer.reset(init_data=new_data)
    
    # for i in range(mem_buffer.horizon()+1):
    #     noise = torch.randn(n_envs, data_dim) * stds +1
    #     new_data = noise
    #     mem_buffer.update(new_data.cuda())
    #     # if i==(round(mem_buffer.horizon()/2)-1):
    #     #     to_be_reset[2,:]=True
    #     #     print(mem_buffer.get(idx=0)[2,:])
    #     #     mem_buffer.reset(to_be_reset=to_be_reset.flatten())
    #     #     print(mem_buffer.get(idx=0)[2,:])

    # print("pos")
    # print(mem_buffer.pos())
    # print("STD")
    # print(mem_buffer.std())
    # print("AVRG")
    # print(mem_buffer.mean())
    # print("AAAAA")
    # print(mem_buffer.horizon())

