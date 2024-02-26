from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype, toNumpyDType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from lrhc_control.utils.shared_data.base_data import NamedSharedDataView

from typing import Dict, Union, List

import numpy as np
import torch

# Training env info

class TrainingEnvDebData(SharedDataView):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_dims: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        force_reconnection: bool = False,
        safe: bool = True):

        basename = "TrainingEnvDebData" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_dims, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = safe,
            force_reconnection=force_reconnection)

class DynamicTrainingEnvNames:

    def __init__(self):

        self._keys = ["step_rt_factor", 
                "total_rt_factor",
                "env_stepping_dt"]
        
        self.idx_dict = dict.fromkeys(self._keys, None)

        # dynamic sim info is by convention
        # put at the start
        for i in range(len(self._keys)):
            
            self.idx_dict[self._keys[i]] = i

    def get(self):

        return self._keys

    def get_idx(self, name: str):

        return self.idx_dict[name]
    
class SharedTrainingEnvInfo(SharedDataBase):
                           
    def __init__(self, 
                namespace: str,
                is_server = False, 
                training_env_params_dict: Dict = None,
                verbose = True, 
                vlevel = VLevel.V2, 
                safe: bool = True,
                force_reconnection: bool = True):
        
        self.namespace = namespace + "SharedTrainingEnvInfo"

        self._terminate = False
        
        self.is_server = is_server

        self.init = None                                                  

        self.training_env_params_dict = training_env_params_dict
        self._parse_sim_dict() # applies changes if needed

        self.param_keys = []

        self.dynamic_info = DynamicTrainingEnvNames()

        if self.is_server:

            # if client info is read on shared memory

            self.param_keys = self.dynamic_info.get() + list(self.training_env_params_dict.keys())

        # actual data
        self.shared_train_env_data = TrainingEnvDebData(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = len(self.param_keys), 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe, 
                    force_reconnection = force_reconnection)
        
        # names
        if self.is_server:

            self.shared_train_env_datanames = StringTensorServer(length = len(self.param_keys), 
                                        basename = "TrainingEnvDataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel, 
                                        force_reconnection = force_reconnection)

        else:

            self.shared_train_env_datanames = StringTensorClient(
                                        basename = "TrainingEnvDataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)
            
        self._is_running = False
    
    def _parse_sim_dict(self):

        if self.training_env_params_dict is not None:
        
            keys = list(self.training_env_params_dict.keys())

            for key in keys:
        
                if self.training_env_params_dict[key] == "cpu":
    
                    self.training_env_params_dict[key] = 0

                if self.training_env_params_dict[key] == "gpu" or \
                    self.training_env_params_dict[key] == "cuda":
        
                    self.training_env_params_dict[key] = 1
    
    def is_running(self):

        return self._is_running
    
    def run(self):
        
        self.shared_train_env_datanames.run()
        
        self.shared_train_env_data.run()
            
        if self.is_server:
            
            names_written = self.shared_train_env_datanames.write_vec(self.param_keys, 0)

            if not names_written:
                
                exception = "Could not write shared train. env. names on shared memory!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
        else:
            
            self.param_keys = [""] * self.shared_train_env_datanames.length()

            names_read = self.shared_train_env_datanames.read_vec(self.param_keys, 0)

            if not names_read:

                exception = "Could not read shared train. env. names on shared memory!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            self.shared_train_env_data.synch_all(read=True, wait=True)
        
        self.param_values = np.full((len(self.param_keys), 1), 
                                fill_value=np.nan, 
                                dtype=toNumpyDType(sharsor_dtype.Float))

        if self.is_server:
            
            for i in range(len(list(self.training_env_params_dict.keys()))):
                
                # writing static sim info

                dyn_info_size = len(self.dynamic_info.get())

                # first m elements are custom info
                self.param_values[dyn_info_size + i, 0] = \
                    self.training_env_params_dict[self.param_keys[dyn_info_size + i]]
                                        
            self.shared_train_env_data.write_wait(row_index=0,
                                    col_index=0,
                                    data=self.param_values)
            
        self._is_running = True
                          
    def write(self,
            dyn_info_name: Union[str, List[str]],
            val: Union[float, List[float]]):

        # always writes to shared memory
        
        if isinstance(dyn_info_name, list):
            
            if not isinstance(val, list):

                exception = "The provided val should be a list of values!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
            if len(val) != len(dyn_info_name):
                
                exception = "Name list and values length mismatch!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            for i in range(len(val)):
                
                idx = self.dynamic_info.get_idx(dyn_info_name[i])
                
                self.param_values[idx, 0] = val[i]
                
                self.shared_train_env_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
            
        elif isinstance(dyn_info_name, str):
            
            idx = self.dynamic_info.get_idx(dyn_info_name)

            self.param_values[idx, 0] = val
        
            self.shared_train_env_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
    
    def synch(self):

        self.shared_train_env_data.synch_all(read=True, wait = True)
    
    def get(self):

        self.synch()

        return self.shared_train_env_data.numpy_view.copy()
    
    def close(self):

        self.shared_train_env_data.close()
        self.shared_train_env_datanames.close()

    def terminate(self):

        # just an alias for legacy compatibility
        self.close()

    def __del__(self):
        
        self.close()

# training info

class Rewards(NamedSharedDataView):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            n_rewards: int = None, 
            reward_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0):

        basename = "Rewards"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=n_rewards,
                    dtype=sharsor_dtype.Float,
                    col_names=reward_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)

class TotRewards(NamedSharedDataView):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            reward_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0):

        basename = "TotRewards"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=1,
                    dtype=sharsor_dtype.Float,
                    col_names=reward_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)
        
class Observations(NamedSharedDataView):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            obs_dim: int = None,
            obs_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0):

        basename = "Observations"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=obs_dim,
                    dtype=sharsor_dtype.Float,
                    col_names=obs_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)

class Actions(NamedSharedDataView):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            action_dim: int = None,
            action_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0):

        basename = "Actions"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=action_dim,
                    dtype=sharsor_dtype.Float,
                    col_names=action_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)
        
class Terminations(SharedDataView):

    def __init__(self,
            namespace: str,
            n_rows: int = None, 
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0):
            
            basename = "Terminations"
    
            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_rows, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomic on 64 bit systems
                dtype=sharsor_dtype.Bool,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                fill_value = fill_value)

class StepCounterEpisode(SharedDataBase):

    class StepCounter(SharedDataView):

        def __init__(self,
                namespace: str,
                n_rows: int = None, 
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                fill_value = 0):
                
                basename = "StepCounterEpisode"
        
                super().__init__(namespace = namespace,
                    basename = basename,
                    is_server = is_server, 
                    n_rows = n_rows, 
                    n_cols = 1, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe, # boolean operations are atomic on 64 bit systems
                    dtype=sharsor_dtype.Int,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value = fill_value)
    
    def __init__(self,
                namespace: str,
                reset_n_steps: int = None,
                n_rows: int = None, 
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False):

        self._using_gpu = with_gpu_mirror

        self._reset_n_steps = reset_n_steps # reset counters every n steps

        self._step_counter = self.StepCounter(namespace = namespace,
                    is_server = is_server, 
                    n_rows = n_rows, 
                    n_cols = 1, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe, # boolean operations are atomic on 64 bit systems
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror)

    def _write(self):

        if self._using_gpu:

            # copy from gpu to cpu
            self._step_counter.synch_mirror(from_gpu=True)

        # copy from cpu to shared memory
        self._step_counter.synch_all(read=False, wait=True)

    def increment(self):
        
        self.get()[:, :] = self.get() + 1

        self._write()

    def decrement(self):
        
        self.get()[:, :] = self.get() - 1

        self._write()

    def get(self):

        return self._step_counter.get_torch_view(gpu=self._using_gpu)
    
    def counter(self):

        return self._step_counter
    
    def run(self):

        self._step_counter.run()

    def close(self):

        self._step_counter.close()

    def _reset_flags(self):

        return self.get() > self._reset_n_steps
    
    def to_be_reset(self):
        
        to_be_reset = torch.nonzero(self._reset_flags().squeeze()).squeeze()

        if to_be_reset.shape[0] == 0:

            return None
        
        else:

            return to_be_reset
    
    def reset(self,
        env_indxs: torch.Tensor = None,
        all: bool = False):

        if env_indxs is not None:

            self.get()[env_indxs, :] = 0

        if all:

            self.get().zero_()

        self._write()

        return self._reset_flags()

class Truncations(SharedDataView):

    def __init__(self,
            namespace: str,
            n_rows: int = None, 
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0):
            
            basename = "Truncations"
    
            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_rows, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomic on 64 bit systems
                dtype=sharsor_dtype.Bool,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                fill_value = fill_value)