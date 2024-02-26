from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from perf_sleep.pyperfsleep import PerfSleep

import numpy as np
import torch

class RemoteStepperPolling(SharedDataBase):
    
    class StepEnv(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True):
            
            basename = "StepEnvFlag"

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = 1, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomdic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)
    
    class RemoteResetRequest(SharedDataView):

        def __init__(self,
                namespace: str,
                n_env: int,
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True):
            
            basename = "RemoteResetRequest"

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_env, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomdic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)

        def reset(self,
            env_indxs: torch.Tensor = None):

            if not self.is_server:

                resets = self.get_torch_view(gpu=False)

                resets[env_indxs, :] = True

                self.synch_all(read=False,wait=True)
        
        def get(self):
            
            self.synch_all(read=True,wait=True)

            return self.get_torch_view(gpu=False)

        def restore(self):

            if not self.is_server:

                resets = self.get_torch_view(gpu=False)

                resets.zero_()

                self.synch_all(read=False,wait=True)

    def __init__(self,
            namespace: str,
            is_server: bool,
            n_envs: int = None,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            force_reconnection: bool = False,
            safe: bool = True):

        self._namespace = namespace
        self._is_server = is_server
        self._verbose = verbose
        self._vlevel = vlevel
        self._force_reconnection = force_reconnection
        self._safe = safe

        self._n_envs = n_envs

        self._is_running = False

        self._perf_timer = PerfSleep()
        
        self.env_step = self.StepEnv(namespace=self._namespace,
                                is_server=self._is_server,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                force_reconnection=self._force_reconnection,
                                safe=self._safe)

        self.remote_resets = self.RemoteResetRequest(namespace=self._namespace,
                                n_env=self._n_envs,
                                is_server=self._is_server,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                force_reconnection=self._force_reconnection,
                                safe=self._safe)

    def wait_for_step_request(self):
        
        if self.is_running():

            while not self.env_step.read_wait(row_index=0, col_index=0)[0]:

                self._perf_timer.clock_sleep(1000) # nanoseconds 

        else:

            exception = f"Not running. Did you call the run()?"

            Journal.log(self.__class__.__name__,
                "wait_for_step_request",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def reset(self,
            env_indxs: torch.Tensor):

        self.remote_resets.reset(env_indxs.squeeze())

    def get_resets(self):

        resets_copy = self.remote_resets.get().clone()

        self.remote_resets.restore()

        return resets_copy

    def wait(self):
        
        if self.is_running():
            
            if self._is_server:

                while not self.env_step.read_wait(row_index=0, col_index=0)[0]:

                    self._perf_timer.clock_sleep(1000) # nanoseconds 
            
            else:

                while self.env_step.read_wait(row_index=0, col_index=0)[0]:

                    self._perf_timer.clock_sleep(1000) # nanoseconds 

        else:

            exception = f"Not running. Did you call the run()?"

            Journal.log(self.__class__.__name__,
                "wait_for_step_request",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
    def step(self):
        
        if self.is_running():

            if self._is_server:

                self.env_step.write_wait(False, 
                        row_index=0,
                        col_index=0)

            else:

                self.env_step.write_wait(True, 
                        row_index=0,
                        col_index=0)
                
        else:

            exception = f"Not running. Did you call the run()?"

            Journal.log(self.__class__.__name__,
                "step",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

    def is_running(self):

        return self._is_running
    
    def run(self):

        self.env_step.run()
        self.remote_resets.run()

        self._n_envs = self.remote_resets.n_rows

        self._is_running = True
    
    def close(self):
        
        self.env_step.close()