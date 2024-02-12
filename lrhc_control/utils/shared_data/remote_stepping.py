from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

import numpy as np

from perf_sleep.pyperfsleep import PerfSleep

class RemoteStepper(SharedDataBase):
    
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
            
    def __init__(self,
            namespace: str,
            is_server: bool,
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

        self._is_running = False

        self._perf_timer = PerfSleep()
        
        self.env_step = self.StepEnv(namespace=self._namespace,
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
    
    def wait_for_step_done(self):
        
        if self.is_running():

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
            
    def stepped(self):
        
        if self.is_running():

            self.env_step.write_wait(False,
                    row_index=0,
                    col_index=0)
        
        else:

            exception = f"Not running. Did you call the run()?"

            Journal.log(self.__class__.__name__,
                "stepped",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
    def is_running(self):

        return self._is_running
    
    def run(self):

        self.env_step.run()

        self._is_running = True
    
    def close(self):
        
        self.env_step.close()

