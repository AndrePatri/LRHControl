from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

import numpy as np

class RemoteStepper(SharedDataBase):
    
    class StepTriggerFlag(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True):
            
            basename = "StepTriggerFlag"

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
    
    class StepDoneFlag(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True):
            
            basename = "StepDoneFlag"

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

        self._is_first_wait = False

        self._is_running = False
        
        self.env_step_trigger = self.StepTriggerFlag(namespace=self._namespace,
                                is_server= not self._is_server,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                force_reconnection=self._force_reconnection,
                                safe=self._safe)

        self.env_step_done = self.StepDoneFlag(namespace=self._namespace,
                                is_server=self._is_server,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                force_reconnection=self._force_reconnection,
                                safe=self._safe)

    def wait(self):
        
        if self.is_running():

            if self._is_server:
                
                self.env_step_trigger.data_sem_acquire() # blocking

            else:

                self.env_step_done.data_sem_acquire() # blocking
                self.env_step_trigger.data_sem_acquire() 
                
        else:

            exception = f"Not running. Did you call the run()?"

            Journal.log(self.__class__.__name__,
                "wait",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
    def signal(self):
        
        if self.is_running():
            
            if self._is_server:
                
                self.env_step_done.data_sem_release()
                self.env_step_trigger.data_sem_release()

            else:

                self.env_step_trigger.data_sem_release()
        
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

        self.env_step_trigger.run()
        self.env_step_done.run()
        
        if self._is_server:

            self.env_step_done.data_sem_acquire() # owns step detection sem by default
        
        else:

            self.env_step_trigger.data_sem_acquire() # owns step trigger sem by default

        self._is_running = True
    
    def close(self):
        
        self.env_step_trigger.close()
        self.env_step_done.close()

