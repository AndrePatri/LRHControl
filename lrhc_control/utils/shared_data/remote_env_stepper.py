from lrhc_control.utils.shared_data.remote_stepping import RemoteStepperPolling

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

from perf_sleep.pyperfsleep import PerfSleep

class RemoteEnvStepper:

    class SimEnvReady(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True):
            
            basename = "SimEnvReadyFlag"

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
    
    class TrainingEnvReady(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True):
            
            basename = "TrainingEnvReady"

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
                n_envs: int = None,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True):

        self._is_running = False

        self._namespace = namespace
        self._is_server = is_server
        self._verbose = verbose
        self._vlevel = vlevel
        self._force_reconnection = force_reconnection
        self._safe = safe

        self._perf_timer = PerfSleep()

        # self._stepper = RemoteStepper(namespace=namespace,
        #                     is_server=is_server,
        #                     verbose=verbose,
        #                     vlevel=vlevel,
        #                     force_reconnection=force_reconnection,
        #                     safe=safe)
        
        self._stepper = RemoteStepperPolling(namespace=namespace,
                            is_server=is_server,
                            n_envs=n_envs,
                            verbose=verbose,
                            vlevel=vlevel,
                            force_reconnection=force_reconnection,
                            safe=safe)
        
        self._sim_env_ready = self.SimEnvReady(namespace=namespace,
                            is_server=is_server,
                            verbose=verbose,
                            vlevel=vlevel,
                            force_reconnection=force_reconnection,
                            safe=safe)

        self._training_env_ready = self.TrainingEnvReady(namespace=namespace,
                            is_server= not is_server,
                            verbose=verbose,
                            vlevel=vlevel,
                            force_reconnection=force_reconnection,
                            safe=safe)
        
        self._is_sim_env_ready = False
        self._is_training_env_ready = False
        
    def is_running(self):

        return self._is_running
    
    def run(self):

        self._stepper.run() # this has to go first since it contains
        # data sem acquisition

        self._sim_env_ready.run()

        self._training_env_ready.run()
        
    def is_sim_env_ready(self):

        return self._sim_env_ready.read_wait(row_index=0, col_index=0)[0]
    
    def is_training_env_ready(self):

        return self._training_env_ready.read_wait(row_index=0, col_index=0)[0]

    def sim_env_ready(self):

        self._sim_env_ready.write_wait(True, 
                row_index=0,
                col_index=0)
    
    def training_env_ready(self):

        self._training_env_ready.write_wait(True, 
                row_index=0,
                col_index=0)
        
    def sim_env_not_ready(self):

        self._sim_env_ready.write_wait(False, 
                row_index=0,
                col_index=0)
    
    def _check_sim_env_ready(self):

        if not self._is_sim_env_ready:

            while not self.is_sim_env_ready():

                self.perf_timer.clock_sleep(1000)
            
            self._is_sim_env_ready = True

    def _check_training_env_ready(self):

        if not self._is_training_env_ready:

            while not self.is_training_env_ready():

                self.perf_timer.clock_sleep(1000)
            
            self._is_training_env_ready = True

    def wait(self):
        
        # if self._is_server:
            
        #     self._check_training_env_ready() # blocking only if tr. env. is not ready

        # else:
            
        #     self._check_sim_env_ready() # blocking only if sim. env. is not ready
            
        self._stepper.wait()
        
    def step(self):

        self._stepper.step()

    def close(self):

        self._stepper.close()
        self._sim_env_ready.close()
        self._training_env_ready.close()