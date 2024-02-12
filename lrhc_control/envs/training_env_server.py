from lrhc_control.utils.shared_data.remote_stepping import RemoteStepper

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

class TrainingEnvServer:

    def __init__(self, 
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

        self._stepper = RemoteStepper(namespace=namespace,
                            is_server=is_server,
                            verbose=verbose,
                            vlevel=vlevel,
                            force_reconnection=force_reconnection,
                            safe=safe)

    def is_running(self):

        return self._is_running
    
    def run(self):

        self._stepper.run()
    
    def wait_for_step_request(self):

        if not self._is_server:

            exception = f"Can only be called if server."

            Journal.log(self.__class__.__name__,
                "wait_for_step_request",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
        self._stepper.wait_for_step_request()

    def wait_for_step_done(self):

        if self._is_server:

            exception = f"Can only be called if client."

            Journal.log(self.__class__.__name__,
                "wait_for_step_done",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
        self._stepper.wait_for_step_done()
    
    def step_env(self):

        if self._is_server:

            exception = f"Can only be called if client."

            Journal.log(self.__class__.__name__,
                "step_env",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
        self._stepper.step()
    
    def stepped(self):

        if not self._is_server:

            exception = f"Can only be called if server."

            Journal.log(self.__class__.__name__,
                "stepped",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

        self._stepper.stepped()

    def close(self):

        self._stepper.close()