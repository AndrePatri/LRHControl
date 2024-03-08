from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsorIPC import Producer, Consumer

import torch

class ProducerWrapper(): 

    def __init__(self,
            namespace: str,
            basename: str,
            verbose: bool = True,
            vlevel: bool = VLevel.V0,
            force_reconnection: bool = False):

        self._producer = Producer(namespace, 
                            basename, 
                            verbose, 
                            vlevel, 
                            force_reconnection)

    def run(self):

        self._producer.run()
    
    def close(self):

        self._producer.close()
    
    def trigger(self):

        self._producer.trigger()
    
    def wait_ack_from(self,
            n_consumers, ms_timeout = -1):

        return self._producer.wait_ack_from(n_consumers=n_consumers, 
                            ms_timeout=ms_timeout)

class ConsumerWrapper(): 

    def __init__(self,
            namespace: str,
            basename: str,
            verbose: bool = True,
            vlevel: bool = VLevel.V0):

        self._consumer = Consumer(namespace, 
                            basename, 
                            verbose, 
                            vlevel)

    def run(self):

        self._consumer.run()
    
    def close(self):

        self._consumer.close()
    
    def wait(self, ms_timeout = -1):

        return self._consumer.wait(ms_timeout)
    
    def ack(self):

        self._consumer.ack()

class RemoteStepperSrvr(Producer):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        super().__init__(namespace=namespace,
            basename="RemoteStepper",
            verbose=verbose,
            vlevel=vlevel,
            force_reconnection=force_reconnection)

class RemoteStepperClnt(Consumer):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0):

        super().__init__(namespace=namespace,
            basename="RemoteStepper",
            verbose=verbose,
            vlevel=vlevel)
    
class SimEnvReadySrvr(Producer):
    
    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        super().__init__(namespace=namespace,
            basename="SimEnvReady",
            verbose=verbose,
            vlevel=vlevel,
            force_reconnection=force_reconnection)

class SimEnvReadyClnt(Consumer):
    
    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0):

        super().__init__(namespace=namespace,
            basename="SimEnvReady",
            verbose=verbose,
            vlevel=vlevel)

class RemoteResetSrvr(Producer):
    
    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        super().__init__(namespace=namespace,
            basename="RemoteReset",
            verbose=verbose,
            vlevel=vlevel,
            force_reconnection=force_reconnection)

class RemoteResetClnt(Consumer):
    
    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0):

        super().__init__(namespace=namespace,
            basename="RemoteReset",
            verbose=verbose,
            vlevel=vlevel)
        
class RemoteResetRequest(SharedDataView):

        def __init__(self,
                namespace: str,
                n_env: int = None,
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
        
        def to_be_reset(self):

            idxs = torch.nonzero(self.get_torch_view())

            if idxs.shape[0] == 0:

                return None
            
            else:
                
                return idxs.flatten()