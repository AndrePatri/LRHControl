from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsorIPC import Producer, Consumer

import torch

class RemoteStepperSrvr(Producer):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        super().__init__(namespace=namespace,
            basename="RemoteStep",
            verbose=verbose,
            vlevel=vlevel,
            force_reconnection=force_reconnection)

class RemoteStepperClnt(Consumer):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0):

        super().__init__(namespace=namespace,
            basename="RemoteStep",
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
        
class RemoteResetRequest(SharedTWrapper):

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
        
            idxs = torch.nonzero(self.get_torch_view().flatten())

            if idxs.shape[0] == 0:

                return None
            
            else:
                
                return idxs.flatten()