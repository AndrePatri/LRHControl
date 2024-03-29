from control_cluster_bridge.cluster_server.control_cluster_server import ControlClusterServer
from typing import List
from SharsorIPCpp.PySharsorIPC import VLevel

class LRhcClusterServer(ControlClusterServer):

    def __init__(self, 
            robot_name: str,
            cluster_size: int, 
            cluster_dt: float, 
            control_dt: float, 
            jnt_names: List[str],
            n_contact_sensors: int = -1,
            contact_linknames: List[str] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V1,
            debug: bool = False,
            use_gpu: bool = True,
            force_reconnection: bool = True):

        self.robot_name = robot_name
                
        super().__init__( 
            namespace=self.robot_name,
            cluster_size=cluster_size, 
            cluster_dt=cluster_dt, 
            control_dt=control_dt, 
            jnt_names=jnt_names,
            n_contact_sensors = n_contact_sensors,
            contact_linknames = contact_linknames, 
            verbose=verbose, 
            vlevel=vlevel,
            debug=debug,
            use_gpu=use_gpu,
            force_reconnection=force_reconnection)