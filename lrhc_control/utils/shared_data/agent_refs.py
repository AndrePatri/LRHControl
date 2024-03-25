from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from SharsorIPCpp.PySharsorIPC import VLevel

from typing import List

class AgentRefs(RhcRefs):
    
    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = -1, 
            n_jnts: int = -1,
            jnt_names: List[str] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            safe: bool = True):

        self.basename = "AgentRobotRef"

        n_contacts=4
        contact_names = []
        for i in range(n_contacts):
            contact_names.append(f"contact_n{i}")

        super().__init__(namespace=namespace,
                    is_server=is_server,
                    n_robots=n_robots,
                    n_jnts=n_jnts,
                    n_contacts=n_contacts,
                    jnt_names=jnt_names,
                    contact_names=contact_names,
                    verbose=verbose,
                    vlevel=vlevel,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    safe=safe,
                    with_torch_view=True)