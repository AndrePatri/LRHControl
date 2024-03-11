from lrhc_control.controllers.rhc.lrhc_cluster_client import LRhcClusterClient

from lrhc_control.controllers.rhc.horizon_based.hybrid_quad_rhc import HybridQuadRhc
from lrhc_control.utils.hybrid_quad_xrdf_gen import get_xrdf_cmds_horizon
from lrhc_control.utils.sys_utils import PathsGetter

from typing import List

class HybridQuadrupedClusterClient(LRhcClusterClient):
    
    def __init__(self, 
            namespace: str, 
            robot_pkg_name: str,
            cluster_size: int,
            isolated_cores_only: bool = False,
            use_only_physical_cores: bool = False,
            core_ids_override_list: List[int] = None,
            verbose: bool = False,
            open_loop: bool = True):
        
        self._open_loop = open_loop

        self._paths = PathsGetter()

        super().__init__(namespace = namespace, 
                        robot_pkg_name = robot_pkg_name,
                        cluster_size=cluster_size,
                        isolated_cores_only = isolated_cores_only,
                        use_only_physical_cores = use_only_physical_cores,
                        core_ids_override_list = core_ids_override_list,
                        verbose = verbose)
    
    def _xrdf_cmds(self):
        
        cmds = get_xrdf_cmds_horizon(robot_pkg_name = self.robot_pkg_name)

        return cmds

    def _generate_controller(self,
                        idx: int):
        
        controller = HybridQuadRhc(
                urdf_path=self._urdf_path, 
                srdf_path=self._srdf_path,
                cluster_size=self.cluster_size,
                robot_name=self.namespace,
                codegen_dir=self.codegen_dir() + f"/KyonRhc{idx}",
                config_path = self._paths.CONFIGPATH,
                dt=0.03,
                n_nodes=31, 
                max_solver_iter = 1,
                open_loop = self._open_loop,
                verbose = self.verbose, 
                debug = True,
                solver_deb_prints = False,
                profile_all = True,
                publish_sol=True)

        return controller 