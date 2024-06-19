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
            set_affinity: bool = False,
            use_mp_fork: bool = False,
            isolated_cores_only: bool = False,
            core_ids_override_list: List[int] = None,
            verbose: bool = False,
            debug: bool = False,
            open_loop: bool = True,
            base_dump_dir: str = "/tmp",
            timeout_ms: int = 60000,
            codegen_override: str = ""):
        
        self._open_loop = open_loop

        self._paths = PathsGetter()

        self._codegen_dir_name = namespace

        self._timeout_ms = timeout_ms
        
        super().__init__(namespace = namespace, 
                        robot_pkg_name = robot_pkg_name,
                        cluster_size=cluster_size,
                        set_affinity = set_affinity,
                        use_mp_fork = use_mp_fork,
                        isolated_cores_only = isolated_cores_only,
                        core_ids_override_list = core_ids_override_list,
                        verbose = verbose,
                        debug = debug,
                        base_dump_dir=base_dump_dir,
                        codegen_override=codegen_override)
    
    def _xrdf_cmds(self):
        cmds = get_xrdf_cmds_horizon(robot_pkg_name = self.robot_pkg_name)
        return cmds

    def _generate_controller(self,
                        idx: int):
        
        codegen_dir = self.codegen_dir() + f"/{self._codegen_dir_name}Rhc{idx}"
        if not self.codegen_dir_override() == "":
            codegen_dir = f"{self.codegen_dir_override()}{idx}"

        controller = HybridQuadRhc(
                urdf_path=self._urdf_path, 
                srdf_path=self._srdf_path,
                config_path = self._paths.CONFIGPATH,
                robot_name=self._namespace,
                codegen_dir=codegen_dir,
                n_nodes=31, 
                dt=0.03,
                max_solver_iter = 1,
                open_loop = self._open_loop,
                verbose = self._verbose, 
                debug = self._debug)
        
        return controller 