from control_cluster_bridge.cluster_client.control_cluster_client import ControlClusterClient

from SharsorIPCpp.PySharsorIPC import Journal, LogType

from typing import List

import os 

from abc import abstractmethod

class LRhcClusterClient(ControlClusterClient):
    
    def __init__(self, 
            namespace: str, 
            robot_pkg_name: str, # robot description ros package name (used to make descr. files available to controllers)
            cluster_size: int,
            set_affinity: bool = False,
            use_mp_fork: bool = False,
            isolated_cores_only: bool = False,
            core_ids_override_list: List[int] = None,
            verbose: bool = False,
            debug: bool = False,
            codegen_base_dirname: str = "CodeGen",
            base_dump_dir: str = "/tmp"):

        self._base_dump_dir = base_dump_dir

        self._temp_path = base_dump_dir + "/" + f"{self.__class__.__name__}" + f"_{namespace}"
        
        self._codegen_base_dirname = codegen_base_dirname
        self._codegen_basedir = self._temp_path + "/" + self._codegen_base_dirname

        if not os.path.exists(self._temp_path):
            os.makedirs(self._temp_path)
        if not os.path.exists(self._codegen_basedir):
            os.makedirs(self._codegen_basedir)

        self.robot_pkg_name = robot_pkg_name

        self._generate_srdf()

        self._generate_urdf()

        super().__init__(namespace = namespace, 
                        cluster_size=cluster_size,
                        isolated_cores_only = isolated_cores_only,
                        set_affinity = set_affinity,
                        use_mp_fork = use_mp_fork,
                        core_ids_override_list = core_ids_override_list,
                        verbose = verbose,
                        debug = debug)
    
    def codegen_dir(self):

        return self._codegen_basedir
    
    def _generate_srdf(self):
        
        Journal.log(self.__class__.__name__,
                        "_generate_srdf",
                        "generating SRDF",
                        LogType.STAT,
                        throw_when_excep = True)

        # we generate the URDF where the Kyon description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        xacro_name = self.robot_pkg_name
        self._srdf_path = self._temp_path + "/" + xacro_name + ".srdf"
        xacro_path = rospackage.get_path(self.robot_pkg_name + "_srdf") + "/srdf/" + xacro_name + ".srdf.xacro"
        
        cmds = self._xrdf_cmds()
        if cmds is None:
            cmds = []
        
        import subprocess
        try:

            xacro_cmd = ["xacro"] + [xacro_path] + cmds + ["-o"] + [self._srdf_path]
            xacro_gen = subprocess.check_call(xacro_cmd)
            Journal.log(self.__class__.__name__,
                        "_generate_srdf",
                        "generated SRDF",
                        LogType.STAT,
                        throw_when_excep = True)
            
        except:
            
            Journal.log(self.__class__.__name__,
                        "_generate_srdf",
                        "Failed to generate Kyon\'s SRDF!!!.",
                        LogType.EXCEP,
                        throw_when_excep = True)
    
    def _generate_urdf(self):
        
        Journal.log(self.__class__.__name__,
                        "_generate_urdf",
                        "Generating URDF",
                        LogType.STAT,
                        throw_when_excep = True)
        
        # we generate the URDF where the Kyon description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        xacro_name = self.robot_pkg_name
        self._urdf_path = self._temp_path + "/" + xacro_name + ".urdf"
        xacro_path = rospackage.get_path(self.robot_pkg_name + "_urdf") + "/urdf/" + xacro_name + ".urdf.xacro"
        
        cmds = self._xrdf_cmds()
        if cmds is None:

            cmds = []

        import subprocess
        try:
            
            xacro_cmd = ["xacro"] + [xacro_path] + cmds + ["-o"] + [self._urdf_path]

            xacro_gen = subprocess.check_call(xacro_cmd)
            
            Journal.log(self.__class__.__name__,
                        "_generate_srdf",
                        "Generated URDF",
                        LogType.STAT,
                        throw_when_excep = True)
            
        except:

            Journal.log(self.__class__.__name__,
                        "_generate_urdf",
                        "Failed to generate Kyon\'s URDF!!!.",
                        LogType.EXCEP,
                        throw_when_excep = True)
            
    @abstractmethod
    def _xrdf_cmds(self):
        
        # to be implemented by parent class (
        # for an example have a look at utils/centauro_xrdf_gen.py)

        pass    
