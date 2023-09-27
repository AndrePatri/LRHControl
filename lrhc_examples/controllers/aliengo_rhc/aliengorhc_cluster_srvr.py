from control_cluster_utils.cluster_server.control_cluster_srvr import ControlClusterSrvr

class AliengoRHClusterSrvr(ControlClusterSrvr):
    
    def __init__(self):

        self._temp_path = "/tmp/" + f"{self.__class__.__name__}"
        
        self.robot_name = "aliengo"

        super().__init__(namespace=self.robot_name)
        
        self._generate_srdf()
        
    def _generate_srdf(self):
        
        print(f"[{self.__class__.__name__}]"  + f"[{self.journal.status}]" + ": generating SRDF for Control Cluster server")

        # we generate the URDF where the Kyon description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        xacro_name = self.robot_name
        self._srdf_path = self._temp_path + "/" + xacro_name + ".srdf"
        xacro_path = rospackage.get_path(self.robot_name + "_srdf") + "/srdf/" + xacro_name + ".srdf.xacro"
        
        import subprocess
        try:

            xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [self._srdf_path]
            xacro_gen = subprocess.check_call(xacro_cmd)

            print(f"[{self.__class__.__name__}]"  + f"[{self.journal.status}]" + ": generated SRDF for Control Cluster server")

        except:

            raise Exception(f"[{self.__class__.__name__}]"  + 
                            f"[{self.journal.status}]" + 
                            ": failed to generate Aliengo\'s SRDF!!!.")
    