# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of LRhcExamples and distributed under the General Public License version 2 license.
# 
# LRhcExamples is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# LRhcExamples is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with LRhcExamples.  If not, see <http://www.gnu.org/licenses/>.
# 
from control_cluster_bridge.cluster_server.control_cluster_srvr import ControlClusterSrvr

class RHClusterSrvr(ControlClusterSrvr):
    
    def __init__(self, 
            robot_name: str):

        self._temp_path = "/tmp/" + f"{self.__class__.__name__}"
        
        self.namespace = robot_name

        self.robot_pkg_name = "aliengo"

        super().__init__(namespace=self.namespace)
        
        self._generate_srdf()
        
    def _generate_srdf(self):
        
        print(f"[{self.__class__.__name__}]"  + f"[{self.journal.status}]" + ": generating SRDF for Control Cluster server")

        # we generate the URDF where the Kyon description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        xacro_name = self.robot_pkg_name
        self._srdf_path = self._temp_path + "/" + xacro_name + ".srdf"
        xacro_path = rospackage.get_path(self.robot_pkg_name + "_srdf") + "/srdf/" + xacro_name + ".srdf.xacro"
        
        import subprocess
        try:

            xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [self._srdf_path]
            xacro_gen = subprocess.check_call(xacro_cmd)

            print(f"[{self.__class__.__name__}]"  + f"[{self.journal.status}]" + ": generated SRDF for Control Cluster server")

        except:

            raise Exception(f"[{self.__class__.__name__}]"  + 
                            f"[{self.journal.status}]" + 
                            ": failed to generate Aliengo\'s SRDF!!!.")
    