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

def get_xrdf_cmds_isaac_centauro(robot_pkg_name: str,
                name = "centauro"):
        
        cmds = {}
        cmds_aux = []
        
        xrdf_cmd_vals = [True, True, True, False, False, False]

        legs = "true" if xrdf_cmd_vals[0] else "false"
        big_wheel = "true" if xrdf_cmd_vals[1] else "false"
        upper_body ="true" if xrdf_cmd_vals[2] else "false"
        velodyne = "true" if xrdf_cmd_vals[3] else "false"
        realsense = "true" if xrdf_cmd_vals[4] else "false"
        floating_joint = "true" if xrdf_cmd_vals[5] else "false"

        cmds_aux.append("legs:=" + legs)
        cmds_aux.append("big_wheel:=" + big_wheel)
        cmds_aux.append("upper_body:=" + upper_body)
        cmds_aux.append("velodyne:=" + velodyne)
        cmds_aux.append("realsense:=" + realsense)
        cmds_aux.append("floating_joint:=" + floating_joint)
        cmds_aux.append("use_abs_mesh_paths:=true") # use absolute paths for meshes

        import rospkg
        rospackage = rospkg.RosPack()
        package_root_path = rospackage.get_path(robot_pkg_name + "_urdf")
        cmds_aux.append("centauro_root:=" + package_root_path)

        cmds[name] = cmds_aux

        return cmds

def get_xrdf_cmds_isaac_aliengo(robot_pkg_name: str,
                        name = "aliengo"):
        
        cmds = {}
        cmds_aux = []

        import rospkg
        rospackage = rospkg.RosPack()
        package_root_path = rospackage.get_path(robot_pkg_name + "_urdf")
        cmds_aux.append("aliengo_root:=" + package_root_path)
        cmds_aux.append("use_abs_mesh_paths:=true") # use absolute paths for meshes
        
        cmds[name] = cmds_aux

        return cmds

def get_xrdf_cmds_horizon():

        cmds = []
        
        xrdf_cmd_vals = [True, True, True, False, False, True]

        legs = "true" if xrdf_cmd_vals[0] else "false"
        big_wheel = "true" if xrdf_cmd_vals[1] else "false"
        upper_body ="true" if xrdf_cmd_vals[2] else "false"
        velodyne = "true" if xrdf_cmd_vals[3] else "false"
        realsense = "true" if xrdf_cmd_vals[4] else "false"
        floating_joint = "true" if xrdf_cmd_vals[5] else "false" # horizon needs a floating joint

        cmds.append("legs:=" + legs)
        cmds.append("big_wheel:=" + big_wheel)
        cmds.append("upper_body:=" + upper_body)
        cmds.append("velodyne:=" + velodyne)
        cmds.append("realsense:=" + realsense)
        cmds.append("floating_joint:=" + floating_joint)

        return cmds
