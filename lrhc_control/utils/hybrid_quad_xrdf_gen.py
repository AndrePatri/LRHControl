from typing import List

def get_xrdf_cmds_isaac(robot_pkg_name: str = None,
                robot_names: List[str] = None):

        cmds = {}
        cmds_aux = []
        
        xrdf_cmd_vals = [False, False, False, False, False, False]

        wheels = "true" if xrdf_cmd_vals[0] else "false"
        upper_body = "true" if xrdf_cmd_vals[1] else "false"
        gripper = "true" if xrdf_cmd_vals[2] else "false"
        sensors = "true" if xrdf_cmd_vals[3] else "false"
        floating_joint = "true" if xrdf_cmd_vals[4] else "false"
        payload = "true" if xrdf_cmd_vals[5] else "false"

        cmds_aux.append("wheels:=" + wheels)
        cmds_aux.append("upper_body:=" + upper_body)
        cmds_aux.append("dagana:=" + gripper)
        cmds_aux.append("sensors:=" + sensors)
        cmds_aux.append("floating_joint:=" + floating_joint)
        cmds_aux.append("payload:=" + payload)
        cmds_aux.append("use_abs_mesh_paths:=true") # use absolute paths for meshes
        
        import rospkg
        rospackage = rospkg.RosPack()
        package_root_path = rospackage.get_path(robot_pkg_name + "_urdf")
        cmds_aux.append(robot_pkg_name + "_root:=" + package_root_path)

        for name in robot_names:
                cmds[name] = cmds_aux

        return cmds

def get_xrdf_cmds_horizon(robot_pkg_name: str = None):

        cmds = []
        
        xrdf_cmd_vals = [True, False, False, False, True, False] # horizon needs 
        # the floating base

        wheels = "true" if xrdf_cmd_vals[0] else "false"
        upper_body = "true" if xrdf_cmd_vals[1] else "false"
        gripper = "true" if xrdf_cmd_vals[2] else "false"
        sensors = "true" if xrdf_cmd_vals[3] else "false"
        floating_joint = "true" if xrdf_cmd_vals[4] else "false"
        payload = "true" if xrdf_cmd_vals[5] else "false"
                
        cmds.append("wheels:=" + wheels)
        cmds.append("upper_body:=" + upper_body)
        cmds.append("dagana:=" + gripper)
        cmds.append("sensors:=" + sensors)
        cmds.append("floating_joint:=" + floating_joint)
        cmds.append("payload:=" + payload)
        
        if robot_pkg_name is not None:

                import rospkg
                rospackage = rospkg.RosPack()
                package_root_path = rospackage.get_path(robot_pkg_name + "_urdf")
                cmds.append(robot_pkg_name + "_root:=" + package_root_path)

        return cmds