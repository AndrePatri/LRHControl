from typing import List
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

def get_xrdf_cmds_isaac(robot_pkg_name: str = None,
                robot_pkg_pref_path: str = None,
                robot_names: List[str] = None):

        if robot_pkg_name=="kyon":
           return get_xrdf_cmds_isaac_kyon(robot_pkg_name=robot_pkg_name,
                        robot_pkg_pref_path=robot_pkg_pref_path,
                        robot_names=robot_names)
        elif robot_pkg_name=="centauro": 
           return get_xrdf_cmds_isaac_centauro(robot_pkg_name=robot_pkg_name,
                        robot_pkg_pref_path=robot_pkg_pref_path,
                        robot_names=robot_names)
        else:
           exception=f"xrdf cmd getter for robot {robot_pkg_name} not supported! Please modify this file to add your own."
           Journal.log("hybrid_quad_xrdf_gen.py",
                "get_xrdf_cmds_horizon",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
           
def get_xrdf_cmds_horizon(robot_pkg_name: str = None,
                robot_pkg_pref_path: str = None):

        if robot_pkg_name=="kyon":
           return get_xrdf_cmds_horizon_kyon(robot_pkg_name=robot_pkg_name,
                        robot_pkg_pref_path=robot_pkg_pref_path)
        elif robot_pkg_name=="centauro": 
           return get_xrdf_cmds_horizon_centauro(robot_pkg_name=robot_pkg_name,
                        robot_pkg_pref_path=robot_pkg_pref_path)
        else:
           exception=f"xrdf cmd getter for robot {robot_pkg_name} not supported! Please modify this file to add your own."
           Journal.log("hybrid_quad_xrdf_gen.py",
                "get_xrdf_cmds_horizon",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

def get_xrdf_cmds_isaac_centauro(robot_pkg_name: str = None,
                robot_pkg_pref_path: str = None,
                robot_names: List[str] = None):

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
        cmds_aux.append("use_abs_mesh_paths:=true") # use absolute paths for meshes             \       
        
        package_root_path = robot_pkg_pref_path + "/" + f"{robot_pkg_name}_urdf"
        cmds_aux.append(robot_pkg_name + "_root:=" + package_root_path)

        for name in robot_names:
                cmds[name] = cmds_aux

        return cmds

def get_xrdf_cmds_horizon_centauro(robot_pkg_name: str = None,
                robot_pkg_pref_path: str = None):

        cmds = []
        
        xrdf_cmd_vals = [True, True, True, False, False, True] # horizon needs 
        # the floating base
        legs = "true" if xrdf_cmd_vals[0] else "false"
        big_wheel = "true" if xrdf_cmd_vals[1] else "false"
        upper_body ="true" if xrdf_cmd_vals[2] else "false"
        velodyne = "true" if xrdf_cmd_vals[3] else "false"
        realsense = "true" if xrdf_cmd_vals[4] else "false"
        floating_joint = "true" if xrdf_cmd_vals[5] else "false"
                
        cmds.append("legs:=" + legs)
        cmds.append("big_wheel:=" + big_wheel)
        cmds.append("upper_body:=" + upper_body)
        cmds.append("velodyne:=" + velodyne)
        cmds.append("realsense:=" + realsense)
        cmds.append("floating_joint:=" + floating_joint)
        cmds.append("use_abs_mesh_paths:=true") # use absolute paths for meshes             \       
        
        if robot_pkg_name is not None:
                package_root_path = robot_pkg_pref_path + "/" + f"{robot_pkg_name}_urdf"
                cmds.append(robot_pkg_name + "_root:=" + package_root_path)

        return cmds

def get_xrdf_cmds_isaac_kyon(robot_pkg_name: str = None,
                robot_pkg_pref_path: str = None,
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
        
        package_root_path = robot_pkg_pref_path + "/" + f"{robot_pkg_name}_urdf"
        cmds_aux.append(robot_pkg_name + "_root:=" + package_root_path)

        for name in robot_names:
                cmds[name] = cmds_aux

        return cmds

def get_xrdf_cmds_horizon_kyon(robot_pkg_name: str = None,
                robot_pkg_pref_path: str = None):

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
                package_root_path = robot_pkg_pref_path + "/" + f"{robot_pkg_name}_urdf"
                cmds.append(robot_pkg_name + "_root:=" + package_root_path)

        return cmds