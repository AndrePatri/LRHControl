from EigenIPC.PyEigenIPC import VLevel, Journal, LogType
from typing import List
from aug_mpc.utils.description_adapter import prepare_xrdf_input
from aug_mpc.utils.description_adapter import rewrite_package_uris

def generate_srdf(robot_name: str, 
        xacro_path: str,
        dump_path: str = "/tmp",
        xrdf_cmds: List[str] = None):
        
        srdf_dump_path = dump_path + "/" + robot_name + ".srdf"
        xacro_path, package_map = prepare_xrdf_input(
            xacro_path=xacro_path,
            dump_path=dump_path)

        if xrdf_cmds is not None:
            xacro_cmd = ["xacro"] + [xacro_path] + xrdf_cmds + ["-o"] + [srdf_dump_path]
        else:
            xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [srdf_dump_path]

        import subprocess
        try:
            xacro_gen = subprocess.check_call(xacro_cmd)
        except:
            Journal.log("xrdf_gen.py",
                "generate_urdf",
                "failed to generate " + robot_name + "\'S SRDF!!!",
                LogType.EXCEP,
                throw_when_excep = True)
        rewrite_package_uris(xrdf_path=srdf_dump_path, package_map=package_map)
        return srdf_dump_path
            
def generate_urdf(robot_name: str, 
    xacro_path: str,
    dump_path: str = "/tmp",
    xrdf_cmds: List[str] = None):

    # we generate the URDF where the description package is located
    urdf_dump_path = dump_path + "/" + robot_name + ".urdf"
    xacro_path, package_map = prepare_xrdf_input(
        xacro_path=xacro_path,
        dump_path=dump_path)
    
    if xrdf_cmds is not None:
        xacro_cmd = ["xacro"] + [xacro_path] + xrdf_cmds + ["-o"] + [urdf_dump_path]
    else:
        xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [urdf_dump_path]

    import subprocess
    try:
        xacro_gen = subprocess.check_call(xacro_cmd)
        # we also generate an updated SRDF
    except:
        Journal.log("xrdf_gen.py",
            "_generate_urdf",
            "Failed to generate " + robot_name + "\'s URDF!!!",
            LogType.EXCEP,
            throw_when_excep = True)
    rewrite_package_uris(xrdf_path=urdf_dump_path, package_map=package_map)
    return urdf_dump_path
