import os
import argparse
import importlib.util
import inspect

from lrhc_control.utils.rt_factor import RtFactor
from lrhc_control.utils.custom_arg_parsing import generate_custom_arg_dict

from control_cluster_bridge.utilities.shared_data.sim_data import SharedEnvInfo

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

# Function to dynamically import a module from a specific file path
def import_env_module(env_path):
    spec = importlib.util.spec_from_file_location("env_module", env_path)
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    return env_module

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sim. env launcher")
    parser.add_argument('--robot_name', type=str, help='Alias to be used for the robot and also shared memory')
    parser.add_argument('--robot_urdf_path', type=str, help='path to the URDF file description for each robot')
    parser.add_argument('--robot_srdf_path', type=str, help='path to the SRDF file description for each robot (used for homing)')
    parser.add_argument('--jnt_imp_config_path', type=str, help='path to a valid YAML file containing information on jnt impedance gains')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--n_contacts', type=int, default=4)
    parser.add_argument('--cluster_dt', default=None, type=float, help='dt at which the control cluster runs')
    parser.add_argument('--dmpdir', type=str, help='directory where data is dumped', default="/root/aux_data")
    parser.add_argument('--remote_stepping', action='store_true', 
                help='Whether to use remote stepping for cluster triggering (to be set during training)')
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=True, help='Whether to use gpu simulation')
    parser.add_argument('--enable_debug', action=argparse.BooleanOptionalAction, default=False, help='Whether to enable debug mode (may introduce significant overhead)')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True, help='Whether to run simulation in headless mode')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--timeout_ms', type=int, help='connection timeout after which the script self-terminates', default=60000)
    parser.add_argument('--physics_dt', type=float, default=5e-4, help='')
    parser.add_argument('--use_custom_jnt_imp', action=argparse.BooleanOptionalAction, default=True, 
        help='Whether to override the default PD controller with a custom one')
    parser.add_argument('--diff_vels', action=argparse.BooleanOptionalAction, default=False, 
        help='Whether to obtain velocities by differentiation or not')
    parser.add_argument('--init_timesteps', type=int, help='initialization timesteps', 
            default=1000)
    
    parser.add_argument('--custom_args_names', nargs='+', default=None,
                            help='list of custom arguments names  to cluster client')
    parser.add_argument('--custom_args_vals', nargs='+', default=None,
                            help='list of custom arguments values to cluster client')
    parser.add_argument('--custom_args_dtype', nargs='+', default=None,
                            help='list of custom arguments data types to cluster client')
    
    parser.add_argument('--env_fname', type=str, 
        default="omni_robo_gym.envs.isaac_env",
        help="env file import pattern (without extension)")
    
    args = parser.parse_args()

    # Ensure custom_args_names, custom_args_vals, and custom_args_dtype have the same length
    custom_opt = generate_custom_arg_dict(args=args)

    robot_names = [args.robot_name]
    robot_urdf_paths = [args.robot_urdf_path]
    robot_srdf_paths = [args.robot_srdf_path]
    control_clust_dts = [float(args.cluster_dt)]
    use_remote_stepping = [args.remote_stepping]
    n_contacts = [args.n_contacts]
    jnt_imp_config_paths = [args.jnt_imp_config_path]
    num_envs = args.num_envs
    control_clust_dt = args.cluster_dt # [s]. Dt at which RHC controllers run 
    headless = args.headless

    # simulation parameters
    remote_env_params = {}
    remote_env_params["physics_dt"] = args.physics_dt # physics_dt?
    remote_env_params["n_envs"] = num_envs 
    remote_env_params["use_gpu"] =  args.use_gpu 
    remote_env_params["control_clust_dt"] = control_clust_dts
    remote_env_params["headless"] = headless
    remote_env_params["debug_enabled"] = args.enable_debug
    remote_env_params.update(custom_opt)
    # sim info to be broadcasted on shared memory
    # adding some data to dict for debugging
    shared_sim_infos = []
    for i in range(len(robot_names)):
        shared_sim_infos.append(SharedEnvInfo(
            namespace=robot_names[i],
            is_server=True, 
            env_params_dict=remote_env_params,
            verbose=True,
            vlevel=VLevel.V2,
            force_reconnection=True))
        shared_sim_infos[i].run()

    env_module=importlib.import_module(args.env_fname)
    classes_in_module = [name for name, obj in inspect.getmembers(env_module, inspect.isclass) 
                        if obj.__module__ == env_module.__name__]
    if len(classes_in_module) == 1:
        cluster_classname=classes_in_module[0]
        Env = getattr(env_module, cluster_classname)
    else:
        class_list_str = ", ".join(classes_in_module)
        Journal.log("launch_remote_env.py",
            "",
            f"Found more than one class in env file {args.env_fname}. Found: {class_list_str}",
            LogType.EXCEP,
            throw_when_excep = False)
        exit()

    env = Env(robot_names=robot_names,
        robot_urdf_paths=robot_urdf_paths,
        robot_srdf_paths=robot_srdf_paths,
        cluster_dt=control_clust_dts,
        jnt_imp_config_paths=jnt_imp_config_paths,
        n_contacts=n_contacts,
        use_remote_stepping=use_remote_stepping,
        name=classes_in_module[0],
        num_envs=num_envs,
        debug=args.enable_debug,
        verbose=args.verbose,
        vlevel=VLevel.V2,
        n_init_step=args.init_timesteps,
        timeout_ms=args.timeout_ms,
        env_opts=remote_env_params,
        use_gpu=args.use_gpu) # create environment
    env.reset(reset_sim=True)

    rt_factor = RtFactor(dt_nom=remote_env_params["physics_dt"],
                window_size=50000)

    while env._simulation_app.is_running():
        
        # try:
        if rt_factor.reset_due():
            rt_factor.reset()
            
        env.step() 
        rt_factor.update()

        for i in range(len(robot_names)):
            robot_name=robot_names[i]
            n_steps = env.cluster_step_counters[robot_name]
            sol_counter = env.cluster_servers[robot_name].solution_counter()
            trigger_counter = env.cluster_servers[robot_name].trigger_counter()
            shared_sim_infos[i].write(dyn_info_name=["sim_rt_factor", 
                                                "total_rt_factor", 
                                                "env_stepping_dt",
                                                "world_stepping_dt",
                                                "time_to_get_states_from_sim",
                                                "cluster_state_update_dt",
                                                "cluster_sol_time",
                                                "n_sim_steps",
                                                "n_cluster_trigger_steps",
                                                "n_cluster_sol_steps",
                                                "sim_time",
                                                "cluster_time"],
                                val=[rt_factor.get(), 
                                    rt_factor.get() * num_envs,
                                    rt_factor.get_avrg_step_time(),
                                    env.debug_data["time_to_step_world"],
                                    env.debug_data["time_to_get_states_from_sim"],
                                    env.debug_data["cluster_state_update_dt"][robot_name],
                                    env.debug_data["cluster_sol_time"][robot_name],
                                    n_steps,
                                    trigger_counter,
                                    sol_counter,
                                    env.debug_data["sim_time"][robot_name],
                                    sol_counter*env.cluster_servers[robot_name].cluster_dt()
                                    ])
