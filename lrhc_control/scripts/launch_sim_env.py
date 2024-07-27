import os
import argparse

import numpy as np
import torch

from lrhc_control.envs.lrhc_sim_env import LRhcIsaacSimEnv
from control_cluster_bridge.utilities.shared_data.sim_data import SharedSimInfo
from omni_robo_gym.utils.rt_factor import RtFactor
from SharsorIPCpp.PySharsorIPC import VLevel

script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

# Function to set CPU affinity
def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sim. env launcher")
    parser.add_argument('--robot_name', type=str, help='Alias to be used for the robot and also shared memory')
    parser.add_argument('--robot_pkg_name', type=str, help='Name of the package for robot description')
    parser.add_argument('--robot_pkg_pref_path', type=str, help='base path to where each robot decription package is located')
    parser.add_argument('--num_envs', type=int)
    parser.add_argument('--cluster_dt', type=float, default=0.03, help='dt at which the control cluster runs')
    parser.add_argument('--dmpdir', type=str, help='directory where data is dumped',default="/root/aux_data")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set 	affinity to')
    parser.add_argument('--contacts_list', nargs='+', default=None,
                        help='Contact sensor list (needs to mathc an available body)')
    parser.add_argument('--remote_stepping', action='store_true', 
                help='Whether to use remote stepping for cluster triggering (to be set during training)')
    parser.add_argument('--cpu_pipeline', action=argparse.BooleanOptionalAction, default=False, help='Whether to use the cpu pipeline (greatly increases GPU RX data)')
    parser.add_argument('--enable_debug', action=argparse.BooleanOptionalAction, default=False, help='Whether to enable debug mode (may introduce significant overhead)')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True, help='Whether to run simulation in headless mode')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--timeout_ms', type=int, help='connection timeout after which the script self-terminates', default=60000)
    parser.add_argument('--default_stiff', type=float, default=200.0, help='default stiffness for low level jnt imp controller')
    parser.add_argument('--default_damp', type=float, default=50.0, help='default damping for low level jnt imp controller')
    parser.add_argument('--start_stiff', type=float, default=200.0, help='stiffness for low level jnt imp controller after controller is activated')
    parser.add_argument('--start_damp', type=float, default=50.0, help='damping for low level jnt imp controller after controller is activated')
    parser.add_argument('--wheel_damp', type=float, default=10.0, help='damping coeff for low level vel control of wheels (if present)')
    parser.add_argument('--wheel_radius', type=float, default=0.124, help='wheel radius (used for contact sensing)')
    parser.add_argument('--spawning_height', type=float, default=0.6, help='initial height at which robots will be spawned')
    parser.add_argument('--physics_dt', type=float, default=1e-3, help='')
    parser.add_argument('--use_custom_jnt_imp', action=argparse.BooleanOptionalAction, default=False, 
        help='Whether to override the default PD controller with a custom one')
    parser.add_argument('--diff_vels', action=argparse.BooleanOptionalAction, default=False, 
        help='Whether to obtain velocities by differentiation or not')
    parser.add_argument('--init_timesteps', type=int, help='initialization timesteps for simulation.', 
            default=5000)

    args = parser.parse_args()

    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)

    robot_name = args.robot_name
    dtype_torch = torch.float32
    num_envs = args.num_envs
    control_clust_dt = args.cluster_dt # [s]. Dt at which RHC controllers run 
    headless = args.headless
    enable_livestream = False
    enable_viewport = False

    # simulation parameters
    sim_params = {}
    sim_params["use_gpu_pipeline"] = not args.cpu_pipeline # disabling gpu pipeline is necessary to be able
    # to retrieve some quantities from the simulator which, otherwise, would have random values
    # sim_params["use_gpu"] = False # does this actually do anything?
    if sim_params["use_gpu_pipeline"]:
        sim_params["device"] = "cuda"
    else:
        sim_params["device"] = "cpu"
    device = sim_params["device"]
    # sim_params["dt"] = 1.0/100.0 # physics_dt?
    sim_params["physics_dt"] = args.physics_dt # physics_dt?
    sim_params["rendering_dt"] = sim_params["physics_dt"]
    sim_params["substeps"] = 1 # number of physics steps to be taken for for each rendering step
    sim_params["gravity"] = np.array([0.0, 0.0, -9.81])
    sim_params["enable_scene_query_support"] = False
    sim_params["use_fabric"] = True # Enable/disable reading of physics buffers directly. Default is True.
    sim_params["replicate_physics"] = True
    # sim_params["worker_thread_count"] = 4
    sim_params["solver_type"] =  1 # 0: PGS, 1:TGS, defaults to TGS. PGS faster but TGS more stable
    sim_params["enable_stabilization"] = True
    # sim_params["bounce_threshold_velocity"] = 0.2
    # sim_params["friction_offset_threshold"] = 0.04
    # sim_params["friction_correlation_distance"] = 0.025
    # sim_params["enable_sleeping"] = True
    # Per-actor settings ( can override in actor_options )
    sim_params["solver_position_iteration_count"] = 4 # defaults to 4
    sim_params["solver_velocity_iteration_count"] = 3 # defaults to 1
    sim_params["sleep_threshold"] = 0.0 # Mass-normalized kinetic energy threshold below which an actor may go to sleep.
    # Allowed range [0, max_float).
    sim_params["stabilization_threshold"] = 1e-5
    # Per-body settings ( can override in actor_options )
    # sim_params["enable_gyroscopic_forces"] = True
    # sim_params["density"] = 1000 # density to be used for bodies that do not specify mass or density
    # sim_params["max_depenetration_velocity"] = 100.0
    # sim_params["solver_velocity_iteration_count"] = 1
    # GPU buffers settings
    # sim_params["gpu_max_rigid_contact_count"] = 512 * 1024
    # sim_params["gpu_max_rigid_patch_count"] = 80 * 1024
    sim_params["gpu_found_lost_pairs_capacity"] = 4096
    sim_params["gpu_found_lost_aggregate_pairs_capacity"] = 4096
    # sim_params["gpu_total_aggregate_pairs_capacity"] = 1024
    # sim_params["gpu_max_soft_body_contacts"] = 1024 * 1024
    # sim_params["gpu_max_particle_contacts"] = 1024 * 1024
    # sim_params["gpu_heap_capacity"] = 64 * 1024 * 1024
    # sim_params["gpu_temp_buffer_capacity"] = 16 * 1024 * 1024
    # sim_params["gpu_max_num_partitions"] = 8

    # create task
    robot_names = [robot_name]

    # contact sensors
    contact_prims = {} # contact sensors to be added
    contact_prims[robot_name] = args.contacts_list # foot contact sensors
    contact_offsets = {}
    contact_offsets[robot_name] = {}
    sensor_radii = {}
    sensor_radii[robot_name] = {}
    if contact_prims[robot_name] is not None:
        for i in range(0, len(contact_prims[robot_name])):
            contact_offsets[robot_name][contact_prims[robot_name][i]] = \
                np.array([0.0, 0.0, 0.0])
            sensor_radii[robot_name][contact_prims[robot_name][i]] = args.wheel_radius      
        
    env = LRhcIsaacSimEnv(headless=headless,
            sim_device = 0,
            enable_livestream=enable_livestream, 
            enable_viewport=enable_viewport,
            debug=args.enable_debug,
            timeout_ms=args.timeout_ms) # create environment

    # now we can import the task (not before, since Omni plugins are loaded 
    # upon environment initialization)
    from lrhc_control.tasks.hybrid_quad_task import HybridQuadTask
    
    task = HybridQuadTask(
            robot_name=robot_name,
            robot_pkg_name=args.robot_pkg_name,
            robot_pkg_pref_path=args.robot_pkg_pref_path,
            integration_dt = sim_params["physics_dt"],
            num_envs = num_envs, 
            cloning_offset = np.array([[0.0, 0.0, args.spawning_height]] * num_envs), 
            env_spacing=6,
            spawning_radius=1.0, 
            use_flat_ground=True, 
            solver_position_iteration_count = sim_params["solver_position_iteration_count"], # applies this to all articulations
            solver_velocity_iteration_count = sim_params["solver_velocity_iteration_count"],
            solver_stabilization_thresh = sim_params["sleep_threshold"],
            default_jnt_stiffness=args.default_stiff, 
            default_jnt_damping=args.default_damp, 
            default_wheel_stiffness=0.0, # wheels are almost never controlled in pos mode
            default_wheel_damping=args.wheel_damp,
            startup_jnt_stiffness=args.start_stiff,
            startup_jnt_damping=args.start_damp,
            startup_wheel_stiffness=0.0, # wheels are almost never controlled in pos mode
            startup_wheel_damping=args.wheel_damp,
            contact_prims = contact_prims,
            contact_offsets = contact_offsets,
            sensor_radii = sensor_radii,
            override_art_controller=args.use_custom_jnt_imp, # uses handmade EXPLICIT controller. This will usually be unstable for relatively high int. dts
            device = device, 
            use_diff_velocities=args.diff_vels, # whether to differentiate velocities numerically
            dtype=dtype_torch,
            debug_enabled = args.enable_debug,
            dump_basepath=args.dmpdir) # writes jnt imp. controller info on shared mem (overhead)
            # and profiles it

    env.set_task(task, 
            cluster_dt = [control_clust_dt],
            backend="torch", 
            use_remote_stepping=[args.remote_stepping],
            n_pre_training_steps=args.init_timesteps, # n of env steps before connecting to training client
            sim_params = sim_params, 
            cluster_client_verbose=args.enable_debug, 
            cluster_client_debug=args.enable_debug,
            verbose=args.verbose, 
            vlevel=VLevel.V2) # add the task to the environment 
    # (includes spawning robots and launching the cluster client for the controllers)
    env.reset(reset_world=True)

    # sim info to be broadcasted on shared memory
    # adding some data to dict for debugging
    sim_params["n_envs"] = num_envs 
    sim_params["control_clust_dt"] = control_clust_dt
    sim_params["headless"] = headless
    sim_params["enable_livestream"] = enable_livestream
    sim_params["enable_viewport"] = enable_viewport
    sim_params["debug_enabled"] = args.enable_debug
    shared_sim_infos = []
    for i in range(len(robot_names)):
        shared_sim_infos.append(SharedSimInfo(
                                namespace=robot_names[i],
                                is_server=True, 
                                sim_params_dict=sim_params,
                                verbose=True,
                                vlevel=VLevel.V2,
                                force_reconnection=True) )
        shared_sim_infos[i].run()

    rt_factor = RtFactor(dt_nom=sim_params["physics_dt"],
                window_size=50000)

    while env._simulation_app.is_running():
        
        # try:
        if rt_factor.reset_due():
            rt_factor.reset()
            
        env.step() 
        rt_factor.update()

        for i in range(len(robot_names)):
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
                                    env.debug_data["cluster_state_update_dt"][robot_names[i]],
                                    env.debug_data["cluster_sol_time"][robot_names[i]],
                                    n_steps,
                                    trigger_counter,
                                    sol_counter,
                                    env.debug_data["sim_time"][robot_names[i]],
                                    sol_counter*env.cluster_servers[robot_name].cluster_dt()
                                    ])

