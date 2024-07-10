import os
import argparse

import numpy as np
import torch

from lrhc_control.envs.lrhc_sim_env import LRhcGzXBotSimEnv
from lrhc_control.tasks.lrhc_task import LRHcGzXBotTask

from control_cluster_bridge.utilities.shared_data.sim_data import SharedSimInfo
from omni_robo_gym.utils.rt_factor import RtFactor
from SharsorIPCpp.PySharsorIPC import VLevel

script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sim. env launcher")
    parser.add_argument('--robot_name', type=str, help='Alias to be used for the robot and also shared memory')
    parser.add_argument('--robot_pkg_name', type=str, help='Name of the package for robot description')
    parser.add_argument('--robot_pkg_pref_path', type=str, help='base path to where each robot decription package is located')
    parser.add_argument('--num_envs', type=int)
    parser.add_argument('--cluster_dt', type=float, default=0.03, help='dt at which the control cluster runs')
    parser.add_argument('--dmpdir', type=str, help='directory where data is dumped',default="/root/aux_data")
    parser.add_argument('--contacts_list', nargs='+', default=["lower_leg_1", "lower_leg_2", "lower_leg_3", "lower_leg_4"],
                        help='Contact sensor list (needs to mathc an available body)')
    parser.add_argument('--remote_stepping', action='store_true', 
                help='Whether to use remote stepping for cluster triggering (to be set if training)')
    parser.add_argument('--db',action=argparse.BooleanOptionalAction,default=False, help='Whether to enable debug mode')
    parser.add_argument('--headless',action=argparse.BooleanOptionalAction,default=True, help='Whether to run simulation in headless mode')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--timeout_ms', type=int, help='connection timeout after which the script self-terminates', default=60000)
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=False, help='Whether to use GPU (if applicable)')
    
    args = parser.parse_args()
    args_dict = vars(args)

    robot_name = args.robot_name
    robot_names = [robot_name]
    num_envs = args.num_envs
    control_clust_dt = args.cluster_dt # [s]. Dt at which RHC controllers should run 
    headless = args.headless

    # simulation parameters
    sim_params = {}
    sim_params["use_gpu"]=args.gpu 
    if sim_params["use_gpu"]:
        sim_params["device"]="cuda"
    else:
        sim_params["device"]="cpu"
    device = sim_params["device"]
    sim_params["physics_dt"] = 1.0/1000.0 # physics_dt
    sim_params["rendering_dt"] = sim_params["physics_dt"]
    sim_params["substeps"] = 1 # number of physics steps to be taken for for each rendering step
    sim_params["gravity"] = np.array([0.0, 0.0, -9.81])
    sim_params["solver_type"] = 1     
    sim_params["n_envs"] = num_envs 
    sim_params["control_clust_dt"] = control_clust_dt
    sim_params["headless"] = headless
    sim_params["debug_enabled"] = args.db

    # contact sensors
    contact_prims = {} # contact sensors to be added
    contact_prims[robot_name] = args.contacts_list # foot contact sensors
    contact_offsets = {}
    contact_offsets[robot_name] = {}
    for i in range(0, len(contact_prims[robot_name])):
        contact_offsets[robot_name][contact_prims[robot_name][i]] = \
            np.array([0.0, 0.0, 0.0])
    sensor_radii = {}
    sensor_radii[robot_name] = {}
    for i in range(0, len(contact_prims[robot_name])):
        sensor_radii[robot_name][contact_prims[robot_name][i]] = 0.124
        
    env = LRhcGzXBotSimEnv(headless=headless,
            debug=args.enable_debug,
            timeout_ms=args.timeout_ms) # create environment
    task = LRHcGzXBotTask()

    env.set_task(task)
    env.reset()

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

    while env.running():
    
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

