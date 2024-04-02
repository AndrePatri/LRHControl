from lrhc_control.envs.linvel_in_place_env import InPlaceVelTrack
from lrhc_control.envs.linvel_env import LinVelEnv

# from lrhc_control.training_algs.ppo.clean_ppo import CleanPPO
# from lrhc_control.training_algs.ppo.mem_eff_ppo import MemEffPPO
from lrhc_control.training_algs.ppo.ppo import PPO

from control_cluster_bridge.utilities.shared_data.sim_data import SharedSimInfo

from SharsorIPCpp.PySharsorIPC import VLevel

import os, argparse

# Function to set CPU affinity
def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

if __name__ == "__main__":  

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set affinity to')
    parser.add_argument('--run_name', type=str, help='Name of training run', default="LRHCTraining")
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--drop_dir', type=str, help='Directory root where all run data will be dumped')
    parser.add_argument('--eval', action='store_true', help='If set, evaluates policy instead of training')
    parser.add_argument('--eval_tsteps', type=int, help='N. timestep to evaluate if eval flag is set', default=1e4)
    parser.add_argument('--use_cpu', action='store_true', help='If set, all the training (data included) will be perfomed on CPU')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--seed', type=int, help='seed', default=1)
    parser.add_argument('--disable_db', action='store_true', help='Whether to disable debug (this includes db prints and remote data logging)')

    args = parser.parse_args()
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)
    
    env = InPlaceVelTrack(namespace=args.ns,
                    verbose=True,
                    vlevel=VLevel.V2,
                    use_gpu=not args.use_cpu,
                    debug=True)
    # env = LinVelEnv(namespace=args.ns,
    #                 verbose=True,
    #                 vlevel=VLevel.V2,
    #                 use_gpu=not args.use_cpu,
    #                 debug=True)
    # getting some sim info for debugging
    sim_data = {}
    sim_info_shared = SharedSimInfo(namespace=args.ns,
                is_server=False,
                safe=False)
    sim_info_shared.run()
    sim_info_keys = sim_info_shared.param_keys
    sim_info_data = sim_info_shared.get().flatten()
    for i in range(len(sim_info_keys)):
        sim_data[sim_info_keys[i]] = sim_info_data[i]
    
    ppo = PPO(env=env, debug=not args.disable_db, seed=args.seed)
    ppo.setup(run_name=args.run_name, 
        verbose=True,
        drop_dir_name=args.drop_dir,
        custom_args = sim_data,
        comment=args.comment)
    
    if not args.eval:
        try:
            while not ppo.is_done():
                if not ppo.learn():
                    ppo.done() 
        except KeyboardInterrupt:
            ppo.done() # dumps model in case it's interrupted by user
    else:
        ppo.eval(n_timesteps=args.eval_tsteps)