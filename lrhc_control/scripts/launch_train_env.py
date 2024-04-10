from lrhc_control.envs.linvel_env import LinVelTrack
from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline
from lrhc_control.envs.heightchange_baseline_env import LRhcHeightChange

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
    parser.add_argument('--dump_checkpoints', action='store_true', help='Whether to dummp model checkpoints during training')
    parser.add_argument('--eval', action='store_true', help='If set, evaluates policy instead of training')
    parser.add_argument('--n_evals', type=int, help='N. of rollouts on which eval is performed', default=None)
    parser.add_argument('--n_timesteps', type=int, help='N. timesteps for each rollout', default=None)
    parser.add_argument('--mpath', type=str, help='Model path to be used for policy evaluation',default=None)
    parser.add_argument('--use_cpu', action='store_true', help='If set, all the training (data included) will be perfomed on CPU')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--seed', type=int, help='seed', default=1)
    parser.add_argument('--disable_db', action='store_true', help='Whether to disable debug (this includes db prints and remote data logging)')

    args = parser.parse_args()
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)
    
    # env = LRhcHeightChange(namespace=args.ns,
    #                 verbose=True,
    #                 vlevel=VLevel.V2,
    #                 use_gpu=not args.use_cpu,
    #                 debug=True)
    env = LinVelTrackBaseline(namespace=args.ns,
                    verbose=True,
                    vlevel=VLevel.V2,
                    use_gpu=not args.use_cpu,
                    debug=True)
    # env = LinVelTrack(namespace=args.ns,
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
        comment=args.comment,
        eval=args.eval,
        model_path=args.mpath,
        n_evals=args.n_evals,
        n_timesteps_per_eval=args.n_timesteps,
        dump_checkpoints=args.dump_checkpoints)

    try:
        while not ppo.is_done():
            if not args.eval:
                if not ppo.learn():
                    ppo.done()
            else: # eval phase
                if not ppo.eval():
                    ppo.done()
    except KeyboardInterrupt:
        ppo.done() # in case it's interrupted by user