from lrhc_control.envs.heightchange_env import LRhcHeightChange
from lrhc_control.training_algs.ppo.clean_ppo import CleanPPO
# from lrhc_control.training_algs.ppo.mem_eff_ppo import MemEffPPO
# from lrhc_control.training_algs.ppo.ppo import PPO

from SharsorIPCpp.PySharsorIPC import VLevel

import os, argparse

from datetime import datetime

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

    args = parser.parse_args()
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)
    
    env = LRhcHeightChange(namespace=args.ns,
                    verbose=True,
                    vlevel=VLevel.V2,
                    use_gpu=not args.use_cpu)

    ppo = CleanPPO(env=env, debug=True)
    time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
    ppo.setup(run_name=time_id + "-" + args.run_name, 
        verbose=True,
        drop_dir_name=args.drop_dir)
    
    if not args.eval:
        try:
            while not ppo.is_done():
                ppo.learn()
        except KeyboardInterrupt:
            ppo.done() # dumps model in case it's interrupted by user
    else:
        ppo.eval(n_timesteps=args.eval_tsteps)