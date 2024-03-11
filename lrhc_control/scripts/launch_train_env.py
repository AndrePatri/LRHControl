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

    args = parser.parse_args()
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)
    
    env = LRhcHeightChange(namespace=args.ns,
                    verbose=True,
                    vlevel=VLevel.V2)

    ppo = CleanPPO(env=env, debug=True)
    time_id = datetime.now().strftime('%Y%m%d%H%M%S')
    ppo.setup(run_name=args.run_name + time_id, 
        verbose=True,
        drop_dir_name=args.drop_dir)
    
    try:
        while not ppo.is_done():
            ppo.learn()
    except KeyboardInterrupt:
        ppo.done() # dumps model in case it's interrupted by user