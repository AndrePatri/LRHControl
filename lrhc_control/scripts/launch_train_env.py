from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline

from lrhc_control.training_algs.ppo.ppo import PPO
from lrhc_control.training_algs.sac.sac import SAC

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
    parser.add_argument('--n_evals', type=int, help='N. of evaluation rollouts to be performed', default=None)
    parser.add_argument('--n_timesteps', type=int, help='Toal n. of timesteps for each evaluation rollout', default=None)
    parser.add_argument('--mpath', type=str, help='Model path to be used for policy evaluation',default=None)
    parser.add_argument('--use_cpu', action='store_true', help='If set, all the training (data included) will be perfomed on CPU')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--seed', type=int, help='seed', default=1)
    parser.add_argument('--disable_db', action='store_true', help='Whether to disable local data logging for the algorithm (reward metrics, etc..)')
    parser.add_argument('--disable_env_db', action='store_true', help='Whether to disable env db data logging on \
                        shared mem (e.g.reward metrics are not available for reading anymore)')
    parser.add_argument('--disable_rmdb', action='store_true', help='Whether to disable remote debug (e.g. data logging on remote servers)')
    parser.add_argument('--override_agent_refs', action='store_true', help='Whether to override automatically generated agent refs (useful for debug)')
    parser.add_argument('--timeout_ms', type=int, help='connection timeout after which the script self-terminates', default=60000)

    args = parser.parse_args()
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)
    
    env = LinVelTrackBaseline(namespace=args.ns,
                    verbose=True,
                    vlevel=VLevel.V2,
                    use_gpu=not args.use_cpu,
                    debug=not args.disable_env_db,
                    override_agent_refs=args.override_agent_refs,
                    timeout_ms=args.timeout_ms)
    
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
    
    algo = PPO(env=env, 
            debug=not args.disable_db, remote_db=not args.disable_rmdb,
            seed=args.seed)
    # algo = SAC(env=env, 
            # debug=not args.disable_db, remote_db=not args.disable_rmdb,
            # seed=args.seed)
    algo.setup(run_name=args.run_name, 
        verbose=True,
        drop_dir_name=args.drop_dir,
        custom_args = sim_data,
        comment=args.comment,
        eval=args.eval,
        model_path=args.mpath,
        n_evals=args.n_evals,
        n_timesteps_per_eval=args.n_timesteps,
        dump_checkpoints=args.dump_checkpoints,
        norm_obs=True)

    try:
        while not algo.is_done():
            if not args.eval:
                if not algo.learn():
                    algo.done()
            else: # eval phase
                if not algo.eval():
                    algo.done()
    except KeyboardInterrupt:
        algo.done() # in case it's interrupted by user