from lrhc_control.controllers.rhc.hybrid_quad_client import HybridQuadrupedClusterClient

import os
import argparse
import multiprocessing as mp

this_script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

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
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set THIS script affinity to')
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--rob_pkg', type=str, help='Robot description package name')
    parser.add_argument('--size', type=int, help='cluster size')
    parser.add_argument('--v', action='store_true', help='run in verbose mode')
    parser.add_argument('--force_cores', action='store_true', help='whether to force RHC controller affinity')
    parser.add_argument('--open_loop', type=bool, action='store_true', help='whether use RHC controllers in open loop mode',
                default=False)
    parser.add_argument('--i_cores_only', type=bool, action='store_true', help='whether use isolated cores only for RHC controllers',
                default=False)
    parser.add_argument('--c_start_idx', type=int, 
            help='start index for cores over which RHC controllers will be distributed',
            default=0)
    parser.add_argument('--c_end_idx', type=int, 
            help='end index for cores over which RHC controllers will be distributed',
            default=mp.cpu_count())

    args = parser.parse_args()

    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)
    namespace = args.ns
    cluster_size = args.size
    verbose = False
    if args.v:
        verbose = True
    core_ids_override_list = None
    if args.force_cores:
        core_ids_override_list = list(range(args.c_start_idx, args.c_end_idx + 1))
                
    control_cluster_client = HybridQuadrupedClusterClient(namespace=namespace, 
                                        robot_pkg_name=args.rob_pkg,
                                        cluster_size=cluster_size,
                                        open_loop = args.open_loop,
                                        isolated_cores_only = args.i_cores_only, 
                                        use_only_physical_cores = False,
                                        core_ids_override_list = core_ids_override_list,
                                        verbose=verbose) # this blocks until connection with the client is established

    control_cluster_client.pre_init() # pre-initialization steps
        
    control_cluster_client.run() # spawns the controllers on separate processes


