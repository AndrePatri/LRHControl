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
    parser.add_argument('--robot_pkg_name', type=str, help='Robot description package name')
    parser.add_argument('--robot_pkg_pref_path', type=str, help='base path to where each robot decription package is located')
    parser.add_argument('--size', type=int, help='cluster size')
    parser.add_argument('--open_loop', action='store_true', help='whether use RHC controllers in open loop mode')
    parser.add_argument('--verbose', action='store_true', help='run in verbose mode')
    parser.add_argument('--enable_debug', action='store_true', help='enable debug mode for cluster client and all controllers')
    parser.add_argument('--dmpdir', type=str, help='directory where data is dumped',default="/root/aux_data")

    parser.add_argument('--force_cores', action='store_true', help='whether to force RHC controller affinity')
    parser.add_argument('--i_cores_only', action='store_true', help='whether use isolated cores only for RHC controllers')
    parser.add_argument('--set_rhc_affinity', action='store_true', help='whether to set the affinity of each rhc controller to a specific core')
    parser.add_argument('--mp_fork', action='store_true', help='whether to mutliprocess fork context')
    parser.add_argument('--c_start_idx', type=int, 
            help='start index for cores over which RHC controllers will be distributed',
            default=0)
    parser.add_argument('--c_end_idx', type=int, 
            help='end index for cores over which RHC controllers will be distributed',
            default=mp.cpu_count())
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--timeout_ms', type=int, help='connection timeout after which the script self-terminates', default=60000)
    parser.add_argument('--codegen_override_dir', type=str, help='Path to base dir where codegen is to be loaded',default="")

    args = parser.parse_args()
    
    if args.mp_fork: # this needs to be in the main
        mp.set_start_method('fork')
    else:
        mp.set_start_method('spawn')
        # mp.set_start_method('forkserver')
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)

    namespace = args.ns
    cluster_size = args.size
    core_ids_override_list = None
    if args.force_cores:
        core_ids_override_list = list(range(args.c_start_idx, args.c_end_idx + 1))

    control_cluster_client = HybridQuadrupedClusterClient(namespace=namespace, 
                                        robot_pkg_name=args.robot_pkg_name,
                                        robot_pkg_pref_path=args.robot_pkg_pref_path,
                                        cluster_size=cluster_size,
                                        open_loop = args.open_loop,
                                        set_affinity = args.set_rhc_affinity,
                                        use_mp_fork = args.mp_fork,
                                        isolated_cores_only = args.i_cores_only, 
                                        core_ids_override_list = core_ids_override_list,
                                        verbose=args.verbose,
                                        debug=args.enable_debug,
                                        base_dump_dir=args.dmpdir,
                                        timeout_ms=args.timeout_ms,
                                        codegen_override=args.codegen_override_dir) # this blocks until connection with the client is established
    control_cluster_client.run() # spawns the controllers on separate processes


