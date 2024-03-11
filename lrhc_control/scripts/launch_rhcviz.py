#!/usr/bin/env python
from rhcviz.RHCViz import RHCViz
from rhcviz.utils.sys_utils import PathsGetter

import os
import argparse

# Function to set CPU affinity
def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Multi Robot Visualizer")
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--dpath', type=str)
    parser.add_argument('--nodes_perc', type=int, default=100)
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set 	affinity to')
    args = parser.parse_args()
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)
        
    syspaths = PathsGetter()
    
    rhcviz = RHCViz(urdf_file_path=args.dpath, 
        rviz_config_path=syspaths.DEFAULT_RVIZ_CONFIG_PATH,
        namespace=args.ns, 
        basename="RHCViz", 
        rate = 100,
        cpu_cores = [0],
        use_only_collisions=False,
        nodes_perc = args.nodes_perc       
        )
    
    rhcviz.run()
