import argparse
import os

# Function to set CPU affinity
def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

if __name__ == '__main__':

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set affinity to')
    parser.add_argument('--dt', type=float, default=0.01, help='Update interval in seconds, default is 0.01')
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, default is False')
    parser.add_argument('--verbose', type=bool, default=True, help='Enable verbose mode, default is True')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')

    args = parser.parse_args()

    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)

    # Use the provided robot name and update interval
    update_dt = args.dt
    debug = args.debug
    verbose = args.verbose

    bridge = None
    if not args.ros2:

        from lrhc_control.utils.rhc_viz.rhc2viz import RhcToVizBridge

        bridge = RhcToVizBridge(namespace=args.ns, 
                        verbose=verbose,
                        rhcviz_basename="RHCViz", 
                        robot_selector=[0, None])
    else:

        from lrhc_control.utils.rhc_viz.rhc2viz2 import RhcToViz2Bridge

        bridge = RhcToViz2Bridge(namespace=args.ns, 
                        verbose=verbose,
                        rhcviz_basename="RHCViz", 
                        robot_selector=[0, None])

    bridge.run(update_dt=update_dt)
