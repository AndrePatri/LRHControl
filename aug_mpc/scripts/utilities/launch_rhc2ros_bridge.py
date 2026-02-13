import argparse
import os


def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as exc:
        print(f"Error setting CPU affinity: {exc}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Launch shared-memory <-> ROS bridge.")
    parser.add_argument('--cores', nargs='+', type=int,
        help='List of CPU cores to set affinity to')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--ns', type=str,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--debug', action='store_true',
        help='Enable debug mode (reserved)')
    parser.add_argument('--verbose', action='store_true',
        help='Enable verbose mode')
    parser.add_argument('--ros2', action='store_true',
        help='Use ROS 2 backend')
    parser.add_argument('--is_client', action='store_true',
        help='If set, run ROS->shared bridge. Otherwise shared->ROS bridge.')
    parser.add_argument('--add_training_data', action='store_true',
        help='Also bridge training-related shared-memory blocks')

    args = parser.parse_args()

    if args.ns is None:
        raise RuntimeError('Missing required --ns argument')

    if args.cores:
        set_affinity(args.cores)

    backend = 'ros2' if args.ros2 else 'ros1'

    if args.is_client:
        from aug_mpc.utils.bridges.ros_to_shared_bridge import RosToSharedMemBridge
        bridge = RosToSharedMemBridge(
            namespace=args.ns,
            backend=backend,
            add_training_data=args.add_training_data,
            verbose=args.verbose,
        )
    else:
        from aug_mpc.utils.bridges.shared_to_ros_bridge import SharedMemToRosBridge
        bridge = SharedMemToRosBridge(
            namespace=args.ns,
            backend=backend,
            add_training_data=args.add_training_data,
            verbose=args.verbose,
        )

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
