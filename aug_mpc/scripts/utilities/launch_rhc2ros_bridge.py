import argparse
from mpc_hive.utilities.sysutils import set_process_affinity, parse_env_slice


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Launch shared-memory <-> ROS bridge.")
    parser.add_argument('--cores', nargs='+', type=str,
        help='CPU cores to set affinity (examples: "2 3 4", "2-5", "2,4,6")')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--ns', type=str,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--remap_ns', type=str,
        help='Namespace to be used for remapping when creating shared memory servers (only used when --is_client is set)')
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
    parser.add_argument('--env_idx', type=str, default=None,
        help='Optional env index or inclusive range (examples: "67", "67-75"). Sender mode only.')

    args = parser.parse_args()

    if args.ns is None:
        raise RuntimeError('Missing required --ns argument')

    if args.cores:
        selected = set_process_affinity(args.cores)
        print(f"Set CPU affinity to cores: {selected}")

    env_start = None
    env_count = 1
    if args.env_idx is not None:
        if args.is_client:
            raise RuntimeError('--env_idx can only be used when running sender mode (without --is_client)')
        env_start, env_count = parse_env_slice(args.env_idx)

    backend = 'ros2' if args.ros2 else 'ros1'

    if args.is_client:
        from aug_mpc.utils.bridges.ros.ros_to_shared_bridge import RosToSharedMemBridge
        bridge = RosToSharedMemBridge(
            namespace=args.ns,
            backend=backend,
            add_training_data=args.add_training_data,
            verbose=args.verbose,
            remap_ns=args.remap_ns,
        )
    else:
        from aug_mpc.utils.bridges.ros.shared_to_ros_bridge import SharedMemToRosBridge
        bridge = SharedMemToRosBridge(
            namespace=args.ns,
            backend=backend,
            add_training_data=args.add_training_data,
            verbose=args.verbose,
            env_idx=env_start,
            env_count=env_count,
        )

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
