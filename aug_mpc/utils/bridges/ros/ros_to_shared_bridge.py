import argparse

from mpc_hive.utilities.bridges.ros.ros_to_shared_bridge import RosToSharedMemBridge as _BaseRosToSharedMemBridge


class RosToSharedMemBridge(_BaseRosToSharedMemBridge):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ROS to shared-memory bridge")
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--remap_ns', type=str, default=None,
        help='Namespace to be used for remapping when creating shared memory servers')
    parser.add_argument('--add_training_data', action='store_true',
        help='Reserved, kept for CLI compatibility')

    args = parser.parse_args()

    backend = "ros2" if args.ros2 else "ros1"

    bridge = RosToSharedMemBridge(
        namespace=args.ns,
        backend=backend,
        add_training_data=args.add_training_data,
        remap_ns=args.remap_ns,
    )

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
