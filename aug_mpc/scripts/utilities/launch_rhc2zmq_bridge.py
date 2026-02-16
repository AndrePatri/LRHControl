import argparse
from mpc_hive.utilities.sysutils import set_process_affinity, parse_env_slice


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Launch shared-memory <-> ZMQ bridge.")
    parser.add_argument('--cores', nargs='+', type=str,
        help='CPU cores to set affinity (examples: "2 3 4", "2-5", "2,4,6")')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--ns', type=str,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--remap_ns', type=str,
        help='Namespace used for destination shared-memory servers (only when --is_client is set)')
    parser.add_argument('--verbose', action='store_true',
        help='Enable verbose mode')
    parser.add_argument('--is_client', action='store_true',
        help='If set, run ZMQ->shared bridge. Otherwise shared->ZMQ bridge.')
    parser.add_argument('--add_training_data', action='store_true',
        help='Also bridge training-related shared-memory blocks')
    parser.add_argument('--env_idx', type=str, default=None,
        help='Optional env index or inclusive range (examples: "67", "67-75"). Sender mode only.')

    parser.add_argument('--queue_size', type=int, default=1,
        help='ZMQ socket queue size (HWM)')
    parser.add_argument('--no_conflate', action='store_true',
        help='Disable latest-only behavior')
    parser.add_argument('--port_base', type=int, default=20000,
        help='Base port used by deterministic endpoint mapping')
    parser.add_argument('--port_span', type=int, default=40000,
        help='Port span used by deterministic endpoint mapping')

    parser.add_argument('--bind_ip', type=str, default='0.0.0.0',
        help='Publisher bind IP/interface (shared->ZMQ mode)')
    parser.add_argument('--source_ip', type=str, default='127.0.0.1',
        help='Sender IP to connect to (ZMQ->shared mode)')
    parser.add_argument('--timeout_ms', type=int, default=0,
        help='Subscriber timeout in ms (ZMQ->shared mode)')
    parser.add_argument('--no_force_reconnection', action='store_true',
        help='Disable force_reconnection when creating destination shared-memory servers')

    args = parser.parse_args()

    if args.ns is None:
        raise RuntimeError('Missing required --ns argument')

    if args.cores:
        selected = set_process_affinity(args.cores)
        print(f"Set CPU affinity to cores: {selected}")

    conflate = not args.no_conflate
    env_start = None
    env_count = 1
    if args.env_idx is not None:
        if args.is_client:
            raise RuntimeError('--env_idx can only be used when running sender mode (without --is_client)')
        env_start, env_count = parse_env_slice(args.env_idx)

    if args.is_client:
        from aug_mpc.utils.bridges.zmq.zmq_to_shared_bridge import ZmqToSharedMemBridge
        bridge = ZmqToSharedMemBridge(
            namespace=args.ns,
            add_training_data=args.add_training_data,
            verbose=args.verbose,
            remap_ns=args.remap_ns,
            queue_size=args.queue_size,
            conflate=conflate,
            timeout_ms=args.timeout_ms,
            source_ip=args.source_ip,
            port_base=args.port_base,
            port_span=args.port_span,
            force_reconnection=not args.no_force_reconnection,
        )
    else:
        from aug_mpc.utils.bridges.zmq.shared_to_zmq_bridge import SharedMemToZmqBridge
        bridge = SharedMemToZmqBridge(
            namespace=args.ns,
            add_training_data=args.add_training_data,
            verbose=args.verbose,
            env_idx=env_start,
            env_count=env_count,
            queue_size=args.queue_size,
            conflate=conflate,
            bind_ip=args.bind_ip,
            port_base=args.port_base,
            port_span=args.port_span,
        )

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
