import argparse

from mpc_hive.utilities.bridges.zmq.zmq_to_shared_bridge import ZmqToSharedMemBridge as _BaseZmqToSharedMemBridge


class ZmqToSharedMemBridge(_BaseZmqToSharedMemBridge):
    pass


ZmqToSharedMem = ZmqToSharedMemBridge


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ZMQ to shared-memory bridge")
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--remap_ns', type=str, default=None,
        help='Namespace used when creating destination shared-memory servers')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--source_ip', type=str, default='127.0.0.1',
        help='Sender IP used to derive stream endpoints')
    parser.add_argument('--queue_size', type=int, default=1,
        help='ZMQ subscriber queue size (HWM)')
    parser.add_argument('--timeout_ms', type=int, default=0,
        help='ZMQ poll timeout in ms for each bridge update')
    parser.add_argument('--no_conflate', action='store_true',
        help='Disable latest-only behavior')
    parser.add_argument('--port_base', type=int, default=20000,
        help='Base port used by deterministic endpoint mapping')
    parser.add_argument('--port_span', type=int, default=40000,
        help='Port span used by deterministic endpoint mapping')

    args = parser.parse_args()

    bridge = ZmqToSharedMemBridge(
        namespace=args.ns,
        remap_ns=args.remap_ns,
        queue_size=args.queue_size,
        conflate=not args.no_conflate,
        timeout_ms=args.timeout_ms,
        source_ip=args.source_ip,
        port_base=args.port_base,
        port_span=args.port_span,
    )

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
