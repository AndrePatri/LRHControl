from EigenIPC.PyEigenIPC import VLevel, LogType, Journal
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.from_zmq import FromZmq
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.abstractions import default_endpoint, ZmqSubscriber
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.defs import MSG_DATA, is_string_tensor

import argparse
import time

from perf_sleep.pyperfsleep import PerfSleep


class ZmqToSharedMemBridge:

    _CATALOG_BASENAME = "ZmqBridgeCatalog"

    def __init__(self,
            namespace: str,
            add_training_data: bool = False,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1,
            conflate: bool = True,
            timeout_ms: int = 0,
            connect: bool = True,
            source_ip: str = "127.0.0.1",
            port_base: int = 20000,
            port_span: int = 40000,
            force_reconnection: bool = True,
            remap_ns: str = None):

        self._namespace = namespace
        self._add_training_data = add_training_data
        self._verbose = verbose
        self._vlevel = vlevel
        self._queue_size = queue_size
        self._conflate = conflate
        self._timeout_ms = timeout_ms
        self._connect = connect
        self._source_ip = source_ip
        self._port_base = port_base
        self._port_span = port_span
        self._force_reconnection = force_reconnection
        self._remap_ns = remap_ns

        self._bridges = {}
        self._catalog_subscriber = None

        self._dt = 0.05
        self._is_running = False

    def _catalog_endpoint(self):

        return default_endpoint(
            namespace=self._namespace,
            basename=self._CATALOG_BASENAME,
            ip=self._source_ip,
            port_base=self._port_base,
            port_span=self._port_span,
        )

    def _init_catalog_subscriber(self):

        endpoint = self._catalog_endpoint()
        self._catalog_subscriber = ZmqSubscriber(
            endpoint=endpoint,
            connect=self._connect,
            queue_size=self._queue_size,
            conflate=self._conflate,
            timeout_ms=self._timeout_ms,
        )
        self._catalog_subscriber.run()

        Journal.log(self.__class__.__name__,
            "_init_catalog_subscriber",
            f"subscribed to catalog on {endpoint}",
            LogType.INFO,
            throw_when_excep=True)

    def _decode_string_tensor(self, np_data):

        decoded = []

        for col_idx in range(np_data.shape[1]):
            raw = bytearray()
            terminated = False

            for row_idx in range(np_data.shape[0]):
                value = int(np_data[row_idx, col_idx])
                for byte_idx in range(4):
                    byte = (value >> (8 * byte_idx)) & 0xFF
                    if byte == 0:
                        terminated = True
                        break
                    raw.append(byte)
                if terminated:
                    break

            decoded.append(raw.decode("utf-8", errors="ignore"))

        return decoded

    def _parse_catalog_specs(self, entries):

        stream_specs = []
        seen = set()

        for entry in entries:
            if entry is None:
                continue

            clean = entry.strip()
            if clean == "":
                continue

            if "|" not in clean:
                Journal.log(self.__class__.__name__,
                    "_parse_catalog_specs",
                    f"ignoring malformed catalog entry '{clean}'",
                    LogType.WARN,
                    throw_when_excep=True)
                continue

            namespace, basename = clean.split("|", 1)
            if basename == "":
                continue

            key = (basename, namespace)
            if key in seen:
                continue
            seen.add(key)

            endpoint = default_endpoint(
                namespace=namespace,
                basename=basename,
                ip=self._source_ip,
                port_base=self._port_base,
                port_span=self._port_span,
            )

            stream_specs.append((basename, namespace, endpoint))

        return stream_specs

    def _read_catalog_once(self):

        if self._catalog_subscriber is None:
            return None

        header, payload = self._catalog_subscriber.recv_latest()
        if header is None:
            return None

        if header.msg_type != MSG_DATA:
            return None

        if not is_string_tensor(header.flags):
            Journal.log(self.__class__.__name__,
                "_read_catalog_once",
                "received non-string catalog payload, ignoring",
                LogType.WARN,
                throw_when_excep=True)
            return None

        np_data = self._catalog_subscriber.payload_to_numpy(header, payload, copy=True)
        entries = self._decode_string_tensor(np_data)

        return self._parse_catalog_specs(entries)

    def _remap_namespace(self, namespace):

        if self._remap_ns is None:
            return namespace

        return namespace.replace(self._namespace, self._remap_ns)

    def _ensure_bridges_from_catalog(self, stream_specs):

        for basename, namespace, endpoint in stream_specs:
            key = (basename, namespace)
            if key in self._bridges:
                continue

            remap_ns_to = self._remap_namespace(namespace)
            bridge = FromZmq(
                basename=basename,
                namespace=namespace,
                endpoint=endpoint,
                connect=self._connect,
                queue_size=self._queue_size,
                conflate=self._conflate,
                timeout_ms=self._timeout_ms,
                verbose=self._verbose,
                vlevel=self._vlevel,
                force_reconnection=self._force_reconnection,
                remap_ns=remap_ns_to,
            )

            self._bridges[key] = bridge

            Journal.log(self.__class__.__name__,
                "_ensure_bridges_from_catalog",
                f"discovered stream {namespace}/{basename} ({endpoint})",
                LogType.INFO,
                throw_when_excep=True)

    def _refresh_catalog(self):

        stream_specs = self._read_catalog_once()
        if stream_specs is None:
            return False

        self._ensure_bridges_from_catalog(stream_specs)

        return True

    def run(self, dt: float = 0.05):

        self._dt = dt
        self._is_running = True

        self._init_catalog_subscriber()

        warn_counter = 0
        while self._is_running and len(self._bridges) == 0:
            got_catalog = self._refresh_catalog()
            if got_catalog:
                break

            if warn_counter % 20 == 0:
                Journal.log(self.__class__.__name__,
                    "run",
                    "waiting for catalog stream to discover bridges",
                    LogType.WARN,
                    throw_when_excep=True)
            warn_counter += 1
            time.sleep(0.05)

        self._run_loop()

    def _run_loop(self):

        info = (
            f"starting ZMQ-to-shared-memory bridge with update dt {self._dt} s "
            f"and namespace {self._namespace}, base remapped to {self._remap_ns}"
        )
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep=True)

        while self._is_running:
            try:
                start_time = time.perf_counter()
                self._update()
                elapsed_time = time.perf_counter() - start_time
                time_to_sleep_ns = int((self._dt - elapsed_time) * 1e9)
                if time_to_sleep_ns < 0:
                    Journal.log(self.__class__.__name__,
                        "run",
                        f"Could not match desired update dt of {self._dt} s. Elapsed {elapsed_time} s.",
                        LogType.WARN,
                        throw_when_excep=True)
                else:
                    PerfSleep.thread_sleep(time_to_sleep_ns)
            except (KeyboardInterrupt, SystemExit):
                break

        self.close()

    def _update(self):

        self._refresh_catalog()

        for bridge in self._bridges.values():
            if not bridge.run():
                continue

            bridge.update(retry_write=False)

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and self._catalog_subscriber is None:
            return

        self._is_running = False

        for bridge in self._bridges.values():
            try:
                bridge.close()
            except Exception:
                pass

        self._bridges = {}

        if self._catalog_subscriber is not None:
            try:
                self._catalog_subscriber.close()
            except Exception:
                pass
            self._catalog_subscriber = None


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
    parser.add_argument('--add_training_data', action='store_true',
        help='Reserved, kept for compatibility')

    args = parser.parse_args()

    bridge = ZmqToSharedMemBridge(
        namespace=args.ns,
        remap_ns=args.remap_ns,
        add_training_data=args.add_training_data,
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
