from EigenIPC.PyEigenIPC import VLevel, LogType, Journal
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.from_zmq import FromZmq
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.abstractions import default_endpoint

from mpc_hive.utilities.shared_data.rhc_data import RobotState, RhcRefs, RhcCmds, RhcStatus
from mpc_hive.utilities.shared_data.cluster_profiling import RhcProfiling

from aug_mpc.utils.shared_data.agent_refs import AgentRefs
from aug_mpc.utils.shared_data.training_env import SharedTrainingEnvInfo
from aug_mpc.utils.shared_data.training_env import Observations, NextObservations
from aug_mpc.utils.shared_data.training_env import TotRewards
from aug_mpc.utils.shared_data.training_env import SubRewards
from aug_mpc.utils.shared_data.training_env import Actions
from aug_mpc.utils.shared_data.training_env import Terminations
from aug_mpc.utils.shared_data.training_env import Truncations
from aug_mpc.utils.shared_data.training_env import EpisodesCounter, TaskRandCounter

import argparse
import time

from perf_sleep.pyperfsleep import PerfSleep


class ZmqToSharedMemBridge:

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
        self._remap_ns = self._namespace if remap_ns is None else remap_ns

        self._bridges = []
        self._template_clients = []
        self._bridge_specs = []

        self._dt = 0.05
        self._is_running = False

    def _init_template_clients(self):

        self._template_clients = [
            RhcStatus(namespace=self._namespace,
                is_server=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            # RobotState(namespace=self._namespace,
            #     is_server=False,
            #     safe=False,
            #     verbose=self._verbose,
            #     vlevel=self._vlevel),
            # RhcRefs(namespace=self._namespace,
            #     is_server=False,
            #     safe=False,
            #     verbose=self._verbose,
            #     vlevel=self._vlevel),
            # RhcCmds(namespace=self._namespace,
            #     is_server=False,
            #     safe=False,
            #     verbose=self._verbose,
            #     vlevel=self._vlevel),
            # RhcProfiling(name=self._namespace,
            #     is_server=False,
            #     safe=False,
            #     verbose=self._verbose,
            #     vlevel=self._vlevel),
        ]

        if self._add_training_data:
            self._template_clients.extend([
                AgentRefs(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                Observations(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                NextObservations(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                TotRewards(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                SubRewards(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                Actions(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                Terminations(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                Truncations(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                EpisodesCounter(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                TaskRandCounter(namespace=self._namespace,
                    is_server=False,
                    safe=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel),
                SharedTrainingEnvInfo(namespace=self._namespace,
                    is_server=False,
                    verbose=self._verbose,
                    vlevel=self._vlevel)
            ])

    def _as_mem_list(self, shared_mem):

        if shared_mem is None:
            return []
        if isinstance(shared_mem, list):
            return [mem for mem in shared_mem if mem is not None]
        return [shared_mem]

    def _collect_bridge_specs(self):

        endpoint_set = set()
        self._bridge_specs = []

        for client in self._template_clients:
            for mem in self._as_mem_list(client.get_shared_mem()):
                basename = mem.getBasename()
                namespace = mem.getNamespace()

                endpoint = default_endpoint(
                    namespace=namespace,
                    basename=basename,
                    ip=self._source_ip,
                    port_base=self._port_base,
                    port_span=self._port_span,
                )

                stream_key = (basename, namespace, endpoint)
                if stream_key in endpoint_set:
                    continue

                endpoint_set.add(stream_key)
                self._bridge_specs.append(stream_key)

    def _close_template_clients(self):

        for client in self._template_clients:
            try:
                client.close()
            except Exception:
                pass

    def _close_bridges(self):

        for bridge in self._bridges:
            try:
                bridge.close()
            except Exception:
                pass

    def _init_from_zmq_bridges(self):

        self._bridges = []
        for basename, namespace, endpoint in self._bridge_specs:
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
                remap_ns=self._remap_ns,
            )
            self._bridges.append(bridge)

        self._run_bridges_until_ready()

    def _bridge_id(self, bridge):

        basename = getattr(bridge, "_basename", "unknown_basename")
        namespace = getattr(bridge, "_namespace", "unknown_namespace")
        endpoint = getattr(bridge, "_endpoint", "unknown_endpoint")
        return f"{namespace}/{basename} ({endpoint})"

    def _run_bridges_until_ready(self):

        pending = list(self._bridges)
        warn_counter = 0

        while len(pending) > 0 and self._is_running:
            next_pending = []
            for bridge in pending:
                if not bridge.run():
                    next_pending.append(bridge)

            pending = next_pending

            if len(pending) > 0:
                if warn_counter % 20 == 0:
                    pending_ids = ", ".join([self._bridge_id(bridge) for bridge in pending])
                    Journal.log(self.__class__.__name__,
                        "_run_bridges_until_ready",
                        f"waiting for first ZMQ packet on {len(pending)} bridge(s): {pending_ids}",
                        LogType.WARN,
                        throw_when_excep=True)
                warn_counter += 1
                time.sleep(0.05)

        if len(pending) > 0:
            pending_ids = ", ".join([self._bridge_id(bridge) for bridge in pending])
            Journal.log(self.__class__.__name__,
                "_run_bridges_until_ready",
                f"failed to initialize {len(pending)} bridge(s): {pending_ids}",
                LogType.WARN,
                throw_when_excep=True)

    def run(self, dt: float = 0.05):

        self._dt = dt
        self._is_running = True

        self._init_template_clients()
        self._collect_bridge_specs()
        self._init_from_zmq_bridges()

        if len(self._bridges) == 0:
            Journal.log(self.__class__.__name__,
                "run",
                "no streams discovered to bridge",
                LogType.WARN,
                throw_when_excep=True)
            self.close()
            return

        self._run_loop()

    def _run_loop(self):

        info = (
            f"starting ZMQ-to-shared-memory bridge with update dt {self._dt} s "
            f"and namespace {self._namespace}, remapped to {self._remap_ns}"
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

        for bridge in self._bridges:
            bridge.update(retry_write=False)

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and len(self._template_clients) == 0:
            return

        self._is_running = False
        self._close_bridges()
        self._close_template_clients()
        self._bridges = []
        self._template_clients = []
        self._bridge_specs = []


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
        help='Also bridge training-related shared-memory blocks')

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
