from EigenIPC.PyEigenIPC import VLevel, LogType, Journal
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.to_zmq import ToZmq
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


class SharedMemToZmqBridge:

    def __init__(self,
            namespace: str,
            add_training_data: bool = False,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1,
            conflate: bool = True,
            bind: bool = True,
            bind_ip: str = "0.0.0.0",
            port_base: int = 20000,
            port_span: int = 40000):

        self._namespace = namespace
        self._add_training_data = add_training_data
        self._verbose = verbose
        self._vlevel = vlevel
        self._queue_size = queue_size
        self._conflate = conflate
        self._bind = bind
        self._bind_ip = bind_ip
        self._port_base = port_base
        self._port_span = port_span

        self._bridges = []
        self._clients = []
        self._shared_mems = []

        self._dt = 0.05
        self._is_running = False

    def _init_clients(self):

        self._clients = [
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
            self._clients.extend([
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

    def _run_clients(self):

        self._shared_mems = []
        for client in self._clients:
            client.run()
            self._shared_mems.extend(self._as_mem_list(client.get_shared_mem()))

    def _close_clients(self):

        for client in self._clients:
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

    def _init_to_zmq_bridges(self):

        self._bridges = []
        for shared_mem in self._shared_mems:
            endpoint = default_endpoint(
                namespace=shared_mem.getNamespace(),
                basename=shared_mem.getBasename(),
                ip=self._bind_ip,
                port_base=self._port_base,
                port_span=self._port_span,
            )

            bridge = ToZmq(
                client=shared_mem,
                endpoint=endpoint,
                bind=self._bind,
                queue_size=self._queue_size,
                conflate=self._conflate,
            )
            bridge.run()
            self._bridges.append(bridge)

            Journal.log(self.__class__.__name__,
                "_init_to_zmq_bridges",
                f"publishing {shared_mem.getNamespace()}/{shared_mem.getBasename()} on {endpoint}",
                LogType.INFO,
                throw_when_excep=True)

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._init_clients()
        self._run_clients()
        self._init_to_zmq_bridges()

        self._is_running = True
        self._run_loop()

    def _run_loop(self):

        info = (
            f"starting shared memory-to-ZMQ bridge with update dt {self._dt} s "
            f"and namespace {self._namespace}"
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
            bridge.update(retry=False)

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and len(self._clients) == 0:
            return

        self._is_running = False
        self._close_bridges()
        self._close_clients()
        self._bridges = []
        self._clients = []
        self._shared_mems = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Shared-memory to ZMQ bridge")
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--bind_ip', type=str, default='0.0.0.0',
        help='IP/interface to bind publisher sockets on')
    parser.add_argument('--queue_size', type=int, default=1,
        help='ZMQ publisher queue size (HWM)')
    parser.add_argument('--no_conflate', action='store_true',
        help='Disable latest-only behavior')
    parser.add_argument('--port_base', type=int, default=20000,
        help='Base port used by deterministic endpoint mapping')
    parser.add_argument('--port_span', type=int, default=40000,
        help='Port span used by deterministic endpoint mapping')
    parser.add_argument('--add_training_data', action='store_true',
        help='Also bridge training-related shared-memory blocks')

    args = parser.parse_args()

    bridge = SharedMemToZmqBridge(
        namespace=args.ns,
        add_training_data=args.add_training_data,
        queue_size=args.queue_size,
        conflate=not args.no_conflate,
        bind_ip=args.bind_ip,
        port_base=args.port_base,
        port_span=args.port_span,
    )

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
