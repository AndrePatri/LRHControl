import argparse

from EigenIPC.PyEigenIPC import VLevel

from mpc_hive.utilities.bridges.zmq.shared_to_zmq_bridge import SharedMemToZmqBridge as _BaseSharedMemToZmqBridge

from aug_mpc.utils.shared_data.agent_refs import AgentRefs
from aug_mpc.utils.shared_data.training_env import SharedTrainingEnvInfo
from aug_mpc.utils.shared_data.training_env import Observations, NextObservations
from aug_mpc.utils.shared_data.training_env import TotRewards
from aug_mpc.utils.shared_data.training_env import SubRewards
from aug_mpc.utils.shared_data.training_env import Actions
from aug_mpc.utils.shared_data.training_env import Terminations
from aug_mpc.utils.shared_data.training_env import Truncations
from aug_mpc.utils.shared_data.training_env import EpisodesCounter, TaskRandCounter


class SharedMemToZmqBridge(_BaseSharedMemToZmqBridge):

    def __init__(self,
            namespace: str,
            add_rhc_internal: bool = False,
            env_idx: int = None,
            env_count: int = 1,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1,
            conflate: bool = True,
            bind: bool = True,
            bind_ip: str = "0.0.0.0",
            port_base: int = 20000,
            port_span: int = 40000,
            add_training_data: bool = False):
        
        self._add_training_data=add_training_data

        super().__init__(
            namespace=namespace,
            add_rhc_internal=add_rhc_internal,
            env_idx=env_idx,
            env_count=env_count,
            verbose=verbose,
            vlevel=vlevel,
            queue_size=queue_size,
            conflate=conflate,
            bind=bind,
            bind_ip=bind_ip,
            port_base=port_base,
            port_span=port_span,
        )
    def _build_extra_clients(self):

        if not self._add_training_data:
            return []

        return [
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
                vlevel=self._vlevel),
        ]


SharedMemToZmq = SharedMemToZmqBridge


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
