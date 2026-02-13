from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.to_ros import ToRos
from EigenIPC.PyEigenIPC import VLevel, LogType, Journal

from mpc_hive.utilities.shared_data.rhc_data import RobotState, RhcRefs, RhcCmds, RhcStatus
from mpc_hive.utilities.shared_data.rhc_data import RhcPred, RhcPredDelta
from mpc_hive.utilities.shared_data.rhc_data import RhcInternal
from mpc_hive.utilities.shared_data.cluster_profiling import RhcProfiling

from mpc_hive.utilities.shared_data.sim_data import SharedEnvInfo

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


class SharedMemToRosBridge:

    def __init__(self,
            namespace: str,
            backend: str = "ros2",
            add_training_data: bool = False,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V1,
            queue_size: int = 1):

        self._namespace = namespace
        self._backend = backend
        self._add_training_data = add_training_data
        self._verbose = verbose
        self._vlevel = vlevel
        self._queue_size = queue_size

        self._bridges = []
        self._clients = []
        self._shared_mems = []

        self._dt = 0.05
        self._is_running = False
        self._node = None

        self._check_backend()

    def _check_backend(self):

        if self._backend not in ("ros1", "ros2"):
            Journal.log(self.__class__.__name__,
                "_check_backend",
                f"backend {self._backend} not supported!",
                LogType.EXCEP,
                throw_when_excep=True)

    def _backend_alive(self):

        if self._backend == "ros1":
            import rospy
            return not rospy.is_shutdown()

        if self._backend == "ros2":
            import rclpy
            return rclpy.ok()

        return True

    def _shutdown_backend(self):

        if self._backend == "ros1":
            try:
                import rospy
                if not rospy.is_shutdown():
                    rospy.signal_shutdown("bridge close requested")
            except Exception:
                pass

        if self._backend == "ros2":
            try:
                import rclpy
                if self._node is not None:
                    self._node.destroy_node()
                    self._node = None
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def _init_clients(self):

        self._clients = [
            RhcStatus(namespace=self._namespace,
                is_server=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RobotState(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcRefs(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcCmds(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            # RhcProfiling(name=self._namespace,
            #     is_server=False,
            #     safe=False,
            #     verbose=self._verbose,
            #     vlevel=self._vlevel),
            # SharedEnvInfo(namespace=self._namespace,
            #     is_server=False,
            #     verbose=self._verbose,
            #     vlevel=self._vlevel)
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

    def _init_to_ros_bridges(self):

        if self._backend == "ros1":
            import rospy
            rospy.init_node("SharedMem2RosBridge_" + self._namespace)
        elif self._backend == "ros2":
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("SharedMem2RosBridge_" + self._namespace)

        self._bridges = []
        for shared_mem in self._shared_mems:
            if self._backend == "ros1":
                bridge = ToRos(client=shared_mem,
                    queue_size=self._queue_size,
                    ros_backend=self._backend)
            else:
                bridge = ToRos(client=shared_mem,
                    queue_size=self._queue_size,
                    ros_backend=self._backend,
                    node=self._node)
            bridge.run()
            self._bridges.append(bridge)

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._init_clients()
        self._run_clients()
        self._init_to_ros_bridges()

        self._is_running = True
        self._run_loop()

    def _run_loop(self):

        info = f"starting shared memory-to-ROS bridge with update dt {self._dt} s" + \
            f" and namespace {self._namespace} ({self._backend})"
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep=True)

        while self._is_running and self._backend_alive():
            try:
                start_time = time.perf_counter()
                self._update()
                elapsed_time = time.perf_counter() - start_time
                time_to_sleep = self._dt - elapsed_time
                if time_to_sleep < 0:
                    Journal.log(self.__class__.__name__,
                        "run",
                        f"Could not match desired update dt of {self._dt} s. Elapsed {elapsed_time} s.",
                        LogType.WARN,
                        throw_when_excep=True)
                else:
                    time.sleep(time_to_sleep)
            except (KeyboardInterrupt, SystemExit):
                break

        self.close()

    def _update(self):

        for bridge in self._bridges:
            bridge.update()

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and len(self._clients) == 0:
            return

        self._is_running = False
        self._close_bridges()
        self._close_clients()
        self._shutdown_backend()
        self._bridges = []
        self._clients = []
        self._shared_mems = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Shared-memory to ROS bridge")
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--add_training_data', action='store_true',
        help='Also bridge training-related shared-memory blocks')

    args = parser.parse_args()

    backend = "ros2" if args.ros2 else "ros1"

    bridge = SharedMemToRosBridge(namespace=args.ns,
                    backend=backend,
                    add_training_data=args.add_training_data)

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
