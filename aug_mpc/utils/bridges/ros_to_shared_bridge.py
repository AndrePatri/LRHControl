from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.from_ros import FromRos
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

from perf_sleep.pyperfsleep import PerfSleep


class RosToSharedMemBridge:

    def __init__(self,
            namespace: str,
            backend: str = "ros2",
            add_training_data: bool = False,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1,
            force_reconnection: bool = True,
            remap_ns: str = None):
        
        self._namespace = namespace
        self._backend = backend
        self._add_training_data = add_training_data
        self._verbose = verbose
        self._vlevel = vlevel
        self._queue_size = queue_size
        self._force_reconnection = force_reconnection
        if remap_ns is not None:
            raise Exception("remap_ns argument is not supported in this version of the bridge")
        
        self._remap_ns=remap_ns

        self._bridges = []
        self._template_clients = []
        self._bridge_endpoints = []

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

    def _init_template_clients(self):

        self._template_clients = [
            RobotState(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcStatus(namespace=self._namespace,
                is_server=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
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
            # SharedEnvInfo(namespace=self._namespace,
            #     is_server=False,
            #     verbose=self._verbose,
            #     vlevel=self._vlevel)
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

    def _collect_endpoints(self):

        endpoint_set = set()
        self._bridge_endpoints = []

        for client in self._template_clients:
            for mem in self._as_mem_list(client.get_shared_mem()):
                basename = mem.getBasename()
                namespace = mem.getNamespace()
                endpoint = (basename, namespace)
                if endpoint not in endpoint_set:
                    endpoint_set.add(endpoint)
                    self._bridge_endpoints.append(endpoint)

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

    def _init_from_ros_bridges(self):

        if self._backend == "ros1":
            import rospy
            rospy.init_node("Ros2SharedMemoryBridge_" + self._namespace)
        elif self._backend == "ros2":
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("Ros2SharedMemoryBridge_" + self._namespace)

        self._bridges = []
        for basename, namespace in self._bridge_endpoints:
            kwargs = dict(
                basename=basename,
                namespace=namespace,
                queue_size=self._queue_size,
                ros_backend=self._backend,
                verbose=self._verbose,
                vlevel=self._vlevel,
                force_reconnection=self._force_reconnection,
                remap_ns=self._remap_ns,
            )
            if self._backend == "ros2":
                kwargs["node"] = self._node
            self._bridges.append(FromRos(**kwargs))

        self._run_bridges_until_ready()

    def _run_bridges_until_ready(self):

        def _bridge_id(bridge):
            basename = getattr(bridge, "_basename", "unknown_basename")
            namespace = getattr(bridge, "_namespace", "unknown_namespace")
            return f"{basename}@{namespace}"

        pending = list(self._bridges)
        warn_counter = 0

        while len(pending) > 0 and self._backend_alive():
            if self._backend == "ros2":
                import rclpy
                rclpy.spin_once(self._node, timeout_sec=0.0)

            next_pending = []
            for bridge in pending:
                if not bridge.run():
                    next_pending.append(bridge)

            pending = next_pending

            if len(pending) > 0:
                if warn_counter % 20 == 0:
                    pending_ids = ", ".join([_bridge_id(bridge) for bridge in pending])
                    Journal.log(self.__class__.__name__,
                        "_run_bridges_until_ready",
                        f"waiting for ROS metadata on {len(pending)} bridge(s): {pending_ids}",
                        LogType.WARN,
                        throw_when_excep=True)
                warn_counter += 1
                time.sleep(0.05)

        if len(pending) > 0:
            pending_ids = ", ".join([_bridge_id(bridge) for bridge in pending])
            Journal.log(self.__class__.__name__,
                "_run_bridges_until_ready",
                f"failed to initialize {len(pending)} bridge(s): {pending_ids}",
                LogType.WARN,
                throw_when_excep=True)

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._init_template_clients()
        self._collect_endpoints()
        self._init_from_ros_bridges()

        self._is_running = True
        self._run_loop()

    def _run_loop(self):

        info = f"starting ROS-to-shared-memory bridge with update dt {self._dt} s" + \
            f" and namespace {self._namespace} ({self._backend}), remapped to {self._remap_ns}"
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

        if self._backend == "ros2" and self._backend_alive():
            import rclpy
            rclpy.spin_once(self._node, timeout_sec=0.0)

        for bridge in self._bridges:
            bridge.update()

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and len(self._template_clients) == 0:
            return

        self._is_running = False
        self._close_bridges()
        self._close_template_clients()
        self._shutdown_backend()
        self._bridges = []
        self._template_clients = []
        self._bridge_endpoints = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ROS to shared-memory bridge")
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--add_training_data', action='store_true',
        help='Also bridge training-related shared-memory blocks')

    args = parser.parse_args()

    backend = "ros2" if args.ros2 else "ros1"

    bridge = RosToSharedMemBridge(namespace=args.ns,
                    backend=backend,
                    add_training_data=args.add_training_data)

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
