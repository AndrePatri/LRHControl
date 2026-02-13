from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.to_ros import *
from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import *
from EigenIPC.PyEigenIPC import *

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from mpc_hive.utilities.shared_data.rhc_data import RobotState
from mpc_hive.utilities.shared_data.rhc_data import RhcRefs
from mpc_hive.utilities.shared_data.rhc_data import RhcCmds
from mpc_hive.utilities.shared_data.rhc_data import RhcStatus
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

from typing import List

class SharedMemToRosBridge():

    def __init__(self,
            namespace: str,
            backend: str = "ros2"):

        self._namespace = namespace
        self._backend = backend

        self._bridges = []
        self._clients = []
        self._shared_mems = [] # List of lists

        self._dt = 0.05

        self._is_running = False

    def _init_clients(self):
        self._clients.append(RobotState(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(RhcRefs(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(RhcCmds(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(RhcStatus(namespace=self._namespace, 
                                is_server=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(SharedEnvInfo(namespace=self._namespace, 
                                is_server=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(AgentRefs(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(Observations(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(NextObservations(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(TotRewards(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(SubRewards(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(Actions(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(Terminations(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(Truncations(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(EpisodesCounter(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        self._clients.append(TaskRandCounter(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
        # self._clients.append(SharedTrainingEnvInfo(namespace=self._namespace, 
        #                         is_server=False, 
        #                         verbose=True, 
        #                         vlevel=VLevel.V1))
        
    def _run_clients(self):
        for client in self._clients:
            client.run()
            shared_mems = []
            shared_mem = client.get_shared_mem() # this method is must be available for all clients 
            # (it must return either a single Client, a List of Clients or None)
            if not isinstance(shared_mem, List): #
                shared_mems.append(shared_mem)
            else: # we assume that in all other cases it will be a List of clients
                for mem in shared_mem:
                    if mem is not None:
                        shared_mems.append(mem)
            self._shared_mems.append(shared_mems)

    def _close_clients(self):
        for client in self._clients:
            client.close()

    def _close_bridges(self):
        for bridge in self._bridges:
            bridge.close()

    def _init_toROS_bridges(self):

        if self._backend == "ros1":
            import rospy
            node = rospy.init_node("Sharsor2RosBridge_" + self._namespace)
            for i in range(len(self._shared_mems)):
                shared_mems_client = self._shared_mems[i]
                for j in range(len(shared_mems_client)):
                    self._bridges.append(ToRos(client=shared_mems_client[j],
                    queue_size = 1,
                    ros_backend = self._backend))
        elif self._backend == "ros2":
            import rclpy
            rclpy.init()
            node = rclpy.create_node("Sharsor2RosBridge_" + self._namespace)
            for i in range(len(self._shared_mems)):
                shared_mems_client = self._shared_mems[i]
                for j in range(len(shared_mems_client)):
                    self._bridges.append(ToRos(client=shared_mems_client[j],
                    queue_size = 1,
                    ros_backend = self._backend,
                    node=node))
        else:
            Journal.log(self.__class__.__name__,
                "_init_toROS_bridges",
                f"backend {self._backend} not supported!",
                LogType.EXCEP,
                throw_when_excep = True)
        
        for bridge in self._bridges:
            bridge.run()

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._init_clients()

        self._run_clients()

        self._init_toROS_bridges()

        self._is_running = True

        self._run()

    def _run(self):

        info = f": starting sharsor-to-ROS bridge with update dt {self._dt} s" + \
            f" with namespace {self._namespace}"
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        start_time = 0.0
        elapsed_time = 0.0
        time_to_sleep_ns = 0

        while self._is_running:
            try:
                start_time = time.perf_counter() 
                self._update()
                elapsed_time = time.perf_counter() - start_time
                time_to_sleep_ns = int((self._dt - elapsed_time) * 1000000000) # [ns]
                if time_to_sleep_ns < 0:
                    warning = f": Could not match desired update dt of {self._dt} s. " + \
                        f"Elapsed time to update {elapsed_time}."
                    Journal.log(self.__class__.__name__,
                        "run",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)
                else:
                    PerfSleep.thread_sleep(time_to_sleep_ns) 
                continue
            except KeyboardInterrupt:
                self.close()

    def _update(self):

        for bridge in self._bridges:
            bridge.update()

    def close(self):

        self._close_clients()
        self._close_bridges()

        self._is_running = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Multi Robot Visualizer")
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--dt', type=float, default=0.01, help='Update interval in seconds, default is 0.01')

    args = parser.parse_args()
    
    backend = "ros2" if args.ros2 else "ros1"

    if args.ns is None:
        Journal.log("ros_bridge.py",
                "ros_bridge",
                "no --ns argument provided!",
                LogType.EXCEP,
                throw_when_excep = True)
    bridge = SharedMemToRosBridge(namespace=args.ns,
                    backend=backend)

    bridge.run(dt=args.dt)

    bridge.close()