from rhcviz.utils.handshake import RHCVizHandshake
from rhcviz.utils.namings import NamingConventions
from rhcviz.utils.string_list_encoding import StringArray

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcInternal

import numpy as np

from SharsorIPCpp.PySharsorIPC import dtype

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

import rclpy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, LivelinessPolicy
from rclpy.qos import QoSProfile

from typing import List

from perf_sleep.pyperfsleep import PerfSleep
import time 

class RhcToViz2Bridge:

    # bridge from shared mem to ROS 2
    
    def __init__(self, 
            namespace: str, 
            verbose = False,
            rhcviz_basename = "RHCViz",
            robot_selector: List = [0, None]):

        self._robot_selector = robot_selector

        self.verbose = verbose

        self.namespace = namespace # defines uniquely the kind of controller 
        # (associated with a specific robot)

        # ros stuff
        self.ros_names = NamingConventions() # rhcviz naming conventions
        self.rhcviz_basename = rhcviz_basename

        rclpy.init()
        self.node = rclpy.create_node(rhcviz_basename + "_" + namespace)
        self._qos_settings = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE, # BEST_EFFORT
                durability=DurabilityPolicy.TRANSIENT_LOCAL, # VOLATILE
                history=HistoryPolicy.KEEP_LAST, # KEEP_ALL
                depth=10,  # Number of samples to keep if KEEP_LAST is used
                liveliness=LivelinessPolicy.AUTOMATIC,
                # deadline=1000000000,  # [ns]
                # partition='my_partition' # useful to isolate communications
                )

        self.handshaker = RHCVizHandshake(handshake_topic=self.ros_names.handshake_topicname(basename=self.rhcviz_basename, 
                                                                                        namespace=self.namespace),
                                        node=self.node,
                                        is_server=True)
        
        self.rhc_q_pub = self.node.create_publisher(Float64MultiArray, 
                                            self.ros_names.rhc_q_topicname(basename=self.rhcviz_basename, 
                                                                        namespace=self.namespace),
                                            qos_profile=self._qos_settings)

        self.robot_q_pub = self.node.create_publisher(Float64MultiArray, 
                                            self.ros_names.robot_q_topicname(basename=self.rhcviz_basename, 
                                                                    namespace=self.namespace),
                                            qos_profile=self._qos_settings)
        
        self.robot_jntnames_pub = self.node.create_publisher(String, 
                                            self.ros_names.robot_jntnames(basename=self.rhcviz_basename, 
                                                                namespace=self.namespace),
                                            qos_profile=self._qos_settings)       

        self.rhc_jntnames_pub = self.node.create_publisher(String, 
                                            self.ros_names.rhc_jntnames(basename=self.rhcviz_basename, 
                                                                namespace=self.namespace),
                                            qos_profile=self._qos_settings)   

        self.cluster_size = None
        self.jnt_names_robot = None
        self.jnt_names_rhc = None

        self.rhc_internal_clients = None
        self.robot_state = None

        self._current_index = 0

        self._update_counter = 0
        self._print_frequency = 100
        
        self._is_running = False

    def _check_selector(self):
        
        if not isinstance(self._robot_selector, List):

            exception = f"robot_selector should be a List!"

            Journal.log(self.__class__.__name__,
                "_parse_selector",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        if not len(self._robot_selector) == 2:

            exception = f"robot_selector should be a List of two values: [start_index, end_idx]!"

            Journal.log(self.__class__.__name__,
                "_parse_selector",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        if not isinstance(self._robot_selector[0], int) or \
            not self._robot_selector[0] >= 0 or \
            not self._robot_selector[0] < self.cluster_size:

            exception = f"first index should be a positive integer, not bigger than{self.cluster_size - 1}!"

            Journal.log(self.__class__.__name__,
                "_parse_selector",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        if not isinstance(self._robot_selector[1], (int, type(None))):

            if self._robot_selector[1] is not None:

                if self._robot_selector[1] <= self._robot_selector[0]:

                    exception = f"second index should be bigger than first index{self._robot_selector[0]}, but got {self._robot_selector[1]}!"

                    Journal.log(self.__class__.__name__,
                        "_parse_selector",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            
                if not self._robot_selector[0] < self.cluster_size:

                    exception = f"second index should be a positive integer, not bigger than{self.cluster_size - 1}!"

                    Journal.log(self.__class__.__name__,
                        "_parse_selector",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
        
        if self._robot_selector[1] is None:

            self._robot_selector[1] = self.cluster_size -1

        self._robot_indexes = list(range(self._robot_selector[0], 
                                    self._robot_selector[1]+1))

    def run(self,
        update_dt: float = 0.01):
                
        # robot state
        self.robot_state = RobotState(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=True,
                                vlevel=VLevel.V2)
        self.robot_state.set_q_remapping(q_remapping=[1, 2, 3, 0]) # remapping from w, i, j, k
        # to rviz conventions (i, k, k, w)

        self.robot_state.run()

        self.cluster_size = self.robot_state.n_robots()
        self.jnt_names_robot = self.robot_state.jnt_names()

        self._check_selector()

        # env selector
        self.env_index = SharedDataView(namespace = self.namespace,
                basename = "EnvSelector",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Int)
        
        self.env_index.run()
        
        rhc_internal_config = RhcInternal.Config(is_server=False, 
                        enable_q=True)
        
        # rhc internal data
        self.rhc_internal_clients = []
        for i in range(len(self._robot_indexes)):
            
            self.rhc_internal_clients.append(RhcInternal(config=rhc_internal_config,
                                                namespace=self.namespace,
                                                rhc_index=self._robot_indexes[i],
                                                verbose=True,
                                                vlevel=VLevel.V2,
                                                safe=False))
            
            self.rhc_internal_clients[self._robot_indexes[i]].run()
           
        # publishing joint names on topic 
        string_array = StringArray()
        self.jnt_names_robot_encoded = string_array.encode(self.jnt_names_robot) # encoding 
        # jnt names in a ; separated string

        self.handshaker.set_n_nodes(self.rhc_internal_clients[0].q.n_cols) # signal to RHViz client
        # the number of nodes of the RHC problem

        self.jnt_names_rhc = self.rhc_internal_clients[0].jnt_names() # assumes all controllers work on the same robot
        self.jnt_names_rhc_encoded = string_array.encode(self.jnt_names_rhc) # encoding 
        # jnt names ifor rhc controllers

        self._is_running = True

        start_time = 0.0
        elapsed_time = 0.0

        time_to_sleep_ns = 0

        info = f": starting bridge with update dt {update_dt} s"
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)

        while self._is_running:

            try:
                
                start_time = time.perf_counter() 

                self.update()

                elapsed_time = time.perf_counter() - start_time

                time_to_sleep_ns = int((update_dt - elapsed_time) * 1e+9) # [ns]

                if time_to_sleep_ns < 0:

                    warning = f": Could not match desired update dt of {update_dt} s. " + \
                        f"Elapsed time to update {elapsed_time}."
                    Journal.log(self.__class__.__name__,
                        "run",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)

                PerfSleep.thread_sleep(time_to_sleep_ns) 

                continue

            except KeyboardInterrupt:

                self.close()

    def update(self):
        
        success = False

        self.env_index.synch_all(read=True, wait=True)

        self._current_index = self.env_index.torch_view[0, 0].item()

        if self._current_index in self._robot_indexes:

            if self._is_running:
            
                # read from shared memory
                self.robot_state.synch_from_shared_mem()
                self.rhc_internal_clients[self._current_index].synch(read=True)

                self._publish()
            
        else:

            warning = f"Current env index {self._current_index} is not within {self._robot_indexes}!" + \
                "No update will be performed"

            Journal.log(self.__class__.__name__,
                "update",
                warning,
                LogType.WARN,
                throw_when_excep = True)

        return success

    def close(self):

        if not self.rhc_internal_clients is None:

            for i in range(len(self.rhc_internal_clients)):

                self.rhc_internal_clients[i].close() # closes servers

        if not self.robot_state is None:

            self.robot_state.close()
    
    def _sporadic_log(self,
                calling_methd: str,
                msg: str,
                logtype: LogType = LogType.INFO):

        if self.verbose and \
            (self._update_counter+1) % self._print_frequency == 0: 
            
            Journal.log(self.__class__.__name__,
                calling_methd,
                msg,
                logtype,
                throw_when_excep = True)
    
    def _contains_nan(self,
                    data: np.ndarray):
        
        return np.isnan(data).any()
    
    def _publish(self):
        
        # continously publish also joint names 
        self.robot_jntnames_pub.publish(String(data=self.jnt_names_robot_encoded))
        
        self.rhc_jntnames_pub.publish(String(data=self.jnt_names_rhc_encoded))

        # publish rhc_q
        rhc_q = self.rhc_internal_clients[self._current_index].q.numpy_view[:, :].flatten()

        root_p_robot = self.robot_state.root_state.get_p(robot_idxs=self._current_index)
        root_q_robot = self.robot_state.root_state.get_q(robot_idxs=self._current_index) # with remapping

        root_q_full = np.concatenate((root_p_robot, root_q_robot), axis=1).flatten()

        jnts_q_robot = self.robot_state.jnts_state.get_q()[self._current_index, :]

        robot_q = np.concatenate((root_q_full, jnts_q_robot), axis=0)

        if not self._contains_nan(rhc_q):

            self.rhc_q_pub.publish(Float64MultiArray(data=rhc_q))

        else:

            self._sporadic_log(calling_methd="_publish", 
                        msg="rhc q data contains some NaN. That data will not be published")
            
        # publish robot_q
        
        if not self._contains_nan(robot_q):

            self.robot_q_pub.publish(Float64MultiArray(data=robot_q))
        
        else:

            self._sporadic_log(calling_methd="_publish", 
                        msg="robot q data contains some NaN. That data will not be published")
