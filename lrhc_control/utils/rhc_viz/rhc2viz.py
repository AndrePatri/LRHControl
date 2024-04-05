from rhcviz.utils.handshake import RHCVizHandshake
from rhcviz.utils.namings import NamingConventions
from rhcviz.utils.string_list_encoding import StringArray

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcInternal
from lrhc_control.utils.shared_data.agent_refs import AgentRefs

import numpy as np

from SharsorIPCpp.PySharsorIPC import dtype

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String

from typing import List

from perf_sleep.pyperfsleep import PerfSleep
import time 

class RhcToVizBridge:

    # bridge from shared mem to ROS
    
    def __init__(self, 
            namespace: str, 
            verbose = False,
            rhcviz_basename = "RHCViz",
            robot_selector: List = [0, None],
            with_agent_refs = False):

        self._robot_selector = robot_selector

        self._with_agent_refs = with_agent_refs

        self.verbose = verbose

        self.namespace = namespace # defines uniquely the kind of controller 
        # (associated with a specific robot)

        # ros stuff
        self.ros_names = NamingConventions() # rhcviz naming conventions
        self.rhcviz_basename = rhcviz_basename

        self.handshaker = RHCVizHandshake(self.ros_names.handshake_topicname(basename=self.rhcviz_basename, 
                                            namespace=self.namespace), 
                            is_server=True)
        
        self.rhc_q_pub = rospy.Publisher(self.ros_names.rhc_q_topicname(basename=self.rhcviz_basename, 
                                        namespace=self.namespace), 
                            Float64MultiArray, 
                            queue_size=10)
        
        self.rhc_refs_pub = rospy.Publisher(self.ros_names.rhc_refs_topicname(basename=self.rhcviz_basename, 
                                        namespace=self.namespace), 
                            Float64MultiArray, 
                            queue_size=10)
        if self._with_agent_refs:
            self.hl_refs_pub = rospy.Publisher(self.ros_names.hl_refs_topicname(basename=self.rhcviz_basename, 
                                        namespace=self.namespace), 
                            Float64MultiArray, 
                            queue_size=10)
            
        self.robot_q_pub = rospy.Publisher(self.ros_names.robot_q_topicname(basename=self.rhcviz_basename, 
                                        namespace=self.namespace), 
                            Float64MultiArray, 
                            queue_size=10)
        
        self.robot_jntnames_pub = rospy.Publisher(self.ros_names.robot_jntnames(basename=self.rhcviz_basename, 
                                        namespace=self.namespace), 
                            String, 
                            queue_size=10)       

        self.rhc_jntnames_pub = rospy.Publisher(self.ros_names.rhc_jntnames(basename=self.rhcviz_basename, 
                                        namespace=self.namespace), 
                            String, 
                            queue_size=10)   

        self.cluster_size = None
        self.jnt_names_robot = None
        self.jnt_names_rhc = None

        self.rhc_internal_clients = None
        self.robot_state = None
        self.rhc_refs = None
        self.agent_refs = None

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
        self.rhc_refs = RhcRefs(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=True,
                                vlevel=VLevel.V2)
        self.rhc_refs.rob_refs.set_q_remapping(q_remapping=[1, 2, 3, 0]) # remapping from w, i, j, k
        # to rviz conventions (i, k, k, w)
        self.rhc_refs.run()
        if self._with_agent_refs:
            self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=True,
                                vlevel=VLevel.V2)
            self.agent_refs.run()

        self.cluster_size = self.robot_state.n_robots()
        self.jnt_names_robot = self.robot_state.jnt_names()

        self._check_selector()

        # env selector
        self.env_index = SharedTWrapper(namespace = self.namespace,
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
           
        rospy.init_node('RHC2ROSBridge')

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

                time_to_sleep_ns = int((update_dt - elapsed_time) * 1000000000) # [ns]

                if time_to_sleep_ns < 0:

                    warning = f": Could not match desired update dt of {update_dt} s. " + \
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
                
    def update(self):
        
        success = False

        self.env_index.synch_all(read=True, retry=True)

        self._current_index = self.env_index.get_numpy_view()[0, 0].item()

        if self._current_index in self._robot_indexes:

            if self._is_running:
            
                # read from shared memory
                self.robot_state.synch_from_shared_mem()
                self.rhc_refs.rob_refs.synch_from_shared_mem()
                if self._with_agent_refs:
                    self.agent_refs.rob_refs.synch_from_shared_mem()
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
        
        self._is_running = False
    
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
        rhc_q = self.rhc_internal_clients[self._current_index].q.get_numpy_view()[:, :].flatten()

        root_q_robot = self.robot_state.root_state.get(data_type="q_full",robot_idxs=self._current_index)
        jnts_q_robot = self.robot_state.jnts_state.get(data_type="q")[self._current_index, :]

        robot_q = np.concatenate((root_q_robot, jnts_q_robot), axis=0)

        # rhc refs
        rhc_ref_pose = self.rhc_refs.rob_refs.root_state.get(data_type="q_full",robot_idxs=self._current_index)
        rhc_ref_twist= self.rhc_refs.rob_refs.root_state.get(data_type="twist",robot_idxs=self._current_index)
        rhc_refs = np.concatenate((rhc_ref_pose, rhc_ref_twist), axis=0)
        
        # high lev refs
        if self._with_agent_refs:
            hl_ref_pose = self.agent_refs.rob_refs.root_state.get(data_type="q_full",robot_idxs=self._current_index)
            hl_ref_twist= self.agent_refs.rob_refs.root_state.get(data_type="twist",robot_idxs=self._current_index)
            hl_refs = np.concatenate((hl_ref_pose, hl_ref_twist), axis=0)
            
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
        
        # publish rhc_refs
        if not self._contains_nan(rhc_refs):
            self.rhc_refs_pub.publish(Float64MultiArray(data=rhc_refs))
        else:
            self._sporadic_log(calling_methd="_publish", 
                        msg="rhc refs data contains some NaN. That data will not be published")
        
        # publish hl_refs
        if self._with_agent_refs:
            if not self._contains_nan(hl_refs):
                self.hl_refs_pub.publish(Float64MultiArray(data=hl_refs))
            else:
                self._sporadic_log(calling_methd="_publish", 
                            msg="high-level refs data contains some NaN. That data will not be published")