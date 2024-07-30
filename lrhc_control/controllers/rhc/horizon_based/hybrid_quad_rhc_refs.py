from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from typing import Union

import numpy as np

class HybridQuadRhcRefs(RhcRefs):

    def __init__(self, 
            gait_manager: GaitManager, 
            robot_index: int,
            namespace: str, # namespace used for shared mem
            verbose = True,
            vlevel = VLevel.V2,
            safe = True):
        
        self.robot_index = robot_index
        self.robot_index_np = np.array(self.robot_index)

        self._step_idx = 0
        self._print_frequency = 100

        self._verbose = verbose

        super().__init__( 
                is_server=False,
                with_gpu_mirror=False,
                namespace=namespace,
                safe=safe,
                verbose=verbose,
                vlevel=vlevel)

        if not isinstance(gait_manager, GaitManager):
            exception = f"Provided gait_manager argument should be of GaitManager type!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
        # handles phase transitions
        self.gait_manager = gait_manager

        self._timelines = self.gait_manager._contact_timelines
        
        self._timeline_names = self.gait_manager._timeline_names

        # task interfaces from horizon for setting commands to rhc
        self._get_tasks()

    def _get_tasks(self):
        # can be overridden by child
        self.base_position = self.gait_manager.task_interface.getTask('base_position')
        self.base_orientation = self.gait_manager.task_interface.getTask('base_orientation')

    def run(self):

        super().run()
        if not (self.robot_index < self.rob_refs.n_robots()):
            exception = f"Provided \(0-based\) robot index {self.robot_index} exceeds number of " + \
                " available robots {self.rob_refs.n_robots()}."
            Journal.log(self.__class__.__name__,
                "run",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        contact_names = self.gait_manager.task_interface.model.cmap.keys()
        if not (self.n_contacts() == len(contact_names)):
            exception = f"N of contacts within problem {len(contact_names)} does not match n of contacts {self.n_contacts()}"
            Journal.log(self.__class__.__name__,
                "run",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
                        
    def step(self, q_base: np.ndarray = None):
        
        if self.is_running():
            
            # updates robot refs from shared mem
            self.rob_refs.synch_from_shared_mem()
            self.phase_id.synch_all(read=True, retry=True)
            self.contact_flags.synch_all(read=True, retry=True)

            phase_id = self.phase_id.read_retry(row_index=self.robot_index,
                                col_index=0)[0]
            
            if phase_id == -1: # custom phases
                contact_flags = self.contact_flags.get_numpy_mirror()[self.robot_index, :]
                is_contact = contact_flags.flatten().tolist() 
                for i in range(len(is_contact)):
                    timeline_name = self._timeline_names[i]
                    timeline = self.gait_manager._contact_timelines[timeline_name]
                    if timeline.getEmptyNodes() > 0: # after shift, always add a stance
                        self.gait_manager.add_stand(timeline_name)
                    if is_contact[i] == False:
                        self.gait_manager.add_flight(timeline_name)

                for timeline_name in self._timeline_names: # sanity check
                    timeline = self.gait_manager._contact_timelines[timeline_name]
                    if timeline.getEmptyNodes() > 0:
                        error = f"Empty nodes detected over the horizon! Make sure to fill the whole horizon with valid phases!!"
                        Journal.log(self.__class__.__name__,
                            "step",
                            error,
                            LogType.EXCEP,
                            throw_when_excep = True)
            else:
                exception = f"Unsupported phase id {phase_id} has been received!"
                Journal.log(self.__class__.__name__,
                    "step",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            # updated internal references with latest available ones
            self._apply_refs_to_tasks(q_base=q_base)
            
            self._step_idx +=1
        
        else:
            exception = f"{self.__class__.__name__} is not running"
            Journal.log(self.__class__.__name__,
                "step",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def _apply_refs_to_tasks(self, q_base = None):
        # can be overridden by child
        base_q_full_ref = self.rob_refs.root_state.get(data_type = "q_full", 
                                    robot_idxs=self.robot_index_np).reshape(-1, 1)
        # self.final_base_xy.setRef(self.base_pose.get_pose().numpy().T)
        self.base_position.setRef(base_q_full_ref) # only uses first three components
        self.base_orientation.setRef(base_q_full_ref) # only uses last 4 components (orient. quaternion)
        
    def reset(self,
            p_ref: np.ndarray,
            q_ref: np.ndarray):

        if self.is_running():

            # resets shared mem
            contact_flags_current = self.contact_flags.get_numpy_mirror()
            phase_id_current = self.phase_id.get_numpy_mirror()
            contact_flags_current[self.robot_index, :] = np.full((1, self.n_contacts()), dtype=np.bool_, fill_value=True)
            phase_id_current[self.robot_index, :] = -1 # defaults to custom phase id

            self.rob_refs.root_state.set(data_type="p", data=p_ref, robot_idxs=self.robot_index_np)
            self.rob_refs.root_state.set(data_type="q", data=q_ref, robot_idxs=self.robot_index_np)
            self.rob_refs.root_state.set(data_type="twist", data=np.zeros((1, 6)), robot_idxs=self.robot_index_np)
                                           
            self.contact_flags.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.contact_flags.n_cols,
                                    read=False)
            self.phase_id.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.phase_id.n_cols,
                                    read=False)
            self.rob_refs.root_state.synch_retry(row_index=self.robot_index, col_index=0, n_rows=1, n_cols=self.rob_refs.root_state.n_cols,
                                    read=False)

            # should also empty the timeline for stepping phases
            self._step_idx = 0

        else:
            exception = f"Cannot call reset() since run() was not called!"
            Journal.log(self.__class__.__name__,
                "reset",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

