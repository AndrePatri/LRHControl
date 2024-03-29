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
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        contact_names = self.gait_manager.task_interface.model.cmap.keys()
        if not (self.n_contacts == len(contact_names)):
            exception = f"N of contacts within problem {len(contact_names)} does not match n of contacts {self.n_contacts}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
                        
    def step(self):
        
        if self.is_running():
            
            # updates robot refs from shared mem
            self.rob_refs.synch_from_shared_mem()
            self.phase_id.synch_all(read=True, retry=True)
            self.contact_flags.synch_all(read=True, retry=True)

            phase_id = self.phase_id.read_retry(row_index=self.robot_index,
                                col_index=0)[0]

            # contact phases
            if phase_id == -1:
                if self.gait_manager.contact_phases['ball_1'].getEmptyNodes() > 0: # there are available nodes on the horizon 
                    # we assume timelines of the same amount at each rhc instant (only checking one contact)
                    contact_flags = self.contact_flags.get_numpy_view()[self.robot_index, :]
                    is_contact = contact_flags.flatten().tolist() 
                    # contact if contact_flags[i] > 0.5
                    self.gait_manager.cycle(is_contact)
                else:
                    if (self._step_idx+1) % self._print_frequency == 0: 
                        # sporadic log
                        warn = f"Trying to add phases to full timeline! No phase will be set."
                        Journal.log(self.__class__.__name__,
                            "step",
                            warn,
                            LogType.WARN,
                            throw_when_excep = True)
            elif phase_id == 0:
                self.gait_manager.stand()
            elif phase_id == 1:
                self.gait_manager.walk()
            elif phase_id == 2:
                self.gait_manager.crawl()
            elif phase_id == 3:
                self.gait_manager.trot()
            elif phase_id == 4:
                self.gait_manager.trot_jumped()
            elif phase_id == 5:
                self.gait_manager.jump()
            elif phase_id == 6:
                self.gait_manager.wheelie()
            else:
                exception = f"Unsupported phase id {phase_id} has been received!"
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            # updated internal references with latest available ones
            self._apply_refs_to_tasks()
            
            self._step_idx +=1
        
        else:
            exception = f"{self.__class__.__name__} is not running"
            Journal.log(self.__class__.__name__,
                "step",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def _apply_refs_to_tasks(self):
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
            contact_flags_current = self.contact_flags.get_numpy_view()
            phase_id_current = self.phase_id.get_numpy_view()
            contact_flags_current[self.robot_index, :] = np.full((1, self.n_contacts), dtype=np.bool_, fill_value=True)
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

