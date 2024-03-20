from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from typing import Union

import numpy as np
import torch

class HybridQuadRhcRefs(RhcRefs):

    def __init__(self, 
            gait_manager: GaitManager, 
            robot_index: int,
            namespace: str, # namespace used for shared mem
            verbose = True,
            vlevel = VLevel.V2,
            safe = True):
        
        self.robot_index = robot_index
        self.robot_index_torch = torch.tensor(self.robot_index)

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

        # task references
        # self.final_base_xy = self.gait_manager.task_interface.getTask('final_base_xy')
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
            
            # updates data from shared mem
            self.rob_refs.synch_from_shared_mem()
            self.phase_id.synch_all(read=True, retry=True)
            self.contact_flags.synch_all(read=True, retry=True)

            phase_id = self.phase_id.read_retry(row_index=self.robot_index,
                                col_index=0)[0]

            # contact phases
            if phase_id == -1:

                if self.gait_manager.contact_phases['ball_1'].getEmptyNodes() > 0: # there are available nodes on the horizon 

                    # we assume timelines of the same amount at each rhc instant (only checking one contact)

                    contact_flags = self.contact_flags.get_torch_view()[self.robot_index, :]

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
            # self.final_base_xy.setRef(self.base_pose.get_pose().numpy().T)
            
            base_q_full_ref = self.rob_refs.root_state.get_q_full(robot_idxs=self.robot_index_torch)

            self.base_position.setRef(base_q_full_ref.numpy().T) # only uses first three components
            self.base_orientation.setRef(base_q_full_ref.numpy().T) # only uses last 4 components (orient. quaternion)

            self._step_idx +=1
        
        else:

            exception = f"{self.__class__.__name__} is not running"

            Journal.log(self.__class__.__name__,
                "step",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
    def reset(self,
            p_ref: Union[torch.Tensor, np.ndarray],
            q_ref: Union[torch.Tensor, np.ndarray]):

        if self.is_running():

            if (not isinstance(p_ref, (torch.Tensor, np.ndarray))) or \
                (not isinstance(q_ref, (torch.Tensor, np.ndarray))):

                exception = f"p_ref and q_ref should be torch tensors or numpy array!"

                Journal.log(self.__class__.__name__,
                    "reset",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            if (not len(p_ref.shape) == 2) or \
                (not len(q_ref.shape) == 2):

                exception = f"p_ref and q_ref should be 2D torch tensors"

                Journal.log(self.__class__.__name__,
                    "reset",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

            if (not p_ref.shape[0] == 1) or \
                (not q_ref.shape[0] == 1):

                exception = f"First dim. of p_ref and q_ref should be 1D"

                Journal.log(self.__class__.__name__,
                    "reset",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            if (not p_ref.shape[1]== 3) or \
                (not q_ref.shape[1] == 4):

                exception = f"Second dim. of either p_ref or q_ref is not consinstent." + \
                                "it should be, respectively, 3 and 4 \(quaternion\)"

                Journal.log(self.__class__.__name__,
                    "reset",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            # resets shared mem

            self.contact_flags.get_torch_view()[self.robot_index, :] = torch.full((1, self.n_contacts), True)
            self.phase_id.get_torch_view()[self.robot_index, :] = -1 # defaults to custom phase id
            self.rob_refs.root_state.set_p(robot_idxs=self.robot_index_torch, p = torch.from_numpy(p_ref))
            self.rob_refs.root_state.set_q(robot_idxs=self.robot_index_torch, q = torch.from_numpy(q_ref))

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

