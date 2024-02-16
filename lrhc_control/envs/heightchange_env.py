from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

class LRhcHeightChange(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32):

        obs_dim = 4
        actions_dim = 1

        env_name = "LRhcHeightChange"

        super().__init__(namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    env_name=env_name,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype)
    
    def _apply_rhc_actions(self,
                agent_action):

        rhc_current_ref = self._rhc_refs.rob_refs.root_state.get_p(gpu=True)
        rhc_current_ref[:, 2:3] = agent_action # overwrite z ref

        if self._use_gpu:

            self._rhc_refs.rob_refs.root_state.set_p(p=rhc_current_ref,
                                            gpu=True) # write ref on gpu
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True) # write from gpu to cpu mirror
            self._rhc_refs.rob_refs.root_state.synch_all(read=False, wait=True) # write mirror to shared mem
            
        else:

            self._rhc_refs.rob_refs.root_state.set_p(p=rhc_current_ref,
                                            gpu=False) # write ref on cpu mirror
            self._rhc_refs.rob_refs.root_state.synch_all(read=False, wait=True) # write mirror to shared mem

    def _compute_reward(self):
        
        return None
        
        self._synch_data()

        rhc_h_ref = self._rhc_refs.rob_refs.root_state.get_p(gpu=True)[:, 2:3] # getting z ref
        robot_h = self._robot_state.root_state.get_p(gpu=True)[:, 2:3]

        h_error = (rhc_h_ref - robot_h)

        # rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=True)
        # rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=True)
        
        # reward = torch.norm(h_error, p=2) + rhc_cost + rhc_const_viol
        reward = torch.norm(h_error, p=2)

        return reward.item()

    def _get_observations(self):
                
        self._synch_data()

        if self._use_gpu:
            
            agent_h_ref = self._agent_refs.rob_refs.root_state.get_p(gpu=True)[:, 2:3] # getting z ref
            robot_h = self._robot_state.root_state.get_p(gpu=True)[:, 2:3]
            rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=True)
            rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=True)

            self._obs[:, 0:1] = robot_h
            self._obs[:, 1:2] = rhc_cost
            self._obs[:, 2:3] = rhc_const_viol
            self._obs[: ,3:4] = agent_h_ref
        
        else:

            agent_h_ref = self._agent_refs.rob_refs.root_state.get_p(gpu=False)[:, 2:3] # getting z ref
            robot_h = self._robot_state.root_state.get_p(gpu=False)[:, 2:3]
            rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=False)
            rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=False)

            self._obs[:, 0:1] = robot_h
            self._obs[:, 1:2] = rhc_cost
            self._obs[:, 2:3] = rhc_const_viol
            self._obs[: ,3:4] = agent_h_ref
    
    def _randomize_agent_refs(self):
        
        agent_p_ref_current = self._agent_refs.rob_refs.root_state.get_p(gpu=True)

        agent_p_ref_current[:, 2:3] = (1.0 - 0.2) * torch.rand_like(agent_p_ref_current[:, 2:3]) + 0.2 # randomize h ref

        self._agent_refs.rob_refs.root_state.set_p(p=agent_p_ref_current,
                                        gpu=True)
