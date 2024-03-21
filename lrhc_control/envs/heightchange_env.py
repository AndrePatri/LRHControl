from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

from lrhc_control.utils.sys_utils import PathsGetter
import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os

class LRhcHeightChange(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32):

        obs_dim = 4
        actions_dim = 1

        n_steps_episode = 4096
        n_steps_task_rand = 150 # randomize agent refs every n steps

        env_name = "LRhcHeightChange"

        debug = True

        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything

        self._epsi = 1e-6
        
        self._h_error_weight = 3
        self._rhc_cnstr_viol_weight = 1
        self._rhc_cost_weight = 1

        self._h_error_scale = 2
        self._rhc_cost_scale = 100
        self._rhc_cnstr_viol_scale = 1

        self._h_cmd_offset = 0.0
        self._h_cmd_scale = 1.0

        self._href_lb = 0.3
        self._href_ub = 0.8
        
        self._this_child_path = os.path.abspath(__file__)

        self._reward_clamp_thresh = 1
        
        super().__init__(namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    n_steps_episode=n_steps_episode,
                    n_steps_task_rand=n_steps_task_rand,
                    env_name=env_name,
                    n_preinit_steps=n_preinit_steps,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype,
                    debug=debug)

    def get_file_paths(self):

        paths = []
        path_getter = PathsGetter()
        paths.append(self._this_child_path)
        paths.append(super()._get_this_file_path())
        paths.append(path_getter.SIMENVPATH)
        for script_path in path_getter.SCRIPTSPATHS:
            paths.append(script_path)
        return paths

    def get_aux_dir(self):

        aux_dirs = []
        path_getter = PathsGetter()

        aux_dirs.append(path_getter.RHCDIR)
        return aux_dirs

    def _apply_actions_to_rhc(self):
        
        agent_action = self.get_actions()

        rhc_latest_ref = self._rhc_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        rhc_latest_ref[:, 2:3] = agent_action * self._h_cmd_scale + self._h_cmd_offset # overwrite z ref
        
        self._rhc_refs.rob_refs.root_state.set(data_type="p", data=rhc_latest_ref,
                                            gpu=self._use_gpu) # write refs

        if self._use_gpu:

            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror

        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem

    def _compute_rewards(self):
        
        # task error
        h_ref = self._agent_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)[:, 2:3] # getting target z ref
        robot_h = self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 2:3]
        h_error = torch.abs((h_ref - robot_h))

        # RHC-related rewards
        rewards = self._rewards.get_torch_view(gpu=self._use_gpu)

        rewards[:, 0:1] = 1.0 - self._h_error_weight * h_error.mul_(self._h_error_scale).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh) 
        rewards[:, 1:2] = 1.0 - self._rhc_cnstr_viol_weight * self._squashed_rhc_cnstr_viol()
        rewards[:, 2:3] = 1.0 - self._rhc_cost_weight * self._squashed_rhc_cost()

        tot_rewards = self._tot_rewards.get_torch_view(gpu=self._use_gpu)
        tot_rewards[:, :] = torch.sum(rewards, dim=1, keepdim=True)

    def _fill_obs(self,
                obs_tensor: torch.Tensor):
        
        # assigns obs to obs_tensor
        agent_h_ref = self._agent_refs.rob_refs.root_state.get(data_type="p",gpu=self._use_gpu)[:, 2:3] # getting z ref
        robot_h = self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 2:3]

        obs_tensor[:, 0:1] = robot_h
        obs_tensor[:, 1:2] = agent_h_ref
        obs_tensor[:, 2:3] = self._squashed_rhc_cnstr_viol()
        obs_tensor[:, 3:4] = self._squashed_rhc_cost()

    def _squashed_rhc_cnstr_viol(self):

        rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=self._use_gpu)

        return rhc_const_viol.mul_(self._rhc_cnstr_viol_scale).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh) 
    
    def _squashed_rhc_cost(self):

        rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=self._use_gpu)

        return rhc_cost.mul_(self._rhc_cost_scale).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh) 
    
    def _randomize_refs(self,
                env_indxs: torch.Tensor = None):
        
        if env_indxs is None:

            agent_p_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="p",gpu=self._use_gpu)
            agent_p_ref_current[:, 2:3] = (self._href_ub-self._href_lb) * torch.rand_like(agent_p_ref_current[:, 2:3]) + self._href_lb # randomize h ref

        else:

            agent_p_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="p",gpu=self._use_gpu)
            agent_p_ref_current[env_indxs, 2:3] = (self._href_ub-self._href_lb) * torch.rand_like(agent_p_ref_current[env_indxs, 2:3]) + self._href_lb

        self._agent_refs.rob_refs.root_state.set(data_type="p", data=agent_p_ref_current,
                                            gpu=self._use_gpu)

        self._synch_refs(gpu=self._use_gpu)
    
    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        obs_names[0] = "robot_h"
        obs_names[1] = "agent_h_ref"
        obs_names[2] = "rhc_const_viol"
        obs_names[3] = "rhc_cost"

        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()

        action_names[0] = "rhc_cmd_h"

        return action_names

    def _get_rewards_names(self):

        n_rewards = 3
        reward_names = [""] * n_rewards

        reward_names[0] = "h_error"
        reward_names[1] = "rhc_const_viol"
        reward_names[2] = "rhc_cost"

        return reward_names