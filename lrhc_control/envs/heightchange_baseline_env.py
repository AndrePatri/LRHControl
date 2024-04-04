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
            dtype: torch.dtype = torch.float32,
            debug: bool = True):

        obs_dim = 3
        actions_dim = 1

        n_steps_episode_lb = 512 # episode length
        n_steps_episode_ub = 4096
        n_steps_task_rand_lb = 64 # agent refs randomization freq
        n_steps_task_rand_ub = 128

        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything

        env_name = "LRhcHeightChange"
        
        self._h_error_scale = 2
        self._cnstr_viol_scale = 1

        self._href_lb = 0.3
        self._href_ub = 0.8
        
        self._this_child_path = os.path.abspath(__file__)
        
        super().__init__(namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    n_steps_episode_lb=n_steps_episode_lb,
                    n_steps_episode_ub=n_steps_episode_ub,
                    n_steps_task_rand_lb=n_steps_task_rand_lb,
                    n_steps_task_rand_ub=n_steps_task_rand_ub,
                    env_name=env_name,
                    n_preinit_steps=n_preinit_steps,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype,
                    debug=debug)

        self._reward_thresh_lb = -1 # used for clipping rewards
        self._obs_threshold_lb = -10 # used for clipping observations
        self._reward_thresh_ub = 1 # overrides parent's defaults
        self._obs_threshold_ub = 10

        self._actions_offsets[:, :] = 0.0 # vxy_cmd 
        self._actions_scalings[:, :] = 1.0 # 0.05

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

        rhc_latest_p_ref = self._rhc_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        rhc_latest_p_ref[:, 2:3] = agent_action

        self._rhc_refs.rob_refs.root_state.set(data_type="p", data=rhc_latest_p_ref,
                                            gpu=self._use_gpu)
        if self._use_gpu:
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem

    def _compute_sub_rewards(self,
                    obs: torch.Tensor):
        
        # task error
        h_ref = obs[:, 1:2]
        robot_h = obs[:, 0:1]
        cnstr_viol = obs[:, 2:3]

        h_error = torch.abs((h_ref - robot_h))

        sub_rewards = self._rewards.get_torch_view(gpu=self._use_gpu)
        sub_rewards[:, 0:1] = 1.0 - h_error * self._h_error_scale
        sub_rewards[:, 1:2] = 1.0 - cnstr_viol

    def _fill_obs(self,
            obs_tensor: torch.Tensor):
                            
        agent_h_ref = self._agent_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)[:, 2:3] # getting z ref
        robot_h = self._robot_state.root_state.get(data_type="p", gpu=self._use_gpu)[:, 2:3]
        # rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=self._use_gpu)
        obs_tensor[:, 0:1] = robot_h
        obs_tensor[:, 1:2] = agent_h_ref
        obs_tensor[:, 2:3] = self._rhc_const_viol()

    def _rhc_const_viol(self):
        rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=self._use_gpu)
        return self._cnstr_viol_scale * rhc_const_viol
           
    def _randomize_refs(self,
                env_indxs: torch.Tensor = None):
        
        agent_p_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        if env_indxs is None:
            agent_p_ref_current[:, 2:3] = (self._href_ub-self._href_lb) * torch.rand_like(agent_p_ref_current[:, 2:3]) + self._href_lb # randomize h ref
        else:
            agent_p_ref_current[env_indxs, 2:3] = (self._href_ub-self._href_lb) * torch.rand_like(agent_p_ref_current[env_indxs, 2:3]) + self._href_lb

        self._agent_refs.rob_refs.root_state.set(data_type="p", data=agent_p_ref_current,
                                            gpu=self._use_gpu)
        self._synch_refs(gpu=self._use_gpu)
    
    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        obs_names[0] = "robot_h"
        obs_names[1] = "agent_h_ref"
        obs_names[2] = "rhc_const_viol"

        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "rhc_cmd_h"
        return action_names

    def _get_rewards_names(self):

        n_rewards = 2
        reward_names = [""] * n_rewards

        reward_names[0] = "h_error"
        reward_names[1] = "rhc_const_viol"
        # reward_names[2] = "rhc_cost"

        return reward_names