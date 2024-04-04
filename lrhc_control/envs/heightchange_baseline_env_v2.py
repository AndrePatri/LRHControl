from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

from lrhc_control.utils.sys_utils import PathsGetter
import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os

class LRhcHeightChangeV2(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True):

        obs_dim = 13
        actions_dim = 6

        n_steps_episode_lb = 512 # episode length
        n_steps_episode_ub = 4096
        n_steps_task_rand_lb = 64 # agent refs randomization freq
        n_steps_task_rand_ub = 128

        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything

        env_name = "LRhcHeightChangeV2"
        
        device = "cuda" if use_gpu else "cpu"
        self._task_weight = 2
        self._task_scale = 1
        self._task_err_weights = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.0) 
        self._task_err_weights[0, 0] = 0.001
        self._task_err_weights[0, 1] = 0.001
        self._task_err_weights[0, 2] = 1.0
        self._task_err_weights[0, 3] = 0.001
        self._task_err_weights[0, 4] = 0.001
        self._task_err_weights[0, 5] = 0.001
        
        self._rhc_cnstr_viol_weight = 1
        self._rhc_cnstr_viol_scale = 1

        self._href_lb = 0.3
        self._href_ub = 0.8
        
        self._this_child_path = os.path.abspath(__file__)

        self._reward_clamp_thresh = 1
        
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
        
        agent_action = self.get_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_p_ref = self._rhc_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        rhc_latest_v_ref = self._rhc_refs.rob_refs.root_state.get(data_type="v", gpu=self._use_gpu)
        rhc_latest_omega_ref = self._rhc_refs.rob_refs.root_state.get(data_type="omega", gpu=self._use_gpu)

        # vxy
        rhc_latest_v_ref[:, 0:2] = agent_action[:, 0:2]
        self._rhc_refs.rob_refs.root_state.set(data_type="v", data=rhc_latest_v_ref,
                                            gpu=self._use_gpu) 
        
        # h_ref
        rhc_latest_p_ref[:, 2:3] = agent_action[:, 2:3] # z ref
        self._rhc_refs.rob_refs.root_state.set(data_type="p", data=rhc_latest_p_ref,
                                            gpu=self._use_gpu) 
        
        # omega_ref
        rhc_latest_omega_ref[:, :] = agent_action[:, 3:6]
        self._rhc_refs.rob_refs.root_state.set(data_type="omega", data=rhc_latest_omega_ref,
                                            gpu=self._use_gpu) 

        if self._use_gpu:
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem

    def _compute_rewards(self):
        
        # task error
        task_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        h_ref = self._agent_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)[:, 2:3]
        task_ref[:, 2:3] = h_ref
        task_meas = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu)
        robot_h = self._robot_state.root_state.get(data_type="p", gpu=self._use_gpu)[:, 2:3]
        task_meas[:, 2:3] = robot_h
        task_error = (task_ref - task_meas) * self._task_err_weights
        task_err_norm = torch.norm(task_error, p=2, dim=1, keepdim=True)
                                      
        # rhc penalties
        # rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=self._use_gpu)
        # rhc_fail_penalty = self._rhc_status.fails.get_torch_view(gpu=self._use_gpu)

        rewards = self._rewards.get_torch_view(gpu=self._use_gpu)
        rewards[:, 0:1] = self._task_weight * (1.0 - (self._task_scale * task_err_norm).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh))
        rewards[:, 1:2] = self._rhc_cnstr_viol_weight * (1.0 - self._squashed_rhc_cnstr_viol())
        
        tot_rewards = self._tot_rewards.get_torch_view(gpu=self._use_gpu)
        tot_rewards[:, :] = torch.sum(rewards, dim=1, keepdim=True)

    def _fill_obs(self,
            obs_tensor: torch.Tensor):
                            
        agent_h_ref = self._agent_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)[:, 2:3] # getting z ref
        robot_h = self._robot_state.root_state.get(data_type="p", gpu=self._use_gpu)[:, 2:3]
        robot_twist_meas = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu)
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)

        # rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=self._use_gpu)
        obs_tensor[:, 0:6] = robot_twist_meas
        obs_tensor[:, 2:3] = robot_h # z component ovewritten with h ref
        obs_tensor[:, 6:12] = agent_twist_ref
        obs_tensor[:, 8:9] = agent_h_ref # z component ovewritten with h ref
        obs_tensor[:, 12:13] = self._squashed_rhc_cnstr_viol()
        
    def _squashed_rhc_cnstr_viol(self):
        rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=self._use_gpu)
        return (self._rhc_cnstr_viol_scale * rhc_const_viol).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh)
        
    def _randomize_refs(self,
                env_indxs: torch.Tensor = None):
        
        agent_p_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        if env_indxs is None:
            agent_twist_ref_current[:, :] = 0 # base should be still
            agent_p_ref_current[:, 2:3] = (self._href_ub-self._href_lb) * torch.rand_like(agent_p_ref_current[:, 2:3]) + self._href_lb # randomize h ref
        else:
            agent_twist_ref_current[env_indxs, :] = 0 # base should be still
            agent_p_ref_current[env_indxs, 2:3] = (self._href_ub-self._href_lb) * torch.rand_like(agent_p_ref_current[env_indxs, 2:3]) + self._href_lb

        self._agent_refs.rob_refs.root_state.set(data_type="p", data=agent_p_ref_current,
                                            gpu=self._use_gpu)
        self._agent_refs.rob_refs.root_state.set(data_type="twist", data=agent_twist_ref_current,
                                            gpu=self._use_gpu)
        self._synch_refs(gpu=self._use_gpu)
    
    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        obs_names[0] = "lin_vel_x"
        obs_names[1] = "lin_vel_y"
        obs_names[2] = "root_h"
        obs_names[3] = "omega_x"
        obs_names[4] = "omega_y"
        obs_names[5] = "omega_z"
        obs_names[6] = "lin_vel_x_ref"
        obs_names[7] = "lin_vel_y_ref"
        obs_names[8] = "root_h_ref"
        obs_names[9] = "omega_x_ref"
        obs_names[10] = "omega_y_ref"
        obs_names[11] = "omega_z_ref"
        obs_names[12] = "rhc_const_viol"

        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd"
        action_names[1] = "vy_cmd"
        action_names[2] = "h_cmd"
        action_names[3] = "roll_twist_cmd"
        action_names[4] = "pitch_twist_cmd"
        action_names[5] = "yaw_twist_cmd"
        return action_names

    def _get_rewards_names(self):

        n_rewards = 2
        reward_names = [""] * n_rewards

        reward_names[0] = "task_error"
        reward_names[1] = "rhc_const_viol"
        # reward_names[2] = "rhc_cost"

        return reward_names