from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

from lrhc_control.utils.sys_utils import PathsGetter
import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os

class BaybladeEnv(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32):

        obs_dim = 3 # [yaw_twist, yaw_twist_ref, rhc_cnstr, ..]
        actions_dim = 2 + 1 + 3 + 4 # [vxy_cmd, h_cmd, twist_cmd, dostep_0, dostep_1, dostep_2, dostep_3]

        n_steps_episode = 4096
        n_steps_task_rand = 150 # randomize agent refs every n steps

        env_name = "BaybladeEnvTask"

        debug = True

        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything

        self._epsi = 1e-6
        
        self._yaw_twist_weight = 1
        self._yaw_twist_scale = 1
        self._yaw_twist_lb = -0.1 #  [rad/s]
        self._yaw_twist_ub = 0.1

        self._rhc_cnstr_viol_weight = 1
        self._rhc_cnstr_viol_scale = 1

        self._rhc_cost_weight = 1
        self._rhc_cost_scale = 1

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
        
        agent_action = self.get_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_p_ref = self._rhc_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        rhc_latest_v_ref = self._rhc_refs.rob_refs.root_state.get(data_type="v", gpu=self._use_gpu)
        rhc_latest_omega_ref = self._rhc_refs.rob_refs.root_state.get(data_type="omega", gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_view(gpu=self._use_gpu)

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

        # contact flags
        rhc_latest_contact_ref[:, :] = (agent_action[:, 6:10] <= 0.5) # keep contact if agent actiom <=5

        if self._use_gpu:
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)

    def _compute_rewards(self):
        
        # task error
        omega_ref = self._agent_refs.rob_refs.root_state.get(data_type="omega", gpu=self._use_gpu)[:, 2:3] # getting target twist ref
        robot_omega = self._robot_state.root_state.get(data_type="omega",gpu=self._use_gpu)[:, 2:3]
        omega_err = torch.abs((omega_ref - robot_omega))

        # RHC-related rewards
        rewards = self._rewards.get_torch_view(gpu=self._use_gpu)

        rewards[:, 0:1] = 1.0 - self._yaw_twist_weight * omega_err.mul_(self._yaw_twist_scale).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh) 
        rewards[:, 1:2] = 1.0 - self._rhc_cnstr_viol_weight * self._squashed_rhc_cnstr_viol()
        # rewards[:, 2:3] = 1.0 - self._rhc_cost_weight * self._squashed_rhc_cost()

        tot_rewards = self._tot_rewards.get_torch_view(gpu=self._use_gpu)
        tot_rewards[:, :] = torch.sum(rewards, dim=1, keepdim=True)

    def _fill_obs(self,
                obs_tensor: torch.Tensor):
        
        # assigns obs to obs_tensor
        agent_omega_ref = self._agent_refs.rob_refs.root_state.get(data_type="omega",gpu=self._use_gpu)[:, 2:3] # getting z omega (local)
        robot_yaw_twist = self._robot_state.root_state.get(data_type="omega",gpu=self._use_gpu)[:, 2:3] # getting meas. z omega (abs)

        obs_tensor[:, 0:1] = robot_yaw_twist
        obs_tensor[:, 1:2] = agent_omega_ref
        obs_tensor[:, 2:3] = self._squashed_rhc_cnstr_viol()
        # obs_tensor[:, 3:4] = self._squashed_rhc_cost()

    def _squashed_rhc_cnstr_viol(self):

        rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=self._use_gpu)
        return rhc_const_viol.mul_(self._rhc_cnstr_viol_scale).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh) 
    
    def _squashed_rhc_cost(self):

        rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=self._use_gpu)
        return rhc_cost.mul_(self._rhc_cost_scale).clamp(-self._reward_clamp_thresh, self._reward_clamp_thresh) 
    
    def _randomize_refs(self,
                env_indxs: torch.Tensor = None):
        
        agent_omega_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="omega",gpu=self._use_gpu)
        if env_indxs is None:
            agent_omega_ref_current[:, 2:3] = (self._yaw_twist_ub-self._yaw_twist_lb) * torch.rand_like(agent_omega_ref_current[:, 2:3]) + self._yaw_twist_lb # randomize twist ref
        else:
            agent_omega_ref_current[env_indxs, 2:3] = (self._yaw_twist_ub-self._yaw_twist_lb) * torch.rand_like(agent_omega_ref_current[env_indxs, 2:3]) + self._yaw_twist_lb # randomize twist ref
        self._agent_refs.rob_refs.root_state.set(data_type="omega", data=agent_omega_ref_current,
                                            gpu=self._use_gpu)

        self._synch_refs(gpu=self._use_gpu)
    
    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        obs_names[0] = "yaw_twist"
        obs_names[1] = "agent_yaw_twist_ref"
        obs_names[2] = "rhc_const_viol"
        # obs_names[3] = "rhc_cost"

        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()

        action_names[0] = "vx_cmd"
        action_names[1] = "vy_cmd"
        action_names[2] = "h_cmd"
        action_names[3] = "roll_twist_cmd"
        action_names[4] = "pitch_twist_cmd"
        action_names[5] = "yaw_twist_cmd"
        action_names[6] = "dostep_0"
        action_names[7] = "dostep_1"
        action_names[8] = "dostep_2"
        action_names[9] = "dostep_3"

        return action_names

    def _get_rewards_names(self):

        n_rewards = 2
        reward_names = [""] * n_rewards

        reward_names[0] = "twist_error"
        reward_names[1] = "rhc_const_viol"
        # reward_names[2] = "rhc_cost"

        return reward_names