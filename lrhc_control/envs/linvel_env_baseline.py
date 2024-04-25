from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase
from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcStatus

import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os

class LinVelTrackBaseline(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True):

        # temporarily creating robot state client to get n jnts
        robot_state_tmp = RobotState(namespace=namespace,
                                is_server=False, 
                                safe=False,
                                verbose=verbose,
                                vlevel=vlevel,
                                with_gpu_mirror=False,
                                with_torch_view=False)
        robot_state_tmp.run()
        rhc_status_tmp = RhcStatus(is_server=False,
                        namespace=namespace, 
                        verbose=verbose, 
                        vlevel=vlevel,
                        with_torch_view=False, 
                        with_gpu_mirror=False)
        rhc_status_tmp.run()

        n_jnts = robot_state_tmp.n_jnts()
        # n_contacts = robot_state_tmp.n_contacts()
        self.contact_names = robot_state_tmp.contact_names()
        self.step_var_dim = rhc_status_tmp.rhc_step_var.tot_dim()
        self.n_nodes = rhc_status_tmp.n_nodes
        robot_state_tmp.close()
        rhc_status_tmp.close()

        obs_dim = 18 + n_jnts + len(self.contact_names) 

        actions_dim = 2 + 1 + 3 + 4 # [vxy_cmd, h_cmd, twist_cmd, dostep_0, dostep_1, dostep_2, dostep_3]

        n_steps_episode_lb = 512 # episode length
        n_steps_episode_ub = 4096
        n_steps_task_rand_lb = 128 # agent refs randomization freq
        n_steps_task_rand_ub = 512

        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything

        env_name = "LinVelTrack"
        
        device = "cuda" if use_gpu else "cpu"

        self._task_weight = 1.0
        self._task_scale = 1.0
        self._task_err_weights = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.0) 
        self._task_err_weights[0, 0] = 1.0
        self._task_err_weights[0, 1] = 1.0
        self._task_err_weights[0, 2] = 1e-6
        self._task_err_weights[0, 3] = 1e-6
        self._task_err_weights[0, 4] = 1e-6
        self._task_err_weights[0, 5] = 1e-6
        self._task_err_weights_norm_coeff = torch.sum(self._task_err_weights).item()

        self._rhc_cnstr_viol_weight = 1.0
        self._rhc_cnstr_viol_scale = 1.0 * 1e-3

        self._rhc_cost_weight = 1.0
        self._rhc_cost_scale = 1e-2 * 1e-3

        self._rhc_step_var_scale = 1e-2

        self._linvel_lb = torch.full((1, 3), dtype=dtype, device=device,
                            fill_value=-0.8) 
        self._linvel_ub = torch.full((1, 3), dtype=dtype, device=device,
                            fill_value=0.8)
        self._linvel_lb[0, 0] = -1.0
        self._linvel_lb[0, 1] = -1.0
        self._linvel_lb[0, 2] = 0.0
        self._linvel_ub[0, 0] = 1.0
        self._linvel_ub[0, 1] = 1.0
        self._linvel_ub[0, 2] = 0.0

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

        self._reward_thresh_lb = 0 # used for clipping rewards
        self._obs_threshold_lb = -1e3 # used for clipping observations
        self._reward_thresh_ub = 1 # overrides parent's defaults
        self._obs_threshold_ub = 1e3

        self._actions_offsets[:, :] = 0.0 # default to no offset and scaling
        self._actions_scalings[:, :] = 1.0 
        self._actions_offsets[:, 6:10] = 1.0 # stepping flags 
        self._actions_scalings[:, 6:10] =  0.1 # 0.1

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

        rhc_latest_twist_ref = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        rhc_latest_p_ref = self._rhc_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)

        rhc_latest_twist_ref[:, 0:2] = agent_action[:, 0:2] # lin vel cmd
        rhc_latest_p_ref[:, 2:3] = agent_action[:, 2:3] # h cmds
        rhc_latest_twist_ref[:, 3:6] = agent_action[:, 3:6] # omega cmd

        self._rhc_refs.rob_refs.root_state.set(data_type="p", data=rhc_latest_p_ref,
                                            gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_ref,
                                            gpu=self._use_gpu) 
        
        # contact flags
        rhc_latest_contact_ref[:, :] = agent_action[:, 6:10] > 0.5 # keep contact if agent action > 0.5

        if self._use_gpu:
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)

    def _fill_obs(self,
            obs_tensor: torch.Tensor):

        robot_jnt_q_meas = self._robot_state.jnts_state.get(data_type="q",gpu=self._use_gpu)
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        robot_twist_meas = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu)
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)

        obs_tensor[:, 0:4] = robot_q_meas # [w, i, j, k] (IsaacSim convention)
        obs_tensor[:, 4:10] = robot_twist_meas
        obs_tensor[:, 10:(10+self._n_jnts)] = robot_jnt_q_meas
        obs_tensor[:, (10+self._n_jnts):((10+self._n_jnts)+6)] = agent_twist_ref
        obs_tensor[:, ((10+self._n_jnts)+6):((10*self._n_jnts)+6+1)] = self._rhc_const_viol()
        obs_tensor[:, ((10+self._n_jnts)+6+1):((10+self._n_jnts)+6+2)] = self._rhc_cost()
        obs_tensor[:, ((10+self._n_jnts)+6+2):((10+self._n_jnts)+6+2+len(self.contact_names))] = self._rhc_step_var()
        
    def _rhc_const_viol(self):
        # rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_mirror(gpu=self._use_gpu) # over the whole horizon
        # return self._rhc_cnstr_viol_scale * rhc_const_viol
        rhc_const_viol = self._rhc_status.rhc_nodes_constr_viol.get_torch_mirror(gpu=self._use_gpu)
        return self._rhc_cnstr_viol_scale * rhc_const_viol[:, 0:1] # just on node 0
    
    def _rhc_cost(self):
        # rhc_cost = self._rhc_status.rhc_cost.get_torch_mirror(gpu=self._use_gpu) # over the whole horizon
        # return self._rhc_cost_scale * rhc_cost
        rhc_cost = self._rhc_status.rhc_nodes_cost.get_torch_mirror(gpu=self._use_gpu)
        return self._rhc_cost_scale * rhc_cost[:, 0:1] # just on node 0
    
    def _rhc_step_var(self):
        step_var = self._rhc_status.rhc_step_var.get_torch_mirror(gpu=self._use_gpu)
        to_be_cat = []
        for i in range(len(self.contact_names)):
            start_idx=i*self.n_nodes
            end_idx=i*self.n_nodes+self.n_nodes
            to_be_cat.append(torch.sum(step_var[:, start_idx:end_idx], dim=1, keepdim=True)/self.n_nodes)
        return self._rhc_step_var_scale * torch.cat(to_be_cat, dim=1) 
    
    def _compute_sub_rewards(self,
                    obs: torch.Tensor):
        
        # task error
        task_meas = obs[:, 4:10] # robot twist meas
        task_ref = obs[:, (10+self._n_jnts):((10+self._n_jnts)+6)] # robot hybrid twist refs
        cnstr_viol = obs[:, ((10+self._n_jnts)+6):((10+self._n_jnts)+6+1)]
        rhc_cost = obs[:, ((10+self._n_jnts)+6+1):((10+self._n_jnts)+6+2)]

        epsi=1e-6
        task_error_perc =  torch.abs((task_ref-task_meas)/(task_ref+epsi)) # error normalized wrt ref
        task_error_index = torch.sum(self._task_err_weights * task_error_perc, dim=1, keepdim=True) \
            / self._task_err_weights_norm_coeff # task index is normalized wrt the task weights, so that is bound 
        # to be in [0, +inf]. A task index of 1 means a 100% average error on the task wrt the reference

        sub_rewards = self._rewards.get_torch_mirror(gpu=self._use_gpu)
        sub_rewards[:, 0:1] = self._task_weight * (1.0 - task_error_index)
        sub_rewards[:, 1:2] = self._rhc_cnstr_viol_weight * (1.0 - cnstr_viol)
        sub_rewards[:, 2:3] = self._rhc_cost_weight * (1.0 - rhc_cost)
        
    def _randomize_refs(self,
                env_indxs: torch.Tensor = None):
        
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        if env_indxs is None:
            agent_twist_ref_current[:, :] = 0 # base should be still
            agent_twist_ref_current[:, 0:3] = (self._linvel_ub-self._linvel_lb) * torch.rand_like(agent_twist_ref_current[:, 0:3]) + self._linvel_lb # randomize h ref
        else:
            agent_twist_ref_current[env_indxs, :] = 0 # base should be still
            agent_twist_ref_current[env_indxs, 0:3] = (self._linvel_ub-self._linvel_lb) * torch.rand_like(agent_twist_ref_current[env_indxs, 0:3]) + self._linvel_lb # randomize h ref

        self._agent_refs.rob_refs.root_state.set(data_type="twist", data=agent_twist_ref_current,
                                            gpu=self._use_gpu)
        self._synch_refs(gpu=self._use_gpu)
    
    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        obs_names[0] = "q_w"
        obs_names[1] = "q_i"
        obs_names[2] = "q_j"
        obs_names[3] = "q_k"
        obs_names[4] = "lin_vel_x"
        obs_names[5] = "lin_vel_y"
        obs_names[6] = "lin_vel_z"
        obs_names[7] = "omega_x"
        obs_names[8] = "omega_y"
        obs_names[9] = "omega_z"
        jnt_names = self._robot_state.jnt_names()
        for i in range(self._n_jnts): # jnt obs (pos):
            obs_names[10 + i] = f"{jnt_names[i]}"
        restart_idx = 9 + self._n_jnts
        obs_names[restart_idx + 1] = "lin_vel_x_ref"
        obs_names[restart_idx + 2] = "lin_vel_y_ref"
        obs_names[restart_idx + 3] = "lin_vel_z_ref"
        obs_names[restart_idx + 4] = "omega_x_ref"
        obs_names[restart_idx + 5] = "omega_y_ref"
        obs_names[restart_idx + 6] = "omega_z_ref"
        obs_names[restart_idx + 7] = "rhc_const_viol"
        obs_names[restart_idx + 8] = "rhc_cost"
        i = 0
        for contact in self.contact_names:
            obs_names[restart_idx + 9 + i] = f"step_var_{contact}"
            i+=1

        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd"
        action_names[1] = "vy_cmd"
        action_names[2] = "h_cmd"
        action_names[3] = "roll_twist_cmd"
        action_names[4] = "pitch_twist_cmd"
        action_names[5] = "yaw_twist_cmd"
        action_names[6] = "contact_0"
        action_names[7] = "contact_1"
        action_names[8] = "contact_2"
        action_names[9] = "contact_3"

        return action_names

    def _get_rewards_names(self):

        n_rewards = 3
        reward_names = [""] * n_rewards

        reward_names[0] = "task_error"
        reward_names[1] = "rhc_const_viol"
        reward_names[2] = "rhc_cost"

        return reward_names