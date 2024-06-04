from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase
from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcStatus
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os
from lrhc_control.utils.episodic_data import EpisodicData

class LinVelTrackBaseline(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True):
        
        action_repeat = 1

        self._add_last_action_to_obs = False
        self._use_horizontal_frame_for_refs = False # usually impractical for task rand to set this to True 
        self._use_local_base_frame = True

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
        n_contacts = len(self.contact_names)
        self.step_var_dim = rhc_status_tmp.rhc_step_var.tot_dim()
        self.n_nodes = rhc_status_tmp.n_nodes
        robot_state_tmp.close()
        rhc_status_tmp.close()

        actions_dim = 2 + 1 + 3 + 4 # [vxy_cmd, h_cmd, twist_cmd, dostep_0, dostep_1, dostep_2, dostep_3]

        self._n_prev_actions = 1 if self._add_last_action_to_obs else 0
        # obs_dim = 4+6+2*n_jnts+2+2+self._n_prev_actions*actions_dim
        obs_dim = 4+6+n_jnts+2+2+self._n_prev_actions*actions_dim
        episode_timeout_lb = 4096 # episode timeouts (including env substepping when action_repeat>1)
        episode_timeout_ub = 8192
        n_steps_task_rand_lb = 256 # agent refs randomization freq
        n_steps_task_rand_ub = 512

        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything

        env_name = "LinVelTrack"
        
        device = "cuda" if use_gpu else "cpu"

        self._task_weight = 1.0
        self._task_scale = 2.0
        self._task_err_weights = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.0) 
        self._task_err_weights[0, 0] = 1.0
        self._task_err_weights[0, 1] = 1.0
        self._task_err_weights[0, 2] = 0.0
        self._task_err_weights[0, 3] = 0.0
        self._task_err_weights[0, 4] = 0.0
        self._task_err_weights[0, 5] = 0.0
        self._task_err_weights_sum = torch.sum(self._task_err_weights).item()

        self._rhc_cnstr_viol_weight = 1.0
        # self._rhc_cnstr_viol_scale = 1.0 * 1e-3
        self._rhc_cnstr_viol_scale = 1.0 * 5e-3

        self._rhc_cost_weight = 1.0
        # self._rhc_cost_scale = 1e-2 * 1e-3
        self._rhc_cost_scale = 1e-2 * 5e-3

        # power penalty
        self._power_weight = 0.0
        self._power_scale = 5e-3
        self._power_penalty_weights = torch.full((1, n_jnts), dtype=dtype, device=device,
                            fill_value=1.0)
        n_jnts_per_limb = round(n_jnts/n_contacts) # assuming same topology along limbs
        pow_weights_along_limb = [1.0] * n_jnts_per_limb
        pow_weights_along_limb[0] = 1.0
        pow_weights_along_limb[1] = 1.0
        pow_weights_along_limb[2] = 1.0
        for i in range(round(n_jnts/n_contacts)):
            self._power_penalty_weights[0, i*n_contacts:(n_contacts*(i+1))] = pow_weights_along_limb[i]
        self._power_penalty_weights_sum = torch.sum(self._power_penalty_weights).item()

        # task rand
        self._twist_ref_lb = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=-0.8) 
        self._twist_ref_ub = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.8)
        # lin vel
        self._twist_ref_lb[0, 0] = -1.5
        self._twist_ref_lb[0, 1] = -1.5
        self._twist_ref_lb[0, 2] = 0.0
        self._twist_ref_ub[0, 0] = 1.5
        self._twist_ref_ub[0, 1] = 1.5
        self._twist_ref_ub[0, 2] = 0.0
        # angular vel
        self._twist_ref_lb[0, 3] = 0.0
        self._twist_ref_lb[0, 4] = 0.0
        self._twist_ref_lb[0, 5] = 0.0
        self._twist_ref_ub[0, 3] = 0.0
        self._twist_ref_ub[0, 4] = 0.0
        self._twist_ref_ub[0, 5] = 0.0

        self._rhc_step_var_scale = 1

        self._this_child_path = os.path.abspath(__file__)

        super().__init__(namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    episode_timeout_lb=episode_timeout_lb,
                    episode_timeout_ub=episode_timeout_ub,
                    n_steps_task_rand_lb=n_steps_task_rand_lb,
                    n_steps_task_rand_ub=n_steps_task_rand_ub,
                    action_repeat=action_repeat,
                    env_name=env_name,
                    n_preinit_steps=n_preinit_steps,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype,
                    debug=debug)

        # overriding parent's defaults 
        self._reward_thresh_lb[:, 0] = -10
        self._reward_thresh_lb[:, 1] = -1
        self._reward_thresh_lb[:, 2] = -1
        self._reward_thresh_lb[:, 3] = -1
        self._reward_thresh_ub[:, 0] = 10
        self._reward_thresh_ub[:, 1] = 1
        self._reward_thresh_ub[:, 2] = 1 
        self._reward_thresh_ub[:, 3] = 1 

        self._obs_threshold_lb = -1e3 # used for clipping observations
        self._obs_threshold_ub = 1e3

        self._actions_offsets[:, :] = 0.0 # default to no offset and scaling
        self._actions_scalings[:, :] = 1.0 
        self._actions_offsets[:, 6:10] = 0.0 # stepping flags 
        self._actions_scalings[:, 6:10] =  1.0 

        # custom db info 
        step_idx_data = EpisodicData("ContactIndex", self._rhc_step_var(gpu=False), self.contact_names)
        self._add_custom_db_info(db_data=step_idx_data)
        
        # other static db info 
        self.custom_db_info["add_last_action_to_obs"] = self._add_last_action_to_obs
        self.custom_db_info["use_horizontal_frame_for_refs"] = self._use_horizontal_frame_for_refs
        self.custom_db_info["use_local_base_frame"] = self._use_local_base_frame

    def _custom_post_init(self):
        # some aux data to avoid allocations at training runtime
        self._robot_twist_meas_h = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu).clone()
        self._robot_twist_meas_b = self._robot_twist_meas_h.clone()
        self._robot_twist_meas_w = self._robot_twist_meas_h.clone()

        # task aux objs
        device = "cuda" if self._use_gpu else "cpu"
        self._ni_scaling = torch.zeros((self._n_envs, 1),dtype=self._dtype,device=device)

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
        rhc_latest_contact_ref[:, :] = agent_action[:, 6:10] > 0 # keep contact if agent action > 0

        if self._use_gpu:
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)

    def _fill_obs(self,
            obs_tensor: torch.Tensor):

        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        robot_jnt_q_meas = self._robot_state.jnts_state.get(data_type="q",gpu=self._use_gpu)
        robot_twist_meas = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu)
        # robot_jnt_v_meas = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)

        next_idx=0
        obs_tensor[:, next_idx:(next_idx+4)] = robot_q_meas # [w, i, j, k] (IsaacSim convention)
        next_idx+=4
        if self._use_local_base_frame: # measurement from world to local base link
            world2base_frame(t_w=robot_twist_meas,q_b=robot_q_meas,t_out=self._robot_twist_meas_b)
            obs_tensor[:, next_idx:(next_idx+6)] = self._robot_twist_meas_b
        else:
            obs_tensor[:, next_idx:(next_idx+6)] = robot_twist_meas
        next_idx+=6
        obs_tensor[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_q_meas
        next_idx+=self._n_jnts
        # obs_tensor[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_v_meas
        # next_idx+=self._n_jnts
        obs_tensor[:, next_idx:(next_idx+2)] = agent_twist_ref[:, 0:2] # high lev agent ref (local base if self._use_local_base_frame)
        next_idx+=2
        obs_tensor[:, next_idx:(next_idx+1)] = self._rhc_const_viol(gpu=self._use_gpu)
        next_idx+=1
        obs_tensor[:, next_idx:(next_idx+1)] = self._rhc_cost(gpu=self._use_gpu)
        next_idx+=1
        # obs_tensor[:, next_idx:(next_idx+len(self.contact_names))] = self._rhc_step_var(gpu=self._use_gpu)
        # next_idx+=len(self.contact_names)
        # adding last action to obs at the back of the obs tensor
        if self._add_last_action_to_obs:
            last_actions = self._actions.get_torch_mirror(gpu=self._use_gpu)
            obs_tensor[:, next_idx:(next_idx+self._n_prev_actions*self.actions_dim())] = last_actions
            next_idx+=self._n_prev_actions*self.actions_dim()

    def _get_custom_db_data(self, 
            episode_finished):
        
        self.custom_db_data["ContactIndex"].update(new_data=self._rhc_step_var(gpu=False), 
                                    ep_finished=episode_finished.cpu())
    
    def _mech_power_penalty(self, jnts_vel, jnts_effort):
        tot_weighted_power = torch.sum((jnts_effort*jnts_vel)*self._power_penalty_weights, dim=1, keepdim=True)/self._power_penalty_weights_sum
        return tot_weighted_power
    
    def _task_err_quadv2(self, task_ref, task_meas):
        delta = 0.01 # [m/s]
        ref_norm = task_ref.norm(dim=1,keepdim=True)
        self._ni_scaling[:, :] = ref_norm
        self._ni_scaling[ref_norm < delta] = delta
        task_error = (task_ref-task_meas)/(self._ni_scaling)
        task_wmse = torch.sum((task_error*task_error)*self._task_err_weights, dim=1, keepdim=True)/self._task_err_weights_sum
        return task_wmse # weighted mean square error (along task dimension)
    
    def _task_err_pseudolinv2(self, task_ref, task_meas):
        task_wmse = self._task_err_quadv2(task_ref=task_ref, task_meas=task_meas)
        return task_wmse.sqrt()
    
    def _task_err_quad(self, task_ref, task_meas):
        task_error = (task_ref-task_meas)
        task_wmse = torch.sum((task_error*task_error)*self._task_err_weights, dim=1, keepdim=True)/self._task_err_weights_sum
        return task_wmse # weighted mean square error (along task dimension)
    
    def _task_err_pseudolin(self, task_ref, task_meas):
        task_wmse = self._task_err_quad(task_ref=task_ref, task_meas=task_meas)
        return task_wmse.sqrt()
    
    def _rhc_const_viol(self, gpu: bool):
        # rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_mirror(gpu=self._use_gpu) # over the whole horizon
        # return self._rhc_cnstr_viol_scale * rhc_const_viol
        rhc_const_viol = self._rhc_status.rhc_nodes_constr_viol.get_torch_mirror(gpu=gpu)
        return self._rhc_cnstr_viol_scale * rhc_const_viol[:, 0:1] # just on node 0
    
    def _rhc_cost(self, gpu: bool):
        # rhc_cost = self._rhc_status.rhc_cost.get_torch_mirror(gpu=self._use_gpu) # over the whole horizon
        # return self._rhc_cost_scale * rhc_cost
        rhc_cost = self._rhc_status.rhc_nodes_cost.get_torch_mirror(gpu=gpu)
        return self._rhc_cost_scale * rhc_cost[:, 0:1] # just on node 0
    
    def _rhc_step_var(self, gpu: bool):
        step_var = self._rhc_status.rhc_step_var.get_torch_mirror(gpu=gpu)
        to_be_cat = []
        for i in range(len(self.contact_names)):
            start_idx=i*self.n_nodes
            end_idx=i*self.n_nodes+self.n_nodes
            to_be_cat.append(torch.sum(step_var[:, start_idx:end_idx], dim=1, keepdim=True)/self.n_nodes)
        return self._rhc_step_var_scale * torch.cat(to_be_cat, dim=1) 
    
    def _compute_sub_rewards(self,
                    obs: torch.Tensor):
        
        # task_error_fun = self._task_err_pseudolin
        task_error_fun = self._task_err_pseudolinv2

        # task error
        # task_meas = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu) # robot twist meas (local base if _use_local_base_frame)
        task_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu) # high level agent refs (hybrid twist)
        # task_error_wmse = self._task_err_quad(task_meas=task_meas, task_ref=task_ref)
        if self._use_local_base_frame and self._use_horizontal_frame_for_refs:
           base2world_frame(t_b=obs[:, 4:10],q_b=obs[:, 0:4],t_out=self._robot_twist_meas_w)
           w2hor_frame(t_w=self._robot_twist_meas_w,q_b=obs[:, 0:4],t_out=self._robot_twist_meas_h)
           task_error_pseudolin = task_error_fun(task_meas=self._robot_twist_meas_h, 
                                                task_ref=task_ref)
        elif self._use_local_base_frame and not self._use_horizontal_frame_for_refs:
            base2world_frame(t_b=obs[:, 4:10],q_b=obs[:, 0:4],t_out=self._robot_twist_meas_w)
            task_error_pseudolin = task_error_fun(task_meas=self._robot_twist_meas_w, 
                                                task_ref=task_ref)
        elif not self._use_local_base_frame and self._use_horizontal_frame_for_refs:
            w2hor_frame(t_w=obs[:, 4:10],q_b=obs[:, 0:4],t_out=self._robot_twist_meas_h)
            task_error_pseudolin = task_error_fun(task_meas=self._robot_twist_meas_h, 
                                                task_ref=task_ref)
        else: # all in world frame
            task_error_pseudolin = task_error_fun(task_meas=obs[:, 4:10], 
                                                task_ref=task_ref) 
        
        # mech power
        jnts_vel = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
        jnts_effort = self._robot_state.jnts_state.get(data_type="eff",gpu=self._use_gpu)
        weighted_mech_power = self._mech_power_penalty(jnts_vel=jnts_vel, 
                                            jnts_effort=jnts_effort)

        sub_rewards = self._rewards.get_torch_mirror(gpu=self._use_gpu)
        sub_rewards[:, 0:1] = self._task_weight * (1.0 - self._task_scale * task_error_pseudolin)
        sub_rewards[:, 1:2] = self._power_weight * (1.0 - self._power_scale * weighted_mech_power)
        sub_rewards[:, 2:3] = self._rhc_cnstr_viol_weight * (1.0 - self._rhc_const_viol(gpu=self._use_gpu))
        sub_rewards[:, 3:4] = self._rhc_cost_weight * (1.0 - self._rhc_cost(gpu=self._use_gpu))
        
    def _randomize_refs(self,
                env_indxs: torch.Tensor = None):
        
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        if env_indxs is None:
            agent_twist_ref_current[:, :] = torch.rand_like(agent_twist_ref_current[:, :]) * (self._twist_ref_ub-self._twist_ref_lb) + self._twist_ref_lb
        else:
            agent_twist_ref_current[env_indxs, :] =  torch.rand_like(agent_twist_ref_current[env_indxs, :]) * (self._twist_ref_ub-self._twist_ref_lb) + self._twist_ref_lb
        self._agent_refs.rob_refs.root_state.set(data_type="twist", data=agent_twist_ref_current,
                                            gpu=self._use_gpu)
        self._synch_refs(gpu=self._use_gpu)
    
    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        next_idx=0
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
        next_idx+=10
        jnt_names = self._robot_state.jnt_names()
        for i in range(self._n_jnts): # jnt obs (pos):
            obs_names[next_idx+i] = f"q_{jnt_names[i]}"
        next_idx+=self._n_jnts
        # for i in range(self._n_jnts): # jnt obs (v):
        #     obs_names[next_idx+i] = f"v_{jnt_names[i]}"
        # next_idx+=self._n_jnts
        obs_names[next_idx] = "lin_vel_x_ref" # specified in the "horizontal frame"
        obs_names[next_idx+1] = "lin_vel_y_ref"
        next_idx+=2
        # i = 0
        # for contact in self.contact_names:
        #     obs_names[next_idx+i] = f"step_var_{contact}"
        #     i+=1        
        # next_idx+=len(self.contact_names)
        obs_names[next_idx] = "rhc_const_viol"
        obs_names[next_idx + 1] = "rhc_cost"
        next_idx+=2
        action_names = self._get_action_names()
        for pre_t_idx in range(self._n_prev_actions):
            for prev_act_idx in range(self.actions_dim()):
                obs_names[next_idx+pre_t_idx*self.actions_dim()+prev_act_idx] = action_names[prev_act_idx]+f"_tm_{pre_t_idx}"

        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd" # twist commands from agent to RHC controller
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

        n_rewards = 4
        reward_names = [""] * n_rewards

        reward_names[0] = "task_error"
        reward_names[1] = "mech_power"
        reward_names[2] = "rhc_const_viol"
        reward_names[3] = "rhc_cost"

        return reward_names