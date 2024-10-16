from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcStatus
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os
from lrhc_control.utils.episodic_data import EpisodicData
from lrhc_control.utils.signal_smoother import ExponentialSignalSmoother

class LinVelTrackBaseline(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            actions_dim: int = 10,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000):
        
        action_repeat = 3 # frame skipping (different agent action every action_repeat
        # env substeps)

        self._single_task_ref_per_episode=True # if True, the task ref is constant over the episode (ie
        # episodes are truncated when task is changed)
        self._twist_meas_is_base_local = False # twist meas from remote sim env are already base-local
        # (usually, this has to be set to True when running on the real robot)
        # we assume the twist meas for the agent to be provided always in local base frame

        self._add_prev_actions_stats_to_obs = True # add actions std, mean + last action over a horizon to obs
        self._add_contact_idx_to_obs=False # add a variable which reflects the magnitute of the contact forces over the horizon
        self._add_rhc_fz_to_obs=True # add estimate vertical contact f to obs
        self._add_internal_rhc_q_to_obs=True # add base orientation internal to the rhc controller (useful when running controller
        # in open loop)
        self._add_rhc_avrg_root_twist_to_obs=True
        self._add_rhc_root_twist_to_obs=True
        self._add_fail_idx_to_obs=True # add a failure index which is directly correlated to env failure due to rhc controller explosion

        self._use_vel_err_sig_smoother=True # whether to smooth vel error signal
        self._vel_err_smoother=None

        # temporarily creating robot state client to get some data
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
        self.step_var_dim = rhc_status_tmp.rhc_fcn.tot_dim()
        self.n_nodes = rhc_status_tmp.n_nodes
        robot_state_tmp.close()
        rhc_status_tmp.close()

        # defining obs dim
        obs_dim=4 # base orientation quaternion 
        obs_dim+=3 # meas omega
        obs_dim+=2*n_jnts # joint pos + vel
        if self._add_internal_rhc_q_to_obs:
            obs_dim+=4 # internal rhc base orientation
        if self._add_rhc_root_twist_to_obs:
            obs_dim+=6
        if self._add_rhc_avrg_root_twist_to_obs:
            obs_dim+=6
        if self._add_contact_idx_to_obs:
            obs_dim+=n_contacts # contact index var
        if self._add_rhc_fz_to_obs:
            obs_dim+=n_contacts
        obs_dim+=2 # 2D lin vel reference to be tracked
        if self._add_fail_idx_to_obs:
            obs_dim+=1 # rhc controller failure index
        if self._add_prev_actions_stats_to_obs:
            obs_dim+=3*actions_dim# previous agent actions statistics (mean, std + last action)

        episode_timeout_lb = 1024 # episode timeouts (including env substepping when action_repeat>1)
        episode_timeout_ub = 1024
        n_steps_task_rand_lb = 400 # agent refs randomization freq
        n_steps_task_rand_ub = 400 # lb not eq. to ub to remove correlations between episodes
        # across diff envs
        random_reset_freq = 10 # a random reset once every n-episodes (per env)
        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything

        env_name = "LinVelTrack"
        
        device = "cuda" if use_gpu else "cpu"
        
        # health reward 
        self._health_value = 0.0

        # task tracking
        self._task_offset = 10.0 # 10.0
        self._task_scale = 10.0 # perc-based
        # self._task_scale = 20.0 # 5.0
        self._task_err_weights = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.0) 
        self._task_err_weights[0, 0] = 1.0
        self._task_err_weights[0, 1] = 1.0
        self._task_err_weights[0, 2] = 1e-6
        self._task_err_weights[0, 3] = 1e-6
        self._task_err_weights[0, 4] = 1e-6
        self._task_err_weights[0, 5] = 1e-6

        # task pred tracking
        self._task_pred_offset = 0.0 # 10.0
        self._task_pred_scale = 0.0 # perc-based
        self._task_pred_err_weights = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.0) 
        self._task_pred_err_weights[0, 0] = 1.0
        self._task_pred_err_weights[0, 1] = 1.0
        self._task_pred_err_weights[0, 2] = 1e-6
        self._task_pred_err_weights[0, 3] = 1e-6
        self._task_pred_err_weights[0, 4] = 1e-6
        self._task_pred_err_weights[0, 5] = 1e-6

        # fail idx
        self._rhc_fail_idx_offset = 0.0
        self._rhc_fail_idx_rew_scale = 0.0 # 1e-4
        self._rhc_fail_idx_scale=1.0

        # power penalty
        self._power_offset = 0 # 10.0
        self._power_scale = 0 # 0.1
        self._power_penalty_weights = torch.full((1, n_jnts), dtype=dtype, device=device,
                            fill_value=1.0)
        n_jnts_per_limb = round(n_jnts/n_contacts) # assuming same topology along limbs
        pow_weights_along_limb = [1.0] * n_jnts_per_limb
        for i in range(round(n_jnts/n_contacts)):
            self._power_penalty_weights[0, i*n_contacts:(n_contacts*(i+1))] = pow_weights_along_limb[i]
        self._power_penalty_weights_sum = torch.sum(self._power_penalty_weights).item()

        # jnt vel penalty 
        self._jnt_vel_offset = 0.0
        self._jnt_vel_scale = 0.0 # 0.3
        self._jnt_vel_penalty_weights = torch.full((1, n_jnts), dtype=dtype, device=device,
                            fill_value=1.0)
        jnt_vel_weights_along_limb = [1.0] * n_jnts_per_limb
        for i in range(round(n_jnts/n_contacts)):
            self._jnt_vel_penalty_weights[0, i*n_contacts:(n_contacts*(i+1))] = jnt_vel_weights_along_limb[i]
        self._jnt_vel_penalty_weights_sum = torch.sum(self._jnt_vel_penalty_weights).item()
        
        # task rand
        self._use_pof0 = False
        self._pof0 = 0.1
        self._twist_ref_lb = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=-0.8) 
        self._twist_ref_ub = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.8)
        # lin vel
        self.max_ref=1.0
        self._twist_ref_lb[0, 0] = -self.max_ref
        self._twist_ref_lb[0, 1] = -self.max_ref
        self._twist_ref_lb[0, 2] = 0.0
        self._twist_ref_ub[0, 0] = self.max_ref
        self._twist_ref_ub[0, 1] = self.max_ref
        self._twist_ref_ub[0, 2] = 0.0
        # angular vel
        self._twist_ref_lb[0, 3] = 0.0
        self._twist_ref_lb[0, 4] = 0.0
        self._twist_ref_lb[0, 5] = 0.0
        self._twist_ref_ub[0, 3] = 0.0
        self._twist_ref_ub[0, 4] = 0.0
        self._twist_ref_ub[0, 5] = 0.0

        self._twist_ref_offset = (self._twist_ref_ub + self._twist_ref_lb)/2.0
        self._twist_ref_scale = (self._twist_ref_ub - self._twist_ref_lb)/2.0

        self._rhc_fcn_scale = 1

        self._this_child_path = os.path.abspath(__file__)

        super().__init__(namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    episode_timeout_lb=episode_timeout_lb,
                    episode_timeout_ub=episode_timeout_ub,
                    n_steps_task_rand_lb=n_steps_task_rand_lb,
                    n_steps_task_rand_ub=n_steps_task_rand_ub,
                    random_reset_freq=random_reset_freq,
                    use_random_safety_reset=True,
                    action_repeat=action_repeat,
                    env_name=env_name,
                    n_preinit_steps=n_preinit_steps,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype,
                    debug=debug,
                    override_agent_refs=override_agent_refs,
                    timeout_ms=timeout_ms,
                    rescale_rewards=True,
                    srew_drescaling=True,
                    srew_tsrescaling=False,
                    use_act_mem_bf=self._add_prev_actions_stats_to_obs,
                    act_membf_size=30)

        # action regularization
        self._actions_diff_rew_offset = 0.0
        if not self._add_prev_actions_stats_to_obs: # we need the action in obs to use this reward
            self._actions_diff_rew_offset=0.0
        self._actions_diff_scale = 0.0#1.0
        self._action_diff_weights = torch.full((1, actions_dim), dtype=dtype, device=device,
                            fill_value=1.0)
        self._action_diff_weights[:, 6:10]=1.0 
        self._prev_actions = torch.full_like(input=self.get_actions(),fill_value=0.0)
        self._prev_actions[:, :] = self.get_actions()
        range_scale=(self._actions_ub-self._actions_lb)/2.0
        self._action_diff_weights[:, :]*=1.0/range_scale
        self._action_diff_w_sum = torch.sum(self._action_diff_weights).item()

        # custom db info 
        step_idx_data = EpisodicData("ContactIndex", self._rhc_fcn(gpu=False), self.contact_names,
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._add_custom_db_info(db_data=step_idx_data)
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=False)
        agent_twist_ref_data = EpisodicData("AgentTwistRefs", agent_twist_ref, 
            ["v_x", "v_y", "v_z", "omega_x", "omega_y", "omega_z"],
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._add_custom_db_info(db_data=agent_twist_ref_data)
        rhc_fail_idx = EpisodicData("RhcFailIdx", self._rhc_fail_idx(gpu=False), ["rhc_cost"],
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._add_custom_db_info(db_data=rhc_fail_idx)

        # other static db info 
        self.custom_db_info["add_last_action_to_obs"] = self._add_prev_actions_stats_to_obs
        self.custom_db_info["twist_meas_is_base_local"] = self._twist_meas_is_base_local
        self.custom_db_info["use_pof0"] = self._use_pof0
        self.custom_db_info["pof0"] = self._pof0
        self.custom_db_info["action_repeat"] = self._action_repeat

    def _custom_post_init(self):
        # overriding parent's defaults 
        self._reward_thresh_lb[:, :]=0.0 # neg rewards can be nasty depending on the algorithm
        self._reward_thresh_ub[:, :]=torch.inf

        self._obs_threshold_lb = -1e3 # used for clipping observations
        self._obs_threshold_ub = 1e3

        v_cmd_max = self.max_ref
        omega_cmd_max = self.max_ref
        self._actions_lb[:, 0:3] = -v_cmd_max 
        self._actions_ub[:, 0:3] = v_cmd_max  
        self._actions_lb[:, 3:6] = -omega_cmd_max # twist cmds
        self._actions_ub[:, 3:6] = omega_cmd_max  
        self._actions_lb[:, 6:10] = -1.0 # contact flags
        self._actions_ub[:, 6:10] = 1.0 

        # some aux data to avoid allocations at training runtime
        self._robot_twist_meas_h = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu).detach().clone()
        self._robot_twist_meas_b = self._robot_twist_meas_h.detach().clone()
        self._robot_twist_meas_w = self._robot_twist_meas_h.detach().clone()
        self._task_ref_h = self._robot_twist_meas_h.detach().clone()
        
        # task aux objs
        device = "cuda" if self._use_gpu else "cpu"
        self._task_err_scaling = torch.zeros((self._n_envs, 1),dtype=self._dtype,device=device)

        self._pof1_b = torch.full(size=(self._n_envs,1),dtype=self._dtype,device=device,fill_value=1-self._pof0)
        self._bernoulli_coeffs = self._pof1_b.clone()
        self._bernoulli_coeffs[:, :] = 1.0

        self._zero_t_aux = torch.zeros((self._n_envs, 1),dtype=self._dtype,device=device)
        self._zero_t_aux_cpu = torch.zeros((self._n_envs, 1),dtype=self._dtype,device="cpu")

        if self._add_prev_actions_stats_to_obs:
            self._defaut_bf_action[:, :] = (self._actions_ub+self._actions_lb)/2.0

        if self._use_vel_err_sig_smoother:
            vel_err_proxy=self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu).detach().clone()
            self._smoothing_horizon=1.0
            self._target_smoothing=0.05
            self._vel_err_smoother=ExponentialSignalSmoother(
                name="VelErrorSmoother",
                signal=vel_err_proxy, # same dimension of vel error
                update_dt=self._substep_dt,
                smoothing_horizon=self._smoothing_horizon,
                target_smoothing=self._target_smoothing,
                debug=self._is_debug,
                dtype=self._dtype,
                use_gpu=self._use_gpu)
            self.custom_db_info["smoothing_horizon"]=self._smoothing_horizon
            self.custom_db_info["target_smoothing"]=self._target_smoothing

    def get_file_paths(self):
        paths=super().get_file_paths()
        paths.append(self._this_child_path)        
        return paths

    def get_aux_dir(self):

        aux_dirs = []
        path_getter = PathsGetter()

        aux_dirs.append(path_getter.RHCDIR)
        return aux_dirs

    def _get_reward_scaling(self):
        if self._single_task_ref_per_episode:
            return self._n_steps_task_rand_ub
        else:
            return self._episode_timeout_ub
        
    def _check_sub_truncations(self):
        # overrides parent
        sub_truncations = self._sub_truncations.get_torch_mirror(gpu=self._use_gpu)
        sub_truncations[:, 0:1] = self._ep_timeout_counter.time_limits_reached()
        if self._single_task_ref_per_episode:
            sub_truncations[:, 1:2] = self._task_rand_counter.time_limits_reached()
    
    def _custom_reset(self): # reset if truncated
        if self._single_task_ref_per_episode:
            return None
        else:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu).cpu()
    
    def _pre_step(self): 
        pass

    def _custom_post_step(self,episode_finished):
        # executed after checking truncations and terminations and remote env reset
        if self._use_gpu:
            time_to_rand_or_ep_finished = torch.logical_or(self._task_rand_counter.time_limits_reached().cuda(),episode_finished)
            self.randomize_task_refs(env_indxs=time_to_rand_or_ep_finished.flatten())
        else:
            time_to_rand_or_ep_finished = torch.logical_or(self._task_rand_counter.time_limits_reached(),episode_finished)
            self.randomize_task_refs(env_indxs=time_to_rand_or_ep_finished.flatten())
                
        if self._vel_err_smoother is not None: # reset smoother
            self._vel_err_smoother.reset(to_be_reset=episode_finished.flatten())

    def _custom_substep_post_substepping(self):
        pass

    def _apply_actions_to_rhc(self):
        
        agent_action = self.get_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_twist_ref = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        # rhc_latest_p_ref = self._rhc_refs.rob_refs.root_state.get(data_type="p", gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)

        # rhc_latest_twist_ref[:, 0:2] = agent_action[:, 0:2] # lin vel cmd
        # rhc_latest_p_ref[:, 2:3] = agent_action[:, 2:3] # h cmds
        # rhc_latest_twist_ref[:, 3:6] = agent_action[:, 3:6] # omega cmd

        rhc_latest_twist_ref[:, 0:6] = agent_action[:, 0:6]
        
        # self._rhc_refs.rob_refs.root_state.set(data_type="p", data=rhc_latest_p_ref,
        #                                     gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_ref,
                                            gpu=self._use_gpu) 
        
        # contact flags
        rhc_latest_contact_ref[:, :] = agent_action[:, 6:10] > 0 # keep contact if agent action > 0

        # actually apply actions to controller
        if self._use_gpu:
            # GPU->CPU --> we cannot use asynchronous data transfer since it's unsafe
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True,non_blocking=False)
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)

    def _fill_obs(self,
            obs: torch.Tensor):

        # measured stuff
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        robot_jnt_q_meas = self._robot_state.jnts_state.get(data_type="q",gpu=self._use_gpu)
        robot_twist_meas = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu)
        robot_jnt_v_meas = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
        
        # refs
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)

        next_idx=0
        obs[:, next_idx:(next_idx+4)] = robot_q_meas # [w, i, j, k] (IsaacSim convention)
        next_idx+=4
        # measured angular velocity
        if not self._twist_meas_is_base_local: # measurement read on shared mem is in
            # world (simulator) world frame -> we want it base-local
            world2base_frame(t_w=robot_twist_meas,q_b=robot_q_meas,t_out=self._robot_twist_meas_b)
            obs[:, next_idx:(next_idx+3)] = self._robot_twist_meas_b[:,3:6]
        else:  # measurement read on shared mem is already in local base frame
            obs[:, next_idx:(next_idx+3)] = robot_twist_meas[:,3:6]
        next_idx+=3
        obs[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_q_meas
        next_idx+=self._n_jnts
        obs[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_v_meas
        next_idx+=self._n_jnts
        if self._add_internal_rhc_q_to_obs:
            obs[:, next_idx:(next_idx+4)] = self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu)
            next_idx+=4
        if self._add_rhc_root_twist_to_obs:
            obs[:, next_idx:(next_idx+6)] = self._rhc_cmds.root_state.get(data_type="twist",gpu=self._use_gpu)
            next_idx+=6
        if self._add_rhc_avrg_root_twist_to_obs:
            obs[:, next_idx:(next_idx+6)] = self._get_avrg_rhc_root_twist() # usually in world frame
            next_idx+=6
        if self._add_contact_idx_to_obs:
            obs[:, next_idx:(next_idx+len(self.contact_names))] = self._rhc_fcn(gpu=self._use_gpu)
            next_idx+=len(self.contact_names)
        if self._add_rhc_fz_to_obs:
            obs[:, next_idx:(next_idx+len(self.contact_names))] = self._rhc_fz(gpu=self._use_gpu, node_idx=0)
            next_idx+=len(self.contact_names)
        obs[:, next_idx:(next_idx+2)] = agent_twist_ref[:, 0:2] # high lev agent refs
        next_idx+=2
        if self._add_fail_idx_to_obs:
            obs[:, next_idx:(next_idx+1)] = self._rhc_fail_idx(gpu=self._use_gpu)
            next_idx+=1
        if self._add_prev_actions_stats_to_obs:
            self._prev_act_idx=next_idx
            obs[:, next_idx:(next_idx+self.actions_dim())]=self._act_mem_buffer.get(idx=0) # last obs
            next_idx+=self.actions_dim()
            obs[:, next_idx:(next_idx+self.actions_dim())]=self._act_mem_buffer.mean(clone=False)
            next_idx+=self.actions_dim()
            obs[:, next_idx:(next_idx+self.actions_dim())]=self._act_mem_buffer.std(clone=False)

    def _get_custom_db_data(self, 
            episode_finished):
        episode_finished = episode_finished.cpu()
        self.custom_db_data["ContactIndex"].update(new_data=self._rhc_fcn(gpu=False), 
                                    ep_finished=episode_finished)
        self.custom_db_data["AgentTwistRefs"].update(new_data=self._agent_refs.rob_refs.root_state.get(data_type="twist",
                                                                                            gpu=False), 
                                    ep_finished=episode_finished)
        self.custom_db_data["RhcFailIdx"].update(new_data=self._rhc_fail_idx(gpu=False), 
                                    ep_finished=episode_finished)
        
    def _tot_mech_pow(self, jnts_vel, jnts_effort, weighted:bool=True):
        if weighted:
            tot_weighted_mech_power = torch.sum((jnts_effort*jnts_vel)*self._power_penalty_weights, dim=1, keepdim=True)/self._power_penalty_weights_sum
            return tot_weighted_mech_power
        else:
            tot_mech_power = torch.sum((jnts_effort*jnts_vel), dim=1, keepdim=True)
            return tot_mech_power
    
    def _drained_mech_pow(self, jnts_vel, jnts_effort):

        mech_pow_jnts=(jnts_effort*jnts_vel)*self._power_penalty_weights
        mech_pow_jnts.clamp_(0.0,torch.inf) # do not account for regenerative power
        drained_mech_pow_tot = torch.sum(mech_pow_jnts, dim=1, keepdim=True)/self._power_penalty_weights_sum
        return drained_mech_pow_tot

    def _jnt_vel_penalty(self, task_ref, jnts_vel):
        delta = 0.01 # [m/s]
        ref_norm = task_ref.norm(dim=1,keepdim=True)
        above_thresh = ref_norm >= delta
        jnts_vel_sqrd=jnts_vel*jnts_vel
        jnts_vel_sqrd[above_thresh.flatten(), :]=0 # no penalty for refs > thresh
        weighted_jnt_vel = torch.sum((jnts_vel_sqrd)*self._jnt_vel_penalty_weights, dim=1, keepdim=True)/self._jnt_vel_penalty_weights_sum
        return weighted_jnt_vel
    
    def _task_perc_err_wms(self, task_ref, task_meas, weights):
        ref_norm = task_ref.norm(dim=1,keepdim=True)
        epsi=1e-3
        self._task_err_scaling[:, :] = ref_norm+epsi
        task_perc_err=self._task_err_wms(task_ref=task_ref, task_meas=task_meas, 
            scaling=self._task_err_scaling, weights=weights)
        perc_err_thresh=2.0 # no more than perc_err_thresh*100 % error on each dim
        task_perc_err.clamp_(0.0,perc_err_thresh**2) 
        return task_perc_err
    
    def _task_err_wms(self, task_ref, task_meas, scaling, weights):
        task_error = (task_ref-task_meas)/scaling
        weighted_error=task_error*weights
        if self._vel_err_smoother is not None:
            self._vel_err_smoother.update(new_signal=weighted_error,
                ep_finished=None # reset done externally
                )
            weighted_error=self._vel_err_smoother.get()
        task_wmse = torch.sum(weighted_error*weighted_error, dim=1, keepdim=True)/torch.sum(weights).item()
        return task_wmse # weighted mean square error (along task dimension)
    
    def _task_perc_err_lin(self, task_ref, task_meas, weights):
        task_wmse = self._task_perc_err_wms(task_ref=task_ref, task_meas=task_meas,
            weights=weights)
        return task_wmse.sqrt()
    
    def _task_err_lin(self, task_ref, task_meas, weights):
        self._task_err_scaling[:, :] = 1
        task_wmse = self._task_err_wms(task_ref=task_ref, task_meas=task_meas, 
            scaling=self._task_err_scaling, weights=weights)
        return task_wmse.sqrt()
    
    def _rhc_fail_idx(self, gpu: bool):
        rhc_fail_idx = self._rhc_status.rhc_fail_idx.get_torch_mirror(gpu=gpu)
        return self._rhc_fail_idx_scale*rhc_fail_idx
    
    def _rhc_fcn(self, gpu: bool):
        step_var = self._rhc_status.rhc_fcn.get_torch_mirror(gpu=gpu)
        to_be_cat = []
        for i in range(len(self.contact_names)):
            start_idx=i*self.n_nodes
            end_idx=i*self.n_nodes+self.n_nodes
            to_be_cat.append(torch.sum(step_var[:, start_idx:end_idx], dim=1, keepdim=True)/self.n_nodes)
        return self._rhc_fcn_scale * torch.cat(to_be_cat, dim=1) 
    
    def _rhc_fz(self, gpu: bool, node_idx:int=0):
        step_var = self._rhc_status.rhc_fcn.get_torch_mirror(gpu=gpu)
        to_be_cat = []
        for i in range(len(self.contact_names)):
            start_idx=i*self.n_nodes+node_idx
            contact_fz=step_var[:, start_idx:start_idx+1]
            to_be_cat.append(contact_fz)
        return self._rhc_fcn_scale * torch.cat(to_be_cat, dim=1)
    
    def _weighted_actions_diff(self, gpu: bool, obs, next_obs):

        if self._add_prev_actions_stats_to_obs:
            prev_act=obs[:,self._prev_act_idx:(self._prev_act_idx+self.actions_dim())]
            action_now=next_obs[:,self._prev_act_idx:(self._prev_act_idx+self.actions_dim())]
            weighted_actions_diff = torch.sum((action_now-prev_act)*self._action_diff_weights,dim=1,keepdim=True)/self._action_diff_w_sum
            return weighted_actions_diff
        else: # not available
            if gpu:
                return self._zero_t_aux
            else:
                return self._zero_t_aux_cpu

    def _compute_substep_rewards(self,
                    obs: torch.Tensor,
                    next_obs: torch.Tensor):
        
        # task_error_fun = self._task_err_lin
        task_error_fun = self._task_perc_err_lin
        
        task_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu) # high level agent refs (hybrid twist)
        substep_avrg_root_twist_w=self._get_avrg_substep_root_twist()
        
        task_error = task_error_fun(task_meas=substep_avrg_root_twist_w, 
            task_ref=task_ref,
            weights=self._task_err_weights)
        
        robot_q_before_stepping=obs[:, 0:4] # we need the starting base orientation to 
        # traslate the reference to the MPC's world frame (which in open loop does not 
        # coincide with the robot's one)
        w2hor_frame(t_w=task_ref, q_b=robot_q_before_stepping, t_out=self._task_ref_h)
        task_pred_error = task_error_fun(task_meas=self._get_avrg_rhc_root_twist(), 
            task_ref=self._task_ref_h,
            weights=self._task_pred_err_weights)
        
        # mech power
        jnts_vel = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
        jnts_effort = self._robot_state.jnts_state.get(data_type="eff",gpu=self._use_gpu)
        # weighted_mech_power = self._tot_mech_pow(jnts_vel=jnts_vel, 
        #                                     jnts_effort=jnts_effort)
        weighted_mech_power = self._drained_mech_pow(jnts_vel=jnts_vel, 
                                            jnts_effort=jnts_effort)
        weighted_jnt_vel = self._jnt_vel_penalty(task_ref=task_ref,jnts_vel=jnts_vel)

        sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)
        sub_rewards[:, 0:1] = self._task_offset-self._task_scale*task_error
        sub_rewards[:, 1:2] = self._task_pred_offset-self._task_pred_scale*task_pred_error
        sub_rewards[:, 2:3] = self._power_offset - self._power_scale * weighted_mech_power
        sub_rewards[:, 3:4] = self._jnt_vel_offset - self._jnt_vel_scale * weighted_jnt_vel
        sub_rewards[:, 4:5] = self._rhc_fail_idx_offset - self._rhc_fail_idx_rew_scale* self._rhc_fail_idx(gpu=self._use_gpu)
        sub_rewards[:, 5:6] = self._health_value # health reward
        sub_rewards[:, 6:7] = self._actions_diff_rew_offset - \
                                        self._actions_diff_scale*self._weighted_actions_diff(gpu=self._use_gpu,
                                                                        obs=obs,next_obs=next_obs) # action regularization reward

    def _randomize_task_refs(self,
        env_indxs: torch.Tensor = None):
        
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        if self._use_pof0: # sample from bernoulli distribution
            torch.bernoulli(input=self._pof1_b,out=self._bernoulli_coeffs) # by default bernoulli_coeffs are 1 if not _use_pof0
        if env_indxs is None:
            random_uniform=torch.full_like(agent_twist_ref_current, fill_value=0.0)
            torch.nn.init.uniform_(random_uniform, a=-1, b=1)
            agent_twist_ref_current[:, :] = random_uniform*self._twist_ref_scale + self._twist_ref_offset
            agent_twist_ref_current[:, :] = agent_twist_ref_current*self._bernoulli_coeffs
        else:
            random_uniform=torch.full_like(agent_twist_ref_current[env_indxs, :], fill_value=0.0)
            torch.nn.init.uniform_(random_uniform, a=-1, b=1)
            agent_twist_ref_current[env_indxs, :] = random_uniform * self._twist_ref_scale + self._twist_ref_offset
            agent_twist_ref_current[env_indxs, :] = agent_twist_ref_current[env_indxs, :]*self._bernoulli_coeffs[env_indxs, :]
            
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
        obs_names[4] = "omega_x"
        obs_names[5] = "omega_y"
        obs_names[6] = "omega_z"
        next_idx+=7
        jnt_names = self._robot_state.jnt_names()
        for i in range(self._n_jnts): # jnt obs (pos):
            obs_names[next_idx+i] = f"q_{jnt_names[i]}"
        next_idx+=self._n_jnts
        for i in range(self._n_jnts): # jnt obs (v):
            obs_names[next_idx+i] = f"v_{jnt_names[i]}"
        next_idx+=self._n_jnts
        if self._add_internal_rhc_q_to_obs:
            obs_names[next_idx] = "q_w_rhc"
            obs_names[next_idx+1] = "q_i_rhc"
            obs_names[next_idx+2] = "q_j_rhc"
            obs_names[next_idx+3] = "q_k_rhc"
            next_idx+=4
        if self._add_rhc_root_twist_to_obs:
            obs_names[next_idx] = "in_vel_x_rhc" 
            obs_names[next_idx+1] = "lin_vel_y_rhc"
            obs_names[next_idx+2] = "lin_vel_z_rhc"
            obs_names[next_idx+3] = "omega_x_rhc" 
            obs_names[next_idx+4] = "omega_y_rhc"
            obs_names[next_idx+5] = "omega_z_rhc"
            next_idx+=6

        if self._add_rhc_avrg_root_twist_to_obs:
            obs_names[next_idx] = "avrg_lin_vel_x_rhc" 
            obs_names[next_idx+1] = "avrg_lin_vel_y_rhc"
            obs_names[next_idx+2] = "avrg_lin_vel_z_rhc"
            obs_names[next_idx+3] = "avrg_omega_x_rhc" 
            obs_names[next_idx+4] = "avrg_omega_y_rhc"
            obs_names[next_idx+5] = "avrg_omega_z_rhc"
            next_idx+=6
        if self._add_contact_idx_to_obs:
            i = 0
            for contact in self.contact_names:
                obs_names[next_idx+i] = f"contact_idx_{contact}"
                i+=1        
            next_idx+=len(self.contact_names)
        if self._add_rhc_fz_to_obs:
            i = 0
            for contact in self.contact_names:
                obs_names[next_idx+i] = f"rhc_fz_{contact}"
                i+=1        
            next_idx+=len(self.contact_names)
        obs_names[next_idx] = "lin_vel_x_ref" # specified in the "horizontal frame"
        obs_names[next_idx+1] = "lin_vel_y_ref"
        next_idx+=2
        if self._add_fail_idx_to_obs:
            obs_names[next_idx] = "rhc_fail_idx"
            next_idx+=1
        if self._add_prev_actions_stats_to_obs:
            action_names = self._get_action_names()
            for prev_act_idx in range(self.actions_dim()):
                obs_names[next_idx+prev_act_idx] = action_names[prev_act_idx]+f"_prev"
            next_idx+=self.actions_dim()
            for prev_act_mean in range(self.actions_dim()):
                obs_names[next_idx+prev_act_mean] = action_names[prev_act_mean]+f"_avrg"
            next_idx+=self.actions_dim()
            for prev_act_mean in range(self.actions_dim()):
                obs_names[next_idx+prev_act_mean] = action_names[prev_act_mean]+f"_std"
        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd" # twist commands from agent to RHC controller
        action_names[1] = "vy_cmd"
        action_names[2] = "vz_cmd"
        action_names[3] = "roll_twist_cmd"
        action_names[4] = "pitch_twist_cmd"
        action_names[5] = "yaw_twist_cmd"
        action_names[6] = "contact_0"
        action_names[7] = "contact_1"
        action_names[8] = "contact_2"
        action_names[9] = "contact_3"

        return action_names

    def _get_rewards_names(self):

        n_rewards = 7
        reward_names = [""] * n_rewards

        reward_names[0] = "task_error"
        reward_names[1] = "task_pred_error"
        reward_names[2] = "mech_power"
        reward_names[3] = "jnt_vel"
        reward_names[4] = "rhc_fail_idx"
        reward_names[5] = "health"
        reward_names[6] = "action_reg"

        return reward_names

    def _get_sub_trunc_names(self):
        sub_trunc_names = []
        sub_trunc_names.append("ep_timeout")
        if self._single_task_ref_per_episode:
            sub_trunc_names.append("task_ref_rand")
        return sub_trunc_names

