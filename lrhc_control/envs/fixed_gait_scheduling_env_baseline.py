from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase
from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcStatus
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os
from lrhc_control.utils.episodic_data import EpisodicData

from lrhc_control.utils.gait_scheduler import QuadrupedGaitPatternGenerator, GaitScheduler

from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline

class FixedGaitSchedEnvBaseline(LinVelTrackBaseline):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000):

        super().__init__(namespace=namespace,
            actions_dim=4, # only contacts
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms)

    def _custom_post_init(self):
        
        super()._custom_post_init()

        self._agent_twist_ref_h = self._robot_twist_meas_h.clone()
        
        phase_period=2.0
        update_dt = self._substep_dt*self._action_repeat
        self._pattern_gen = QuadrupedGaitPatternGenerator(phase_period=phase_period)
        gait_params_walk = self._pattern_gen.get_params("walk")
        n_phases = gait_params_walk["n_phases"]
        phase_period = gait_params_walk["phase_period"]
        phase_offset = gait_params_walk["phase_offset"]
        phase_thresh = gait_params_walk["phase_thresh"]
        self._gait_scheduler_walk = GaitScheduler(
            n_phases=n_phases,
            n_envs=self._n_envs,
            update_dt=update_dt,
            phase_period=phase_period,
            phase_offset=phase_offset,
            phase_thresh=phase_thresh,
            use_gpu=self._use_gpu,
            dtype=self._dtype
        )
        gait_params_trot = self._pattern_gen.get_params("trot")
        n_phases = gait_params_trot["n_phases"]
        phase_period = gait_params_trot["phase_period"]
        phase_offset = gait_params_trot["phase_offset"]
        phase_thresh = gait_params_trot["phase_thresh"]
        self._gait_scheduler_trot = GaitScheduler(
            n_phases=n_phases,
            n_envs=self._n_envs,
            update_dt=update_dt,
            phase_period=phase_period,
            phase_offset=phase_offset,
            phase_thresh=phase_thresh,
            use_gpu=self._use_gpu,
            dtype=self._dtype
        )

    def get_file_paths(self):
        paths=super().get_file_paths()
        paths.append(os.path.abspath(__file__))        
        return paths

    def _custom_post_step(self,episode_finished):
        super()._custom_post_step(episode_finished=episode_finished)
        # executed after checking truncations and terminations
        if self._use_gpu:
            self._gait_scheduler_walk.reset(to_be_reset=episode_finished.cuda().flatten())
            self._gait_scheduler_trot.reset(to_be_reset=episode_finished.cuda().flatten())
        else:
            self._gait_scheduler_walk.reset(to_be_reset=episode_finished.cpu().flatten())
            self._gait_scheduler_trot.reset(to_be_reset=episode_finished.cuda().flatten())

    def _apply_actions_to_rhc(self):
        # just override how actions are applied wrt base env

        agent_action = self.get_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_twist_ref = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)

        # overwriting agent actions with gait scheduler ones
        self._gait_scheduler_walk.step()
        self._gait_scheduler_trot.step()
        walk_to_trot_thresh=0.8 # [m/s]
        stopping_thresh=0.05
        have_to_go_fast=agent_twist_ref_current.norm(dim=1,keepdim=True)>walk_to_trot_thresh
        have_to_stop=agent_twist_ref_current.norm(dim=1,keepdim=True)<stopping_thresh
        # default to walk
        agent_action[:, :] = self._gait_scheduler_walk.get_signal(clone=True)
        # for fast enough refs, trot
        agent_action[have_to_go_fast.flatten(), :] = \
            self._gait_scheduler_trot.get_signal(clone=True)[have_to_go_fast.flatten(), :]
        agent_action[have_to_stop.flatten(), :] = 1.0
        
        # refs have to be applied in the MPC's horizontal frame
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        w2hor_frame(t_w=agent_twist_ref_current,q_b=robot_q_meas,t_out=self._agent_twist_ref_h)
        # 2D lin vel applied directly to MPC
        rhc_latest_twist_ref[:, 0:2] = self._agent_twist_ref_h[:, 0:2] # 2D lin vl
        rhc_latest_twist_ref[:, 5:6] = self._agent_twist_ref_h[:, 5:6] # yaw twist

        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_ref,
                                            gpu=self._use_gpu) 
        
        # agent sets contact flags
        rhc_latest_contact_ref[:, :] = agent_action[:, 0:4] > self._gait_scheduler_walk.threshold() # keep contact if agent action > 0

        if self._use_gpu:
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)
    
    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "contact_0"
        action_names[1] = "contact_1"
        action_names[2] = "contact_2"
        action_names[3] = "contact_3"

        return action_names