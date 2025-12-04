import torch
from typing import Dict

from EigenIPC.PyEigenIPC import VLevel

from aug_mpc.envs.fake_pos_env_with_demo import FakePosEnvWithDemo


class GaitSchedulingEnv(FakePosEnvWithDemo):
    """
    Same as FakePosEnvWithDemo but does not write MPC twist references
    to shared memory. Contact flags are still written so gait scheduling
    can be exercised without overriding twist refs.
    """

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            env_opts: Dict = {}):

        super().__init__(namespace=namespace,
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms,
            env_opts=env_opts)

        # Override gait transition defaults for this env
        self._env_opts["walk_to_trot_thresh"] = 0.4
        self._env_opts["walk_to_trot_thresh_omega"] = 0.5
        self._env_opts["stopping_thresh"] = self._env_opts.get("stopping_thresh", 0.02)
        # Adaptive gait cadence based on refs: (min,max) event intervals between successive lift-offs
        self.phase_period_walk_min=0.4
        self.phase_period_walk_max=0.8
        self.phase_period_trot_min=0.55
        self.phase_period_trot_max=1.2
        # phase accumulators (initialized lazily once demo envs are known)
        self._walk_phase = None
        self._trot_phase = None
        self._prev_contact_walk = None
        self._prev_contact_trot = None
        self._walk_offsets = None
        self._trot_offsets = None
        self._walk_init_mask = None
        self._trot_init_mask = None

    def _write_rhc_refs(self):
        """Do not touch MPC twist references; only push contact flags if needed."""
        if self._use_gpu:
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True, non_blocking=False)
            self._rhc_refs.rob_refs.contact_pos.synch_mirror(from_gpu=True,non_blocking=False)

        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)
        self._rhc_refs.rob_refs.contact_pos.synch_all(read=False, retry=True)

    def get_file_paths(self):
        paths = super().get_file_paths()
        import os
        paths.append(os.path.abspath(__file__))
        return paths

    def _override_actions_with_demo(self):
        """Use measured twist to drive gait scheduling while keeping twist refs untouched."""
        if self.demo_active():
            agent_action = self.get_actions()

            agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
            # use current MPC refs to decide gait mode
            rhc_twist_refs = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)

            # lazy init phase accumulators and offsets
            if self._walk_phase is None:
                self._walk_phase = torch.zeros((self._n_demo_envs, 1), device=self._device, dtype=self._dtype)
                self._trot_phase = torch.zeros((self._n_demo_envs, 1), device=self._device, dtype=self._dtype)
                # Walk sequence: FL -> FR -> BR -> BL (clockwise)
                # Contact liftoff occurs when sin crosses 0; offset = pi - desired_liftoff_phase
                self._walk_offsets = torch.tensor([[torch.pi, torch.pi/2, 0.0, 3*torch.pi/2]],
                                                  device=self._device, dtype=self._dtype)
                self._trot_offsets = torch.tensor([[0.0, torch.pi, torch.pi, 0.0]],
                                                  device=self._device, dtype=self._dtype)
                self._prev_contact_walk = torch.ones((self._n_demo_envs, 4), device=self._device, dtype=torch.bool)
                self._prev_contact_trot = torch.ones((self._n_demo_envs, 4), device=self._device, dtype=torch.bool)
                self._walk_init_mask = torch.zeros((self._n_demo_envs,), device=self._device, dtype=torch.bool)
                self._trot_init_mask = torch.zeros((self._n_demo_envs,), device=self._device, dtype=torch.bool)

            # adapt cadence based on ref speed
            speed = rhc_twist_refs[:, 0:2].norm(dim=1, keepdim=True) + rhc_twist_refs[:, 5:6].abs()
            # map speed to period in [min,max]
            def map_period(speed_val, pmin, pmax, smin=0.0, smax=1.5):
                speed_clamped = torch.clamp(speed_val, smin, smax)
                alpha = (speed_clamped - smin) / (smax - smin + 1e-6)
                return pmax - alpha * (pmax - pmin)
            walk_event = map_period(speed, self.phase_period_walk_min, self.phase_period_walk_max)
            trot_event = map_period(speed, self.phase_period_trot_min, self.phase_period_trot_max)
            # convert event intervals to full cycle periods expected by scheduler
            walk_period_full = walk_event * 4  # sequential legs
            trot_period_full = trot_event * 2  # two diagonal events
            # update phases with a chirp-style integrator
            omega_walk = 2 * torch.pi / torch.clamp(walk_period_full, min=1e-3)
            omega_trot = 2 * torch.pi / torch.clamp(trot_period_full, min=1e-3)
            dt = self._substep_dt * self._action_repeat
            self._walk_phase = (self._walk_phase + omega_walk * dt) % (2 * torch.pi)
            self._trot_phase = (self._trot_phase + omega_trot * dt) % (2 * torch.pi)

            # use planar speed and yaw rate only for transitions
            speed_lin = rhc_twist_refs[:, 0:2].norm(dim=1, keepdim=True)
            speed_yaw = rhc_twist_refs[:, 5:6].abs()
            have_to_go_fast_linvel = speed_lin > self._env_opts["walk_to_trot_thresh"]

            have_to_go_fast_omega = speed_yaw > self._env_opts["walk_to_trot_thresh_omega"]
            have_to_go_fast = torch.logical_or(have_to_go_fast_linvel, have_to_go_fast_omega)

            demo_mask = self._demo_envs_idxs_bool
            fast_and_demo = torch.logical_and(have_to_go_fast.flatten(), demo_mask)
            have_to_go_slow_and_demo = torch.logical_and(~have_to_go_fast.flatten(), demo_mask)

            have_to_stop_linvel = speed_lin < self._env_opts["stopping_thresh"]
            have_to_stop_omega = speed_yaw < self._env_opts["stopping_thresh"]
            have_to_stop = torch.logical_and(have_to_stop_linvel, have_to_stop_omega)
            stop_and_demo = torch.logical_and(have_to_stop.flatten(), demo_mask)

            # Walk contact pattern from phase accumulator (50% duty via sin > 0)
            walk_phase = self._walk_phase[self._env_to_gait_sched_mapping[self._demo_envs_idxs_bool], :]
            walk_sin = torch.sin(walk_phase + self._walk_offsets)
            is_contact_walk = walk_sin > 0.0
            walk_idxs = self._env_to_gait_sched_mapping[self._demo_envs_idxs_bool]
            liftoff_walk = torch.logical_and(self._prev_contact_walk, torch.logical_not(is_contact_walk))
            liftoff_walk = torch.logical_and(liftoff_walk, self._walk_init_mask[walk_idxs].unsqueeze(1))
            contact_flag_walk = torch.ones_like(is_contact_walk, dtype=self._dtype)
            contact_flag_walk[liftoff_walk] = -1.0
            agent_action[self._demo_envs_idxs, 6:10] = contact_flag_walk
            self._prev_contact_walk = is_contact_walk.detach().clone()
            self._walk_init_mask[walk_idxs] = True

            if fast_and_demo.any():
                trot_phase = self._trot_phase[self._env_to_gait_sched_mapping[fast_and_demo], :]
                trot_sin = torch.sin(trot_phase + self._trot_offsets)
                is_contact_trot = trot_sin > 0.0
                idxs = self._env_to_gait_sched_mapping[fast_and_demo]
                liftoff_trot = torch.logical_and(self._prev_contact_trot[idxs, :], torch.logical_not(is_contact_trot))
                liftoff_trot = torch.logical_and(liftoff_trot, self._trot_init_mask[idxs].unsqueeze(1))
                contact_flag_trot = torch.ones_like(is_contact_trot, dtype=self._dtype)
                contact_flag_trot[liftoff_trot] = -1.0
                agent_action[fast_and_demo, 6:10] = contact_flag_trot
                self._prev_contact_trot[idxs, :] = is_contact_trot.detach().clone()
                self._trot_init_mask[idxs] = True

            if stop_and_demo.any():
                agent_action[stop_and_demo, 6:10] = 1.0

            if self._env_opts["full_demo"]:
                if self._twist_smoother is not None:
                    if have_to_go_slow_and_demo.any():
                        agent_twist_ref_current[have_to_go_slow_and_demo, 0:6] = agent_twist_ref_current[have_to_go_slow_and_demo, 0:6]
                    self._twist_smoother.update(new_signal=agent_twist_ref_current[self._demo_envs_idxs, :])
                    agent_action[self._demo_envs_idxs, 0:6] = self._twist_smoother.get()
                else:
                    if have_to_go_slow_and_demo.any():
                        agent_twist_ref_current[have_to_go_slow_and_demo, 0:6] = agent_twist_ref_current[have_to_go_slow_and_demo, 0:6]
                    agent_action[self._demo_envs_idxs, 0:6] = agent_twist_ref_current[self._demo_envs_idxs, :]

                agent_action[stop_and_demo, 0:6] = 0.0
