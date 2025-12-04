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

            self._gait_scheduler_walk.step()
            self._gait_scheduler_trot.step()

            have_to_go_fast_linvel = rhc_twist_refs[:, 0:2].norm(dim=1, keepdim=True) > self._env_opts["walk_to_trot_thresh"]
            have_to_go_fast_omega = rhc_twist_refs[:, 3:6].norm(dim=1, keepdim=True) > self._env_opts["walk_to_trot_thresh_omega"]
            have_to_go_fast = torch.logical_or(have_to_go_fast_linvel, have_to_go_fast_omega)

            fast_and_demo = torch.logical_and(have_to_go_fast.flatten(), self._demo_envs_idxs_bool)
            have_to_go_slow_and_demo = ~fast_and_demo

            have_to_stop_linvel = rhc_twist_refs[:, 0:2].norm(dim=1, keepdim=True) < self._env_opts["stopping_thresh"]
            have_to_stop_omega = rhc_twist_refs[:, 3:6].norm(dim=1, keepdim=True) < self._env_opts["stopping_thresh"]
            have_to_stop = torch.logical_and(have_to_stop_linvel, have_to_stop_omega)
            stop_and_demo = torch.logical_and(have_to_stop.flatten(), self._demo_envs_idxs_bool)

            walk_signal = self._gait_scheduler_walk.get_signal(clone=True)[self._env_to_gait_sched_mapping[self._demo_envs_idxs_bool], :]
            is_contact_walk = walk_signal > self._gait_scheduler_walk.threshold()
            agent_action[self._demo_envs_idxs, 6:10] = 2.0 * is_contact_walk - 1.0

            if fast_and_demo.any():
                trot_signal = self._gait_scheduler_trot.get_signal(clone=True)[self._env_to_gait_sched_mapping[fast_and_demo], :]
                is_contact_trot = trot_signal > self._gait_scheduler_trot.threshold()
                agent_action[fast_and_demo, 6:10] = 2.0 * is_contact_trot - 1.0

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
