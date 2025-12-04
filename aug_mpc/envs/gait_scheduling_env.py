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
