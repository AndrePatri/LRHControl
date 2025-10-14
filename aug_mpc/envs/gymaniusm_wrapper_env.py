from aug_mpc.utils.wrappers.gymnasium_env import Gymnasium2LRHCEnv
import numpy as np

class GymnasiumWrapperEnv(Gymnasium2LRHCEnv):

    def __init__(namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            env_opts: Dict = {}):
        
        env_opts["handle_final_obs"]
        env_opts["gym_env_dtype"]=gym_env_dtype
        super().__init__(env_type=env_opts["env_type"],
            namespace=namespace,
            verbose=verbose,
            vlevel=vlevel,
            debug=debug,
            use_gpu=use_gpu,
            render=env_opts["render"],
            seed=env_opts["seed"],
            gym_env_dtype=env_opts["gym_env_dtype"],
            handle_final_obs=env_opts["handle_final_obs"])