from omni_custom_gym.tasks.custom_task import CustomTask

from control_cluster_utils.utilities.control_cluster_defs import RobotClusterCmd

import numpy as np
import torch

from lrhc_examples.utils.xrdf_gen import get_xrdf_cmds_isaac

class AliengoExampleTask(CustomTask):
    def __init__(self, 
                cluster_dt: float, 
                integration_dt: float,
                num_envs = 1,
                device = "cuda", 
                cloning_offset: np.array = np.array([0.0, 0.0, 0.0]),
                replicate_physics: bool = True,
                offset=None, 
                env_spacing = 5.0, 
                use_flat_ground = True,
                default_jnt_stiffness = 300.0,
                default_jnt_damping = 20.0,
                dtype = torch.float64) -> None:

        # trigger __init__ of parent class
        CustomTask.__init__(self,
                    name = self.__class__.__name__, 
                    robot_name = "aliengo",
                    num_envs = num_envs,
                    device = device, 
                    cloning_offset = cloning_offset,
                    replicate_physics = replicate_physics,
                    offset = offset, 
                    env_spacing = env_spacing, 
                    use_flat_ground = use_flat_ground,
                    default_jnt_stiffness = default_jnt_stiffness,
                    default_jnt_damping = default_jnt_damping,
                    dtype = dtype)
        
        self.cluster_dt = cluster_dt
        self.integration_dt = integration_dt
        
    def _xrdf_cmds(self):

        cmds = get_xrdf_cmds_isaac()

        return cmds
      
    def post_reset(self):
        # self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        # self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # # randomize all envs
        # indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        # self.reset(indices)

        a = 1
    
    def reset(self, env_ids=None):

        super().reset()

    def pre_physics_step(self, 
            actions: RobotClusterCmd = None, 
            is_first_control_step = False) -> None:
        
        if is_first_control_step:

            no_gains_pos = torch.full((self.num_envs, self.robot_n_dofs), 
                        100.0, 
                        device = self.torch_device, 
                        dtype=self.torch_dtype)
            no_gains_vel = torch.full((self.num_envs, self.robot_n_dofs), 
                        10, 
                        device = self.torch_device, 
                        dtype=self.torch_dtype)
            self._jnt_imp_controller.set_gains(pos_gains = no_gains_pos,
                                vel_gains = no_gains_vel)
        
        if actions is not None:
            
            self._jnt_imp_controller.set_refs(pos_ref = actions.jnt_cmd.q, 
                                            vel_ref = actions.jnt_cmd.v, 
                                            eff_ref = actions.jnt_cmd.eff)
                    
            self._jnt_imp_controller.apply_refs()

        # print("cmd debug" + "\n" + 
        #         "q_cmd: " + str(actions.jnt_cmd.q) + "\n" + 
        #         "v_cmd: " + str(actions.jnt_cmd.v) + "\n" + 
        #         "eff_cmd: " + str(actions.jnt_cmd.eff))

    def get_observations(self):
        
        self._get_robots_state() # updates robot states

        return self.obs

    def calculate_metrics(self) -> None:

        # compute reward based on angle of pole and cart velocity
        reward = 0

        return reward

    def is_done(self) -> None:
        # cart_pos = self.obs[:, 0]
        # pole_pos = self.obs[:, 2]

        # # reset the robot if cart has reached reset_dist or pole is too far from upright
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        # self.resets = resets

        return True
