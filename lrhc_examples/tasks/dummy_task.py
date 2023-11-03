# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of LRhcExamples and distributed under the General Public License version 2 license.
# 
# LRhcExamples is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# LRhcExamples is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with LRhcExamples.  If not, see <http://www.gnu.org/licenses/>.
# 
from omni_robo_gym.tasks.custom_task import CustomTask

from control_cluster_bridge.utilities.control_cluster_defs import RobotClusterCmd

import numpy as np
import torch

from lrhc_examples.utils.xrdf_gen import get_xrdf_cmds_isaac_centauro
from lrhc_examples.utils.xrdf_gen import get_xrdf_cmds_isaac_aliengo

class ExampleTask(CustomTask):

    def __init__(self, 
                cluster_dt: float, 
                integration_dt: float,
                num_envs = 1,
                device = "cuda", 
                cloning_offset: np.array = np.array([0.0, 0.0, 0.0]),
                replicate_physics: bool = True,
                offset=None, 
                env_spacing = 5.0, 
                spawning_radius = 1.0,
                use_flat_ground = True,
                default_jnt_stiffness = 300.0,
                default_jnt_damping = 20.0,
                robot_names = ["aliengo"],
                robot_pkg_names = ["aliengo"],
                contact_prims = None,
                dtype = torch.float64) -> None:

        if contact_prims is None:

            contact_prims = {}

            for i in range(len(robot_names)):
                
                contact_prims[robot_names[i]] = [] # no contact sensors

        # trigger __init__ of parent class
        CustomTask.__init__(self,
                    name = self.__class__.__name__, 
                    robot_names = robot_names,
                    robot_pkg_names = robot_pkg_names,
                    num_envs = num_envs,
                    contact_prims = contact_prims,
                    device = device, 
                    cloning_offset = cloning_offset,
                    spawning_radius = spawning_radius,
                    replicate_physics = replicate_physics,
                    offset = offset, 
                    env_spacing = env_spacing, 
                    use_flat_ground = use_flat_ground,
                    default_jnt_stiffness = default_jnt_stiffness,
                    default_jnt_damping = default_jnt_damping,
                    dtype = dtype, 
                    self_collide = [False] * len(robot_names), 
                    fix_base = [False] * len(robot_names))
        
        self.cluster_dt = cluster_dt
        self.integration_dt = integration_dt
        
    def _xrdf_cmds(self):
        
        cmds = {}   

        for i in range(0, len(self.robot_names)):

            if (self.robot_pkg_names[i] == "centauro"):
                
                cmds.update(get_xrdf_cmds_isaac_centauro(self.robot_names[i]))
            
            if (self.robot_pkg_names[i] == "aliengo"):
                
                cmds.update(get_xrdf_cmds_isaac_aliengo(self.robot_names[i]))

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
            robot_name: str,
            actions: RobotClusterCmd = None) -> None:
        
        if actions is not None:
            
            self._jnt_imp_controllers[robot_name].set_refs(pos_ref = actions.jnt_cmd.q, 
                                            vel_ref = actions.jnt_cmd.v, 
                                            eff_ref = actions.jnt_cmd.eff)
                    
            self._jnt_imp_controllers[robot_name].apply_refs()

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
