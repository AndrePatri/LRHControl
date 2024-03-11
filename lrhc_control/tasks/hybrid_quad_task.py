from lrhc_control.tasks.lrhc_task import LRHcIsaacTask

from lrhc_control.utils.hybrid_quad_xrdf_gen import get_xrdf_cmds_isaac

import numpy as np
import torch

class HybridQuadTask(LRHcIsaacTask):
    
    def __init__(self, 
            integration_dt: float,
            robot_name: str,
            robot_pkg_name: str,
            num_envs = 1,
            device = "cuda", 
            cloning_offset: np.array = None,
            replicate_physics: bool = True,
            solver_position_iteration_count: int = 4,
            solver_velocity_iteration_count: int = 1,
            solver_stabilization_thresh: float = 1e-5,
            offset=None, 
            env_spacing = 5.0, 
            spawning_radius = 1.0, 
            use_flat_ground = True,
            default_jnt_stiffness = 100.0,
            default_jnt_damping = 10.0,
            default_wheel_stiffness = 0.0,
            default_wheel_damping = 10.0,
            startup_jnt_stiffness = 50,
            startup_jnt_damping = 5,
            startup_wheel_stiffness = 0.0,
            startup_wheel_damping = 10.0,
            contact_prims = None,
            contact_offsets = None,
            sensor_radii = None,
            use_diff_velocities = True,
            override_art_controller = False,
            dtype: torch.dtype = torch.float32,
            debug_mode_jnt_imp = False):
        
        self.hybrid_quad_rob_name = robot_name
        self.hybrid_quad_robot_pkg_name = robot_pkg_name

        robot_names = [self.hybrid_quad_rob_name]
        robot_pkg_names = [self.hybrid_quad_robot_pkg_name]
        
        name = self.__class__.__name__

        super().__init__(integration_dt=integration_dt,
                robot_names=robot_names,
                robot_pkg_names=robot_pkg_names, 
                contact_prims=contact_prims,
                contact_offsets=contact_offsets,
                sensor_radii=sensor_radii,
                num_envs=num_envs,
                device=device, 
                cloning_offset=cloning_offset,
                replicate_physics=replicate_physics,
                solver_position_iteration_count=solver_position_iteration_count,
                solver_velocity_iteration_count=solver_velocity_iteration_count,
                solver_stabilization_thresh=solver_stabilization_thresh,
                offset=offset, 
                env_spacing=env_spacing, 
                spawning_radius=spawning_radius,
                use_flat_ground=use_flat_ground,
                default_jnt_stiffness=default_jnt_stiffness,
                default_jnt_damping=default_jnt_damping,
                default_wheel_stiffness=default_wheel_stiffness,
                default_wheel_damping=default_wheel_damping,
                startup_jnt_stiffness=startup_jnt_stiffness,
                startup_jnt_damping=startup_jnt_damping,
                startup_wheel_stiffness=startup_wheel_stiffness,
                startup_wheel_damping=startup_wheel_damping,
                override_art_controller=override_art_controller,
                use_diff_velocities=use_diff_velocities,
                dtype=dtype,
                debug_mode_jnt_imp=debug_mode_jnt_imp)

    def _xrdf_cmds(self):

        cmds = get_xrdf_cmds_isaac(n_robots=1, 
                    robot_pkg_name=self.hybrid_quad_robot_pkg_name) 
        
        return cmds