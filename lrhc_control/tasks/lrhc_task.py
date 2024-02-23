from omni_robo_gym.tasks.isaac_task import IsaacTask

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds
from control_cluster_bridge.utilities.shared_data.jnt_imp_control import JntImpCntrlData

from typing import List

import numpy as np
import torch

class LRHcIsaacTask(IsaacTask):
    
    def __init__(self, 
            robot_names: List[str],
            robot_pkg_names: List[str],
            integration_dt: float,
            num_envs: int,
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
            dtype = torch.float64,
            debug_mode_jnt_imp = False) -> None:

        if cloning_offset is None:
        
            cloning_offset = np.array([[0.0, 0.0, 0.0]] * num_envs)
        
        if contact_prims is None:

            contact_prims = {}

            for i in range(len(robot_names)):
                
                contact_prims[robot_names[i]] = [] # no contact sensors

        # trigger __init__ of parent class
        IsaacTask.__init__(self,
                    name = self.__class__.__name__, 
                    integration_dt = integration_dt,
                    robot_names = robot_names,
                    robot_pkg_names = robot_pkg_names,
                    num_envs = num_envs,
                    contact_prims = contact_prims,
                    contact_offsets = contact_offsets,
                    sensor_radii = sensor_radii,
                    device = device, 
                    cloning_offset = cloning_offset,
                    spawning_radius = spawning_radius,
                    replicate_physics = replicate_physics,
                    solver_position_iteration_count = solver_position_iteration_count,
                    solver_velocity_iteration_count = solver_velocity_iteration_count,
                    solver_stabilization_thresh = solver_stabilization_thresh,
                    offset = offset, 
                    env_spacing = env_spacing, 
                    use_flat_ground = use_flat_ground,
                    default_jnt_stiffness = default_jnt_stiffness,
                    default_jnt_damping = default_jnt_damping,
                    default_wheel_stiffness = default_wheel_stiffness,
                    default_wheel_damping = default_wheel_damping,
                    override_art_controller = override_art_controller,
                    dtype = dtype, 
                    self_collide = [False] * len(robot_names), 
                    fix_base = [False] * len(robot_names),
                    merge_fixed = [True] * len(robot_names),
                    debug_mode_jnt_imp = debug_mode_jnt_imp)
        
        self.use_diff_velocities = use_diff_velocities
        
        self._debug_mode_jnt_imp = debug_mode_jnt_imp

        self.startup_jnt_stiffness = startup_jnt_stiffness
        self.startup_jnt_damping = startup_jnt_damping
        self.startup_wheel_stiffness = startup_wheel_stiffness
        self.startup_wheel_damping = startup_wheel_damping

        self.jnt_imp_cntrl_shared_data = {}
    
    def _custom_post_init(self):

        # will be called at the end of the post-initialization steps

        for i in range(0, len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            from SharsorIPCpp.PySharsorIPC import VLevel

            self.jnt_imp_cntrl_shared_data[robot_name] = JntImpCntrlData(is_server = True, 
                                            n_envs = self.num_envs, 
                                            n_jnts = self.robot_n_dofs[robot_name],
                                            jnt_names = self.jnt_imp_controllers[robot_name].jnts_names,
                                            namespace = robot_name, 
                                            verbose = True, 
                                            force_reconnection = True,
                                            vlevel = VLevel.V2)

            self.jnt_imp_cntrl_shared_data[robot_name].run()

    def _update_jnt_imp_cntrl_shared_data(self):

        if self._debug_mode_jnt_imp:

            for i in range(0, len(self.robot_names)):
            
                robot_name = self.robot_names[i]

                success = True

                # updating all the jnt impedance data - > this introduces some overhead. 
                # disable this with debug_mode_jnt_imp when debugging is not necessary
                success = self.jnt_imp_cntrl_shared_data[robot_name].pos_err_view.write(
                    self.jnt_imp_controllers[robot_name].pos_err(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].vel_err_view.write(
                    self.jnt_imp_controllers[robot_name].vel_err(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].pos_gains_view.write(
                    self.jnt_imp_controllers[robot_name].pos_gains(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].vel_gains_view.write(
                    self.jnt_imp_controllers[robot_name].vel_gains(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].eff_ff_view.write(
                    self.jnt_imp_controllers[robot_name].eff_ref(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].pos_view.write(
                    self.jnt_imp_controllers[robot_name].pos(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].pos_ref_view.write(
                    self.jnt_imp_controllers[robot_name].pos_ref(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].vel_view.write(
                    self.jnt_imp_controllers[robot_name].vel(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].vel_ref_view.write(
                    self.jnt_imp_controllers[robot_name].vel_ref(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].eff_view.write(
                    self.jnt_imp_controllers[robot_name].eff(), 0, 0
                    ) and success
                success = self.jnt_imp_cntrl_shared_data[robot_name].imp_eff_view.write(
                    self.jnt_imp_controllers[robot_name].imp_eff(), 0, 0
                    ) and success

                if not success:
                    
                    message = f"[{self.__class__.__name__}]" + \
                        f"[{self._journal.status}]" + \
                        ": Could not update all jnt. imp. controller info on shared memory," + \
                        " probably because the data was already owned at the time of writing. Data might be lost"

                    print(message)
    
    def post_reset(self):

        pass
    
    def reset(self, 
            env_indxs: torch.Tensor = None,
            robot_names: List[str]=None):

        super().reset(env_indxs=env_indxs, 
                    robot_names=robot_names)

    def _step_jnt_imp_control(self,
                        robot_name: str, 
                        actions: RhcCmds = None,
                        env_indxs: torch.Tensor = None):

        if env_indxs is not None:

            if not isinstance(env_indxs, torch.Tensor):
                    
                msg = "Provided env_indxs should be a torch tensor of indexes!"
            
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]: " + msg)
        
        # always updated imp. controller internal state (jnt imp control is supposed to be
        # always running)
        success = self.jnt_imp_controllers[robot_name].update_state(pos = self.jnts_q(robot_name=robot_name), 
                                                    vel = self.jnts_v(robot_name=robot_name),
                                                    eff = None # not needed by jnt imp control
                                                    )
        
        if not all(success):
            
            print(success)

            exception = "Could not update the whole joint impedance state!!"
            
            raise Exception(exception)

        if actions is not None:
            
            # if new actions are received, also update references

            if env_indxs is not None:
                
                # only use actions if env_indxs is provided
                success = self.jnt_imp_controllers[robot_name].set_refs(
                                            pos_ref = actions.jnts_state.get_q(gpu=self.using_gpu)[env_indxs, :], 
                                            vel_ref = actions.jnts_state.get_v(gpu=self.using_gpu)[env_indxs, :], 
                                            eff_ref = actions.jnts_state.get_eff(gpu=self.using_gpu)[env_indxs, :],
                                            robot_indxs = env_indxs)

            if not all(success):
            
                print(success)

                exception = "Could not update jnt imp refs!!"
                
                raise Exception(exception)
        
        # # jnt imp. controller actions are always applied
        self.jnt_imp_controllers[robot_name].apply_cmds()

        self._update_jnt_imp_cntrl_shared_data() # only if debug_mode_jnt_imp is enabled
        
    def pre_physics_step(self, 
            robot_name: str, 
            actions: RhcCmds = None,
            env_indxs: torch.Tensor = None) -> None:

        # just step joint impedance control
        self._step_jnt_imp_control(robot_name = robot_name,
                                actions = actions,
                                env_indxs = env_indxs)

    def close(self):

        for i in range(0, len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            # closing shared memory
            self.jnt_imp_cntrl_shared_data[robot_name].close()

# class LRhcTraingTaks():

    