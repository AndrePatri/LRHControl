from omni_robo_gym.tasks.isaac_task import IsaacTask

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds
from control_cluster_bridge.utilities.shared_data.jnt_imp_control import JntImpCntrlData

from typing import List, Dict

import numpy as np
import torch

from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import VLevel

class LRhcTaskBase():
    def __init__(self, 
        robot_names: List[str],
        robot_pkg_names: List[str],
        robot_pkg_prefix_paths: List[str],
        integration_dt: float,
        num_envs: int,
        device = "cuda",
        dtype:torch.dtype=torch.float32,
        debug:bool=False,
        dump_basepath:str="/tmp"
        ):
        a = 1

class LRHcIsaacTask(IsaacTask):
    
    def __init__(self, 
            robot_names: List[str],
            robot_pkg_names: List[str],
            robot_pkg_prefix_paths: List[str],
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
            contact_prims:Dict[str, List] = None,
            contact_offsets:Dict[str, List] = None,
            sensor_radii:Dict[str, List] = None,
            use_diff_velocities = False,
            override_art_controller = False,
            dtype = torch.float64,
            debug_enabled = False,
            dump_basepath: str = "/tmp") -> None:

        if cloning_offset is None:
            cloning_offset = np.array([[0.0, 0.0, 0.0]] * num_envs)

        # trigger __init__ of parent class
        IsaacTask.__init__(self,
                    name = self.__class__.__name__, 
                    integration_dt = integration_dt,
                    robot_names = robot_names,
                    robot_pkg_names = robot_pkg_names,
                    robot_pkg_prefix_paths = robot_pkg_prefix_paths,
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
                    use_diff_velocities = use_diff_velocities,
                    debug_enabled = debug_enabled,
                    dump_basepath = dump_basepath) 
        
        self.startup_jnt_stiffness = startup_jnt_stiffness
        self.startup_jnt_damping = startup_jnt_damping
        self.startup_wheel_stiffness = startup_wheel_stiffness
        self.startup_wheel_damping = startup_wheel_damping

        self.jnt_imp_cntrl_shared_data = {}
    
    def _custom_post_init(self):

        # will be called at the end of the post-initialization steps

        for i in range(0, len(self.robot_names)):
            
            robot_name = self.robot_names[i]
            self.jnt_imp_cntrl_shared_data[robot_name] = JntImpCntrlData(is_server = True, 
                                            n_envs = self.num_envs, 
                                            n_jnts = self.robot_n_dofs[robot_name],
                                            jnt_names = self.jnt_imp_controllers[robot_name].jnts_names,
                                            namespace = robot_name, 
                                            verbose = True, 
                                            force_reconnection = True,
                                            vlevel = VLevel.V2,
                                            use_gpu=self.using_gpu,
                                            safe=True)
            self.jnt_imp_cntrl_shared_data[robot_name].run()

    def _update_jnt_imp_cntrl_shared_data(self):

        if self._debug_enabled:
            
            for i in range(0, len(self.robot_names)):
            
                robot_name = self.robot_names[i]
                # updating all the jnt impedance data - > this may introduce a significant overhead,
                # on CPU and, if using GPU, also there 
                # disable this with debug_mode_jnt_imp when debugging is not necessary
                imp_data = self.jnt_imp_cntrl_shared_data[robot_name].imp_data_view
                # set data
                imp_data.set(data_type="pos_err",
                        data=self.jnt_imp_controllers[robot_name].pos_err(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="vel_err",
                        data=self.jnt_imp_controllers[robot_name].vel_err(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="pos_gains",
                        data=self.jnt_imp_controllers[robot_name].pos_gains(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="vel_gains",
                        data=self.jnt_imp_controllers[robot_name].vel_gains(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="eff_ff",
                        data=self.jnt_imp_controllers[robot_name].eff_ref(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="pos",
                        data=self.jnt_imp_controllers[robot_name].pos(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="pos_ref",
                        data=self.jnt_imp_controllers[robot_name].pos_ref(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="vel",
                        data=self.jnt_imp_controllers[robot_name].vel(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="vel_ref",
                        data=self.jnt_imp_controllers[robot_name].vel_ref(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="eff",
                        data=self.jnt_imp_controllers[robot_name].eff(),
                        gpu=self.using_gpu)
                imp_data.set(data_type="imp_eff",
                        data=self.jnt_imp_controllers[robot_name].imp_eff(),
                        gpu=self.using_gpu)
                # copy from GPU to CPU if using gpu
                if self.using_gpu:
                    imp_data.synch_mirror(from_gpu=True)
                # write copies to shared memory
                imp_data.synch_all(read=False, retry=False)
                        
    def post_reset(self):

        pass
    
    def reset(self, 
            env_indxs: torch.Tensor = None,
            robot_names: List[str]=None,
            randomize: bool = False):

        super().reset(env_indxs=env_indxs, 
                    robot_names=robot_names,
                    randomize=randomize)

    def _step_jnt_imp_control(self,
                        robot_name: str, 
                        actions: RhcCmds = None,
                        env_indxs: torch.Tensor = None):

        if env_indxs is not None and self._debug_enabled:
            if not isinstance(env_indxs, torch.Tensor):
                error = "Provided env_indxs should be a torch tensor of indexes!"
                Journal.log(self.__class__.__name__,
                    "_step_jnt_imp_control",
                    error,
                    LogType.EXCEP,
                    True)
            if self.using_gpu:
                if not env_indxs.device.type == "cuda":
                        error = "Provided env_indxs should be on GPU!"
                        Journal.log(self.__class__.__name__,
                        "_step_jnt_imp_control",
                        error,
                        LogType.EXCEP,
                        True)
            else:
                if not env_indxs.device.type == "cpu":
                    error = "Provided env_indxs should be on CPU!"
                    Journal.log(self.__class__.__name__,
                        "_step_jnt_imp_control",
                        error,
                        LogType.EXCEP,
                        True)
                
        # always updated imp. controller internal state (jnt imp control is supposed to be
        # always running)
        self.jnt_imp_controllers[robot_name].update_state(pos = self.jnts_q(robot_name=robot_name), 
                vel = self.jnts_v(robot_name=robot_name),
                eff = self.jnts_eff(robot_name=robot_name))

        if actions is not None and env_indxs is not None:
            # if new actions are received, also update references
            # (only use actions if env_indxs is provided)
            self.jnt_imp_controllers[robot_name].set_refs(
                    pos_ref = actions.jnts_state.get(data_type="q", gpu=self.using_gpu)[env_indxs, :], 
                    vel_ref = actions.jnts_state.get(data_type="v", gpu=self.using_gpu)[env_indxs, :], 
                    eff_ref = actions.jnts_state.get(data_type="eff", gpu=self.using_gpu)[env_indxs, :],
                    robot_indxs = env_indxs)

        # # jnt imp. controller actions are always applied
        self.jnt_imp_controllers[robot_name].apply_cmds()

        if self._debug_enabled:
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

class LRHcGzXBotTask(LRhcTaskBase):
    def __init__(self):
        a=1

class LRhcMJXBot2Task(LRhcTaskBase):
    def __init__(self):
        a=1 

class LRhcGzTask(LRhcTaskBase):
    def __init__(self):
        a=1

class LRhcMJTask(LRhcTaskBase):
    def __init__(self):
        a=1 