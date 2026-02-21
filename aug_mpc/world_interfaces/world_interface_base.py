from aug_mpc.controllers.rhc.augmpc_cluster_server import AugMpcClusterServer
from aug_mpc.utils.shared_data.remote_stepping import RemoteStepperClnt
from aug_mpc.utils.shared_data.remote_stepping import RemoteResetClnt
from aug_mpc.utils.shared_data.remote_stepping import RemoteResetRequest
from aug_mpc.utils.jnt_imp_control_base import JntImpCntrlBase
from aug_mpc.utils.hybrid_quad_xrdf_gen import get_xrdf_cmds
from aug_mpc.utils.xrdf_gen import generate_srdf, generate_urdf
from aug_mpc.utils.math_utils import quaternion_difference
from aug_mpc.utils.custom_arg_parsing import extract_custom_xacro_args, merge_xacro_cmds

from aug_mpc.utils.filtering import FirstOrderFilter

from mpc_hive.utilities.homing import RobotHomer
from mpc_hive.utilities.shared_data.jnt_imp_control import JntImpCntrlData

from EigenIPC.PyEigenIPC import VLevel, Journal, LogType, dtype
from EigenIPC.PyEigenIPC import StringTensorServer
from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper

from typing import List, Dict, TypeVar

import os
import inspect
import signal
import time

import numpy as np
import torch

from abc import ABC, abstractmethod

JntImpCntrlChild = TypeVar('JntImpCntrlChild', bound='JntImpCntrlBase')

class AugMPCWorldInterfaceBase(ABC):

    def __init__(self,
                robot_names: List[str],
                robot_urdf_paths: List[str],
                robot_srdf_paths: List[str],
                jnt_imp_config_paths: List[str],
                n_contacts: List[int],
                cluster_dt: List[float],
                use_remote_stepping: List[bool],
                name: str = "AugMPCWorldInterfaceBase",
                num_envs: int = 1,
                debug = False,
                verbose: bool = False,
                vlevel: VLevel = VLevel.V1,
                n_init_step: int = 0,
                timeout_ms: int = 60000,
                env_opts: Dict = None,
                use_gpu: bool = True,
                dtype: torch.dtype = torch.float32,
                dump_basepath: str = "/tmp",
                override_low_lev_controller: bool = False):

        # checks on input args
        # type checks
        if not isinstance(robot_names, List):
            exception = "robot_names must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(robot_urdf_paths, List):
            exception = "robot_urdf_paths must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(robot_srdf_paths, List):
            exception = "robot_srdf_paths must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(cluster_dt, List):
            exception = "cluster_dt must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(use_remote_stepping, List):
            exception = "use_remote_stepping must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(n_contacts, List):
            exception = "n_contacts must be a list (of integers)!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(jnt_imp_config_paths, List):
            exception = "jnt_imp_config_paths must be a list paths!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
        # dim checks
        if not len(robot_urdf_paths) == len(robot_names):
            exception = f"robot_urdf_paths has len {len(robot_urdf_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(robot_srdf_paths) == len(robot_names):
            exception = f"robot_srdf_paths has len {len(robot_srdf_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(cluster_dt) == len(robot_names):
            exception = f"cluster_dt has len {len(cluster_dt)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(use_remote_stepping) == len(robot_names):
            exception = f"use_remote_stepping has len {len(use_remote_stepping)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(robot_srdf_paths) == len(robot_names):
            exception = f"robot_srdf_paths has len {len(robot_srdf_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(jnt_imp_config_paths) == len(robot_names):
            exception = f"jnt_imp_config_paths has len {len(jnt_imp_config_paths)}" + \
             f" while robot_names {len(robot_names)}"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        self._remote_exit_flag=None

        self._name=name
        self._num_envs=num_envs
        self._debug=debug
        self._verbose=verbose
        self._vlevel=vlevel
        self._force_reconnection=True
        self._timeout_ms=timeout_ms
        self._use_gpu=use_gpu
        self._device = "cuda" if self._use_gpu else "cpu"
        self._dtype=dtype
        self._robot_names=robot_names
        self._env_opts={}
        self._env_opts["deact_when_failure"]=True
        self._env_opts["filter_jnt_vel"]=False
        self._env_opts["filter_cutoff_freq"]=10.0 # [Hz]
        self._env_opts["filter_sampling_rate"]=100 # rate at which state is filtered [Hz]
        self._env_opts["add_remote_exit_flag"]=False # add shared data server to trigger a remote exit
        self._env_opts["wheel_joint_patterns"]=["wheel"]
        self._env_opts["filter_wheel_pos_ref"]=True
        self._env_opts["zero_wheel_eff_ref"]=True

        self._env_opts["enable_height_sensor"]=False
        self._env_opts["height_sensor_resolution"]=0.16
        self._env_opts["height_sensor_pixels"]=10
        self._env_opts["height_sensor_lateral_offset"]=0.0
        self._env_opts["height_sensor_forward_offset"]=0.0

        self._env_opts["run_cluster_bootstrap"] = False
        
        self._filter_step_ssteps_freq=None

        self._env_opts.update(env_opts)

        self.step_counter = 0 # global step counter
        self._n_init_steps = n_init_step # n steps to be performed before applying solutions from control clusters
        self._srdf_dump_paths = robot_srdf_paths
        self._homers = {} 
        self._homing = None
        self._jnt_imp_cntrl_shared_data = {}
        self._jnt_imp_controllers = {}
        self._jnt_imp_config_paths = {}

        # control cluster data
        self.cluster_sim_step_counters = {}
        self.cluster_servers = {}
        self._trigger_sol = {}
        self._wait_sol = {}
        self._cluster_dt = {}
        self._robot_urdf_paths={}
        self._robot_srdf_paths={}
        self._contact_names={}
        self._num_contacts={}

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._cluster_dt[robot_name]=cluster_dt[i]
            self._robot_urdf_paths[robot_name]=robot_urdf_paths[i]
            self._robot_srdf_paths[robot_name]=robot_srdf_paths[i]
            self._contact_names[robot_name]=None
            self._num_contacts[robot_name]=n_contacts[i]
            self._jnt_imp_config_paths[robot_name]=jnt_imp_config_paths[i]
        # db data
        self.debug_data = {}
        self.debug_data["time_to_step_world"] = np.nan
        self.debug_data["time_to_get_states_from_env"] = np.nan
        self.debug_data["cluster_sol_time"] = {}
        self.debug_data["cluster_state_update_dt"] = {}
        self.debug_data["sim_time"] = {}
        self.debug_data["cluster_time"] = {}
        
        self._env_timer = time.perf_counter()

        # remote sim stepping options
        self._timeout = timeout_ms # timeout for remote stepping
        self._use_remote_stepping = use_remote_stepping
        # should use remote stepping
        self._remote_steppers = {}
        self._remote_resetters = {}
        self._remote_reset_requests = {}
        self._is_first_trigger = {}

        self._closed = False
             
        self._this_child_path=os.path.abspath(inspect.getfile(self.__class__))
        self._descr_dump_path=dump_basepath+"/"+f"{self.__class__.__name__}"
        self._urdf_dump_paths = {}
        self._srdf_dump_paths = {}
        self.xrdf_cmd_vals = [] # by default empty, needs to be overriden by
        # child class
        self._world_iface_files_server=None

        self._override_low_lev_controller=override_low_lev_controller

        self._root_p = {}
        self._root_q = {}
        self._jnts_q = {} 
        self._root_p_prev = {} # used for num differentiation
        self._root_q_prev = {} # used for num differentiation
        self._jnts_q_prev = {} # used for num differentiation
        self._root_v_prev = {} # used for num differentiation
        self._root_omega_prev = {} # used for num differentiation
        self._root_p_default = {} 
        self._root_q_default = {}
        self._jnts_q_default = {}
        
        self._gravity_normalized = {}
        self._gravity_normalized_base_loc = {}

        self._root_v = {}
        self._root_v_base_loc = {}
        self._root_v_default = {}
        self._root_omega = {}
        self._root_omega_base_loc = {}
        self._root_omega_default = {}
        self._root_a = {}
        self._root_a_base_loc = {}
        self._root_alpha = {}
        self._root_alpha_base_loc = {}

        self._jnts_v = {}
        self._jnt_vel_filter = {}
        self._jnts_v_default = {}
        self._jnts_eff = {}
        self._jnts_eff_default = {}

        self._root_pos_offsets = {} 
        self._root_q_offsets = {} 
        self._root_q_offsets_yaw = {}
        self._root_q_yaw_rel_ws = {}

        self._parse_env_opts()

        self._enable_height_shared = self._env_opts["enable_height_sensor"]
        self._height_sensor_resolution = self._env_opts["height_sensor_resolution"]
        self._height_sensor_pixels = self._env_opts["height_sensor_pixels"]

        self._pre_setup() # child's method

        self._init_world() # after this point all info from sim or robot is 
        # available

        self._publish_world_interface_files()
        
        setup_ok=self._setup()
        if not setup_ok:
            self.close()
        
        self._exit_request=False
        signal.signal(signal.SIGINT, self.signal_handler)   

    def signal_handler(self, sig, frame):
        Journal.log(self.__class__.__name__,
            "signal_handler",
            "received SIGINT -> cleaning up",
            LogType.WARN)
        self._exit_request=True
    
    def __del__(self):
        self.close()
    
    def is_closed(self):
        return self._closed
    
    def close(self) -> None:
        if not self._closed:
            for i in range(len(self._robot_names)):
                if self._robot_names[i] in self.cluster_servers:
                    self.cluster_servers[self._robot_names[i]].close()
                if self._use_remote_stepping[i]: # remote signaling
                    if self._robot_names[i] in self._remote_reset_requests:
                        self._remote_reset_requests[self._robot_names[i]].close()
                        self._remote_resetters[self._robot_names[i]].close()
                        self._remote_steppers[self._robot_names[i]].close()
                if self._robot_names[i] in self._jnt_imp_cntrl_shared_data:
                    jnt_imp_shared_data=self._jnt_imp_cntrl_shared_data[self._robot_names[i]]
                    if jnt_imp_shared_data is not None:
                        jnt_imp_shared_data.close()
            if self._remote_exit_flag is not None:
                self._remote_exit_flag.close()
            if self._world_iface_files_server is not None:
                self._world_iface_files_server.close()
            self._close()
            self._closed=True

    def _collect_world_interface_files(self):
        files = [self._this_child_path]
        # prefer generated URDF/SRDF if available, fallback to provided xacros
        if len(self._urdf_dump_paths) > 0:
            files.extend(list(self._urdf_dump_paths.values()))
        else:
            files.extend(list(self._robot_urdf_paths.values()))
        if len(self._srdf_dump_paths) > 0:
            files.extend(list(self._srdf_dump_paths.values()))
        else:
            files.extend(list(self._robot_srdf_paths.values()))
        files.extend(list(self._jnt_imp_config_paths.values()))
        # remove duplicates while preserving order
        unique_files=[]
        for f in files:
            if f not in unique_files:
                unique_files.append(f)
        return unique_files

    def _publish_world_interface_files(self):

        if not any(self._use_remote_stepping):
            return
        self._world_iface_files_server=StringTensorServer(length=1,
            basename="SharedWorldInterfaceFilesDropDir",
            name_space=self._robot_names[0],
            verbose=self._verbose,
            vlevel=self._vlevel,
            force_reconnection=True)
        self._world_iface_files_server.run()
        combined_paths=", ".join(self._collect_world_interface_files())
        while not self._world_iface_files_server.write_vec([combined_paths], 0):
            Journal.log(self.__class__.__name__,
            "_publish_world_interface_files",
            f"Failed to pub world interface files. Retrying...",
            LogType.WARN)
            time.sleep(0.1)
        Journal.log(self.__class__.__name__,
            "_publish_world_interface_files",
            f"World interface files advertised: {combined_paths}",
            LogType.STAT)

    def _setup(self) -> bool:
    
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]

            # normalized gravity vector
            self._gravity_normalized[robot_name]=torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._gravity_normalized[robot_name][:, 2]=-1.0
            self._gravity_normalized_base_loc[robot_name]=self._gravity_normalized[robot_name].detach().clone()

            # Pre-allocate yaw-related buffers once and reuse them in root_q_yaw_rel().
            q_ref = self._root_q[robot_name]
            self._root_q_offsets_yaw[robot_name] = torch.zeros(
                (self._num_envs,), dtype=q_ref.dtype, device=q_ref.device)
            self._root_q_yaw_rel_ws[robot_name] = {
                "yaw_abs": torch.zeros((self._num_envs,), dtype=q_ref.dtype, device=q_ref.device),
                "yaw_rel": torch.zeros((self._num_envs,), dtype=q_ref.dtype, device=q_ref.device),
                "yaw_sin": torch.zeros((self._num_envs,), dtype=q_ref.dtype, device=q_ref.device),
                "yaw_cos": torch.zeros((self._num_envs,), dtype=q_ref.dtype, device=q_ref.device),
                "q_abs_unit": torch.zeros_like(q_ref),
                "q_yaw_abs": torch.zeros_like(q_ref),
                "q_yaw_rel": torch.zeros_like(q_ref),
                "q_yaw_abs_conj": torch.zeros_like(q_ref),
                "q_pr": torch.zeros_like(q_ref),
                "q_rel": torch.zeros_like(q_ref),
            }

            self.cluster_sim_step_counters[robot_name]=0
            self._is_first_trigger[robot_name] = True
            if not isinstance(self._cluster_dt[robot_name], (float)):
                exception = f"cluster_dt[{i}] should be a float!"
                Journal.log(self.__class__.__name__,
                    "_setup",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = False)
                return False
            self._cluster_dt[robot_name] = self._cluster_dt[robot_name]
            self._trigger_sol[robot_name] = True # allow first trigger
            self._wait_sol[robot_name] = False

            # initialize a lrhc cluster server for communicating with rhc controllers
            self.cluster_servers[robot_name] = AugMpcClusterServer(cluster_size=self._num_envs, 
                        cluster_dt=self._cluster_dt[robot_name], 
                        control_dt=self.physics_dt(), 
                        jnt_names=self._robot_jnt_names(robot_name=robot_name), 
                        n_contacts=self._n_contacts(robot_name=robot_name),
                        contact_linknames=self._contact_names[robot_name], 
                        verbose=self._verbose, 
                        vlevel=self._vlevel,
                        debug=self._debug, 
                        robot_name=robot_name,
                        use_gpu=self._use_gpu,
                        force_reconnection=self._force_reconnection,
                        timeout_ms=self._timeout,
                        enable_height_sensor=self._enable_height_shared,
                        height_grid_size=self._height_sensor_pixels,
                        height_grid_resolution=self._height_sensor_resolution)
            self.cluster_servers[robot_name].run()
            self.debug_data["cluster_sol_time"][robot_name] = np.nan
            self.debug_data["cluster_state_update_dt"][robot_name] = np.nan
            self.debug_data["sim_time"][robot_name] = np.nan
            # remote sim stepping
            if self._use_remote_stepping[i]:
                self._remote_steppers[robot_name] = RemoteStepperClnt(namespace=robot_name,
                                                            verbose=self._debug,
                                                            vlevel=self._vlevel)
                self._remote_resetters[robot_name] = RemoteResetClnt(namespace=robot_name,
                                                            verbose=self._debug,
                                                            vlevel=self._vlevel)
                self._remote_reset_requests[robot_name] = RemoteResetRequest(namespace=robot_name,
                                                                    n_env=self._num_envs,
                                                                    is_server=True,
                                                                    verbose=self._debug,
                                                                    vlevel=self._vlevel, 
                                                                    force_reconnection=self._force_reconnection, 
                                                                    safe=False)
                self._remote_steppers[robot_name].run()
                self._remote_resetters[robot_name].run()
                self._remote_reset_requests[robot_name].run()
            else:
                self._remote_steppers[robot_name] = None
                self._remote_reset_requests[robot_name] = None
                self._remote_resetters[robot_name] = None

            self._homers[robot_name] = RobotHomer(srdf_path=self._srdf_dump_paths[robot_name], 
                            jnt_names=self._robot_jnt_names(robot_name=robot_name),
                            filter=True)
            robot_homing=torch.from_numpy(self._homers[robot_name].get_homing().reshape(1,-1))
            if "cuda" in self._device:
                robot_homing=robot_homing.cuda()
            self._homing=robot_homing.repeat(self._num_envs, 1)

            self._jnts_q_default[robot_name] = self._homing
            self._set_jnts_to_homing(robot_name=robot_name)
            self._set_root_to_defconfig(robot_name=robot_name)
            self._reset_sim()

            self._init_safe_cluster_actions(robot_name=robot_name)

            Journal.log(self.__class__.__name__,
                "_setup",
                f"Will use joint impedance config at {self._jnt_imp_config_paths[robot_name]} for {robot_name}",
                LogType.STAT)
            
            self._jnt_imp_controllers[robot_name] = self._generate_jnt_imp_control(robot_name=robot_name)
            self._jnt_imp_controllers[robot_name].set_velocity_controlled_joints(
                name_patterns=self._env_opts["wheel_joint_patterns"],
                filter_pos_ref=self._env_opts["filter_wheel_pos_ref"],
                zero_eff_ref=self._env_opts["zero_wheel_eff_ref"])
            self._jnt_imp_cntrl_shared_data[robot_name] = JntImpCntrlData(is_server=True, 
                                            n_envs=self._num_envs, 
                                            n_jnts=len(self._robot_jnt_names(robot_name=robot_name)),
                                            jnt_names=self._robot_jnt_names(robot_name=robot_name),
                                            namespace=robot_name, 
                                            verbose=self._verbose, 
                                            force_reconnection=self._force_reconnection,
                                            vlevel=self._vlevel,
                                            use_gpu=self._use_gpu,
                                            safe=False)
            self._jnt_imp_cntrl_shared_data[robot_name].run()

            self._jnt_vel_filter[robot_name]=None
            if self._env_opts["filter_jnt_vel"]:
                self._jnt_vel_filter[robot_name]=FirstOrderFilter(dt=1.0/self._env_opts["filter_sampling_rate"],
                    filter_BW=self._env_opts["filter_cutoff_freq"],
                    rows=self._num_envs,
                    cols=len(self._robot_jnt_names(robot_name=robot_name)),
                    device=self._device,
                    dtype=self._dtype)
                
                physics_rate=1.0/self.physics_dt()
                self._filter_step_ssteps_freq=int(physics_rate/self._env_opts["filter_sampling_rate"])
                if self._filter_step_ssteps_freq <=0:
                    Journal.log(self.__class__.__name__,
                        "_setup",
                        f"The filter_sampling_rate should be smaller that the physics rate ({physics_rate} Hz)",
                        LogType.EXCEP,
                        throw_when_excep=True)

            for n in range(self._n_init_steps): # run some initialization steps
                if hasattr(self, "_alter_twist_warmup"):
                    self._alter_twist_warmup(robot_name=robot_name, env_indxs=None)
                self._step_world()
               
            self._read_jnts_state_from_robot(robot_name=robot_name,
                env_indxs=None)
            self._read_root_state_from_robot(robot_name=robot_name,
                    env_indxs=None)
            # allow child to perform additional warmup validations (e.g., terrain/tilt)
            # retry_done = False
            if hasattr(self, "_post_warmup_validation"):
                failing = self._post_warmup_validation(robot_name=robot_name)
                if failing is not None and failing.numel() > 0:
                    # retry: reset only failing envs, rerun warmup, revalidate once
                    failing = failing.to(self._device)
                    Journal.log(self.__class__.__name__,
                        "_setup",
                        f"Warmup validation failed for {robot_name}, envs indexes {failing.tolist()}",
                        LogType.EXCEP,
                        throw_when_excep=True)
                else:
                    Journal.log(self.__class__.__name__,
                        "_setup",
                        f"Warmup validation passed for {robot_name}",
                        LogType.INFO)
            
            # write some inits for all robots
            # self._update_root_offsets(robot_name)
            self._synch_default_root_states(robot_name=robot_name)
            epsi=0.03 # adding a bit of height to avoid initial penetration
            self._root_p_default[robot_name][:, 2]=self._root_p_default[robot_name][:, 2]+epsi
            
            reset_ok=self._reset(env_indxs=None,
                robot_name=robot_name,
                reset_cluster=True,
                reset_cluster_counter=False,
                randomize=True,
                acquire_offsets=True) # resets everything, updates the cluster with fresh reset states 
            # and acquire offsets
            if not reset_ok:
                return False
            
            # cluster setup here
            control_cluster=self.cluster_servers[robot_name]

            control_cluster.pre_trigger()
            to_be_activated=control_cluster.get_inactive_controllers()
            if to_be_activated is not None:
                control_cluster.activate_controllers(
                    idxs=to_be_activated)   

            if self._env_opts["run_cluster_bootstrap"]:
                cluster_setup_ok=self._setup_mpc_cluster(robot_name)
                if not cluster_setup_ok:
                    return False
                self._set_cluster_actions(robot_name=robot_name) # write last cmds
                self._apply_cmds_to_jnt_imp_control(robot_name=robot_name) # apply to robot
            
            if self._use_remote_stepping[i]:
                step_wait_ok = self._wait_for_remote_step_req(robot_name=robot_name)
                if not step_wait_ok:
                   return False
            
            self._set_startup_jnt_imp_gains(robot_name=robot_name) # set gains to
            # startup config (usually lower)

            control_cluster.pre_trigger()
            control_cluster.trigger_solution(bootstrap=False) # trigger first solution (in real-time iteration) before first call to step to ensure that first solution is ready when step is called the first time
            
        if self._env_opts["add_remote_exit_flag"]:
            self._remote_exit_flag=SharedTWrapper(namespace = self._robot_names[0],# use first robot as name
                basename = "IbridoRemoteEnvExitFlag",
                is_server = True, 
                n_rows = 1, 
                n_cols = 1, 
                verbose = True, 
                vlevel = self._vlevel,
                safe = False,
                dtype=dtype.Bool,
                force_reconnection=True,
                fill_value = False)
            self._remote_exit_flag.run()

        self._setup_done=True

        return self._setup_done

    def _setup_mpc_cluster(self, robot_name: str):

        control_cluster = self.cluster_servers[robot_name]
        
        # self._set_state_to_cluster(robot_name=robot_name)
        rhc_state = control_cluster.get_state()
        root_twist=rhc_state.root_state.get(data_type="twist", robot_idxs = None, gpu=self._use_gpu)
        jnt_v=rhc_state.jnts_state.get(data_type="v", robot_idxs = None, gpu=self._use_gpu) 
        root_twist[:, :]=0 # override meas state to make sure MPC bootstrap uses zero velocity
        jnt_v[:, :]=0 

        control_cluster.write_robot_state()
        
        # trigger bootstrap solution (solvers will run up to convergence) 
        control_cluster.trigger_solution(bootstrap=True) # this will trigger the bootstrap solver with the initial state,
        # which will run until convergence before returning
        wait_ok=control_cluster.wait_for_solution() # blocking
        if not wait_ok:
            return False
        failed = control_cluster.get_failed_controllers(gpu=self._use_gpu)
        if failed is not None:
            failed_idxs = torch.nonzero(failed).squeeze(-1)
            if failed_idxs.numel() > 0:
                Journal.log(self.__class__.__name__,
                    "_setup",
                    f"Bootstrap solution failed for {robot_name} | n_failed: {failed_idxs.numel()}, idxs: {failed_idxs.cpu().tolist()}",
                    LogType.EXCEP,
                    throw_when_excep=False)
                return False

        return True

    def step(self) -> bool:

        success=False

        if self._remote_exit_flag is not None:
            # check for exit request
            self._remote_exit_flag.synch_all(read=True, retry = False)
            self._exit_request=self._exit_request or \
                bool(self._remote_exit_flag.get_numpy_mirror()[0, 0].item())

        if self._exit_request:
            self.close()
            
        if self.is_running() and (not self.is_closed()):
            if self._debug:
                pre_step_ok=self._pre_step_db()
                if not pre_step_ok:
                    return False
                self._env_timer=time.perf_counter()
                self._step_world()
                self.debug_data["time_to_step_world"] = \
                    time.perf_counter() - self._env_timer
                self._post_world_step_db()
                success=True
            else:
                pre_step_ok=self._pre_step()
                if not pre_step_ok:
                    return False
                self._step_world()
                self._post_world_step()
                success=True
        
        return success
    
    def render(self, mode:str="human") -> None:
        self._render_sim(mode)

    def reset(self,
        env_indxs: torch.Tensor = None,
        reset_cluster: bool = False,
        reset_cluster_counter = False,
        randomize: bool = False,
        reset_sim: bool = False) -> None:

        for i in range(len(self._robot_names)):
            robot_name=self._robot_names[i]
            reset_ok=self._reset(robot_name=robot_name,
                env_indxs=env_indxs,
                randomize=randomize,
                reset_cluster=reset_cluster,
                reset_cluster_counter=reset_cluster_counter)
            if not reset_ok:
                return False    
            self._set_startup_jnt_imp_gains(robot_name=robot_name,
                env_indxs=env_indxs)
            
        if reset_sim:
            self._reset_sim()
        
        return True

    def _reset_cluster(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            reset_cluster_counter: bool = False):
        
        control_cluster = self.cluster_servers[robot_name]

        reset_ok=control_cluster.reset_controllers(idxs=env_indxs)
        if not reset_ok:
            return False
        
        self._set_state_to_cluster(robot_name=robot_name,
            env_indxs=env_indxs)
        control_cluster.write_robot_state() # writes to shared memory

        if reset_cluster_counter:
            self.cluster_sim_step_counters[robot_name] = 0 
        
        return True

    def _step_jnt_vel_filter(self,
            robot_name: str, 
            env_indxs: torch.Tensor = None):
        
        self._jnt_vel_filter[robot_name].update(refk=self.jnts_v(robot_name=robot_name, env_idxs=env_indxs), 
            idxs=env_indxs)

    def _set_state_to_cluster(self, 
        robot_name: str, 
        env_indxs: torch.Tensor = None,
        base_loc: bool = True):
        
        if self._debug:
            if not isinstance(env_indxs, (torch.Tensor, type(None))):
                msg = "Provided env_indxs should be a torch tensor of indexes!"
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]: " + msg)
            
        control_cluster = self.cluster_servers[robot_name]
        # floating base
        rhc_state = control_cluster.get_state()
        # configuration
        rhc_state.root_state.set(data=self.root_p_rel(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="p", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_q(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="q", robot_idxs = env_indxs, gpu=self._use_gpu)
        # rhc_state.root_state.set(data=self.root_q_yaw_rel(robot_name=robot_name, env_idxs=env_indxs), 
        #         data_type="q", robot_idxs = env_indxs, gpu=self._use_gpu)
        # twist
        rhc_state.root_state.set(data=self.root_v(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="v", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_omega(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="omega", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        # angular accc.
        rhc_state.root_state.set(data=self.root_a(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="a", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_alpha(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="alpha", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        # gravity vec
        rhc_state.root_state.set(data=self.gravity(robot_name=robot_name, env_idxs=env_indxs,base_loc=base_loc), 
                data_type="gn", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        # joints
        rhc_state.jnts_state.set(data=self.jnts_q(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="q", robot_idxs = env_indxs, gpu=self._use_gpu)
        
        v_jnts=self.jnts_v(robot_name=robot_name, env_idxs=env_indxs)
        if self._jnt_vel_filter[robot_name] is not None: # apply filtering
            v_jnts=self._jnt_vel_filter[robot_name].get(idxs=env_indxs)
        rhc_state.jnts_state.set(data=v_jnts, 
            data_type="v", robot_idxs = env_indxs, gpu=self._use_gpu) 
        rhc_state.jnts_state.set(data=self.jnts_eff(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="eff", robot_idxs = env_indxs, gpu=self._use_gpu) 

        # height map
        if self._enable_height_shared:
            hdata = self._height_imgs[robot_name]
            if env_indxs is not None:
                hdata = hdata[env_indxs]
            flat = hdata.reshape(hdata.shape[0], -1)
            rhc_state.height_sensor.set(data=flat, data_type=None, robot_idxs=env_indxs, gpu=self._use_gpu)

        # Updating contact state for selected contact links
        self._update_contact_state(robot_name=robot_name, env_indxs=env_indxs)
    
    def _update_contact_state(self, 
            robot_name: str, 
            env_indxs: torch.Tensor = None):
        for i in range(0, self.cluster_servers[robot_name].n_contact_sensors()):
            contact_link = self.cluster_servers[robot_name].contact_linknames()[i]
            f_contact = self._get_contact_f(robot_name=robot_name,
                contact_link=contact_link,
                env_indxs=env_indxs)
            if f_contact is not None:
                self.cluster_servers[robot_name].get_state().contact_wrenches.set(data=f_contact, data_type="f",
                                contact_name=contact_link, 
                                robot_idxs = env_indxs, 
                                gpu=self._use_gpu)
                    
    def _init_safe_cluster_actions(self,
                            robot_name: str):

        # this does not actually write on shared memory, 
        # but it's enough to get safe actions for the simulator before the 
        # cluster starts to receive data from the controllers
        control_cluster = self.cluster_servers[robot_name]
        rhc_cmds = control_cluster.get_actions()
        n_jnts = rhc_cmds.n_jnts()
        
        null_action = torch.zeros((self._num_envs, n_jnts), 
                        dtype=self._dtype,
                        device=self._device)
        rhc_cmds.jnts_state.set(data=self._homing, data_type="q", gpu=self._use_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="v", gpu=self._use_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="eff", gpu=self._use_gpu)

    def _pre_step_db(self) -> None:
        
        # cluster step logic here
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]

            if self._override_low_lev_controller:
                # if overriding low-lev jnt imp. this has to run at the highest
                # freq possible
                start=time.perf_counter()
                self._read_jnts_state_from_robot(robot_name=robot_name)
                self.debug_data["time_to_get_states_from_env"]= time.perf_counter()-start
                
                self._write_state_to_jnt_imp(robot_name=robot_name)
                self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)

            if self._jnt_vel_filter[robot_name] is not None and \
                (self.cluster_sim_step_counters[robot_name]+1) % self._filter_step_ssteps_freq == 0:
                # filter joint vel at a fixed frequency wrt sim steps
                if not self._override_low_lev_controller:
                    # we need a fresh sensor reading
                    self._read_jnts_state_from_robot(robot_name=robot_name)
                self._step_jnt_vel_filter(robot_name=robot_name, env_indxs=None)

            control_cluster = self.cluster_servers[robot_name]
            if control_cluster.is_cluster_instant(self.cluster_sim_step_counters[robot_name]):
                wait_ok=control_cluster.wait_for_solution() # this is blocking
                if not wait_ok:
                    return False
                failed = control_cluster.get_failed_controllers(gpu=self._use_gpu)
                self._set_cluster_actions(robot_name=robot_name) # write last cmds to low level control
                if not self._override_low_lev_controller:
                    self._apply_cmds_to_jnt_imp_control(robot_name=robot_name) # apply to robot
                    # we can update the jnt state just at the rate at which the cluster needs it
                    start=time.perf_counter()
                    self._read_jnts_state_from_robot(robot_name=robot_name, env_indxs=None)
                else:
                    # read state necessary for cluster
                    start=time.perf_counter()
                self._read_root_state_from_robot(robot_name=robot_name, 
                    env_indxs=None)
                self.debug_data["time_to_get_states_from_env"]= time.perf_counter()-start
                start=time.perf_counter()
                self._set_state_to_cluster(robot_name=robot_name, 
                    env_indxs=None)
                control_cluster.write_robot_state()
                self.debug_data["cluster_state_update_dt"][robot_name] = time.perf_counter()-start

                self._update_jnt_imp_cntrl_shared_data() # only if debug_mode_jnt_imp is enabled

                if self._use_remote_stepping[i]:
                    self._remote_steppers[robot_name].ack() # signal cluster stepping is finished
                    if failed is not None and self._env_opts["deact_when_failure"]: # deactivate robot completely 
                        self._deactivate(env_indxs=failed,
                            robot_name=robot_name)
                    wait_reset_ok=self._process_remote_reset_req(robot_name=robot_name) # wait for remote reset request (blocking)
                    wait_step_ok=self._wait_for_remote_step_req(robot_name=robot_name)
                    if not wait_reset_ok or not wait_step_ok:   
                        return False
                else:
                    if failed is not None:
                        reset_ok=self._reset(env_indxs=failed,
                            robot_name=robot_name,
                            reset_cluster=True,
                            reset_cluster_counter=False,
                            randomize=True)
                        if not reset_ok:
                            return False
                        self._set_startup_jnt_imp_gains(robot_name=robot_name,
                            env_indxs=failed)

                    control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

                control_cluster.pre_trigger() # performs pre-trigger steps, like retrieving
                # values of some rhc flags on shared memory
                control_cluster.trigger_solution() # trigger only active controllers

        return True

    def _pre_step(self) -> None:
        
        # cluster step logic here
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            if self._override_low_lev_controller:
                # if overriding low-lev jnt imp. this has to run at the highest
                # freq possible
                self._read_jnts_state_from_robot(robot_name=robot_name)
                self._write_state_to_jnt_imp(robot_name=robot_name)
                self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)

            if self._jnt_vel_filter[robot_name] is not None and \
                (self.cluster_sim_step_counters[robot_name]+1) % self._filter_step_ssteps_freq == 0:
                # filter joint vel at a fixed frequency wrt sim steps
                if not self._override_low_lev_controller:
                    # we need a fresh sensor reading
                    self._read_jnts_state_from_robot(robot_name=robot_name)
                self._step_jnt_vel_filter(robot_name=robot_name, env_indxs=None)

            control_cluster = self.cluster_servers[robot_name]
            if control_cluster.is_cluster_instant(self.cluster_sim_step_counters[robot_name]):
                wait_ok=control_cluster.wait_for_solution() # this is blocking
                if not wait_ok:
                    return False
                failed = control_cluster.get_failed_controllers(gpu=self._use_gpu)
                self._set_cluster_actions(robot_name=robot_name) # set last cmds to low level control
                if not self._override_low_lev_controller:
                    self._apply_cmds_to_jnt_imp_control(robot_name=robot_name) # apply to robot
                    # we can update the jnt state just at the rate at which the cluster needs it
                    self._read_jnts_state_from_robot(robot_name=robot_name, env_indxs=None)
                # read state necessary for cluster
                self._read_root_state_from_robot(robot_name=robot_name, 
                    env_indxs=None)
                # write last robot state to the cluster of controllers
                self._set_state_to_cluster(robot_name=robot_name, 
                    env_indxs=None)
                control_cluster.write_robot_state() # write on shared mem

                if self._use_remote_stepping[i]:
                    self._remote_steppers[robot_name].ack() # signal cluster stepping is finished
                    if failed is not None and self._env_opts["deact_when_failure"]:
                        self._deactivate(env_indxs=failed,
                            robot_name=robot_name)
                    wait_reset_ok=self._process_remote_reset_req(robot_name=robot_name) # wait for remote reset request (blocking)
                    wait_step_ok=self._wait_for_remote_step_req(robot_name=robot_name)
                    if not wait_reset_ok or not wait_step_ok:   
                        return False
                else:
                    if failed is not None:
                        reset_ok=self._reset(env_indxs=failed,
                            robot_name=robot_name,
                            reset_cluster=True,
                            reset_cluster_counter=False,
                            randomize=True)
                        if not reset_ok:
                            return False
                        self._set_startup_jnt_imp_gains(robot_name=robot_name,
                            env_indxs=failed)
                    control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

                control_cluster.pre_trigger() # performs pre-trigger steps, like retrieving
                # values of some rhc flags on shared memory
                control_cluster.trigger_solution() # trigger only active controllers
        
        return True
    
    def _post_world_step_db(self) -> bool:

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            control_cluster = self.cluster_servers[robot_name]
            self.cluster_sim_step_counters[robot_name]+=1 # this has to be update with sim freq
            if self._debug:
                self.debug_data["sim_time"][robot_name]=self.world_time(robot_name=robot_name)
                self.debug_data["cluster_sol_time"][robot_name] = \
                    control_cluster.solution_time()
                
        self.step_counter +=1
    
    def _post_world_step(self) -> bool:

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            self.cluster_sim_step_counters[robot_name]+=1
        self.step_counter +=1
                    
    def _reset(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False,
            reset_cluster: bool = False,
            reset_cluster_counter = False,
            acquire_offsets: bool = False):
        
        # resets the state of target robot and env to the defaults
        self._reset_state(env_indxs=env_indxs, 
            robot_name=robot_name,
            randomize=randomize)
        
        # and jnt imp. controllers
        self._reset_jnt_imp_control(robot_name=robot_name,
                env_indxs=env_indxs)
        
        # read reset state
        self._read_root_state_from_robot(robot_name=robot_name,
                env_indxs=env_indxs)
        self._read_jnts_state_from_robot(robot_name=robot_name,
            env_indxs=env_indxs)

        if self._jnt_vel_filter[robot_name] is not None:
            self._jnt_vel_filter[robot_name].reset(idxs=env_indxs)

        if acquire_offsets:
            self._update_root_offsets(robot_name=robot_name,
                    env_indxs=env_indxs)
            
        if reset_cluster: # reset controllers remotely
            reset_ok=self._reset_cluster(env_indxs=env_indxs,
                robot_name=robot_name,
                reset_cluster_counter=reset_cluster_counter)
            if not reset_ok:
                return False
            
        return True
        
    def _randomize_yaw(self,
            robot_name: str,
            env_indxs: torch.Tensor = None):

        root_q_default = self._root_q_default[robot_name]
        if env_indxs is None:
            env_indxs = torch.arange(root_q_default.shape[0])

        num_indices = env_indxs.shape[0]
        yaw_angles = torch.rand((num_indices,), 
                        device=root_q_default.device) * 2 * torch.pi  # uniformly distributed random angles
        
        # Compute cos and sin once
        cos_half = torch.cos(yaw_angles / 2)
        root_q_default[env_indxs, :] = torch.stack((cos_half, 
                                torch.zeros_like(cos_half),
                                torch.zeros_like(cos_half), 
                                torch.sin(yaw_angles / 2)), dim=1).reshape(num_indices, 4)

    def _deactivate(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):
        
        # deactivate jnt imp controllers for given robots and envs (makes the robot fall)
        self._jnt_imp_controllers[robot_name].deactivate(robot_indxs=env_indxs)
    
    def _n_contacts(self, robot_name: str) -> List[int]:
        return self._num_contacts[robot_name]
    
    def root_p(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_p[robot_name]
        else:
            return self._root_p[robot_name][env_idxs, :]

    def root_p_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            rel_pos = torch.sub(self.root_p(robot_name=robot_name),
                self._root_pos_offsets[robot_name])
        else:
            rel_pos = torch.sub(self.root_p(robot_name=robot_name,
                                                env_idxs=env_idxs), 
                    self._root_pos_offsets[robot_name][env_idxs, :])
        return rel_pos
    
    def root_q(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_q[robot_name]
        else:
            return self._root_q[robot_name][env_idxs, :]

    def root_q_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return quaternion_difference(self._root_q_offsets[robot_name], 
                                self.root_q(robot_name=robot_name))
        rel_q = quaternion_difference(self._root_q_offsets[robot_name][env_idxs, :], 
                            self.root_q(robot_name=robot_name,
                                            env_idxs=env_idxs))
        return rel_q

    def _quat_to_yaw_wxyz(self, q: torch.Tensor, out: torch.Tensor = None):
        # Quaternion convention is w, x, y, z.
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        num = 2.0 * (w * z + x * y)
        den = 1.0 - 2.0 * (y * y + z * z)
        if out is None:
            return torch.atan2(num, den)
        return torch.atan2(num, den, out=out)

    def _yaw_to_quat_wxyz(self, yaw: torch.Tensor, like_q: torch.Tensor,
            out: torch.Tensor = None):
        q = out
        if q is None:
            q = torch.zeros((yaw.shape[0], 4), dtype=like_q.dtype, device=like_q.device)
        else:
            q.zero_()
        q[:, 0] = torch.cos(yaw / 2.0)
        q[:, 3] = torch.sin(yaw / 2.0)
        return q

    def _quat_conjugate_wxyz(self, q: torch.Tensor, out: torch.Tensor = None):
        qi = out
        if qi is None:
            qi = torch.empty_like(q)
        qi[:, :] = q
        qi[:, 1:] = -qi[:, 1:]
        return qi

    def _quat_multiply_wxyz(self, q1: torch.Tensor, q2: torch.Tensor,
            out: torch.Tensor = None):
        q_out = out
        if q_out is None:
            q_out = torch.empty_like(q1)
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        q_out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        q_out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        q_out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        q_out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return q_out

    def _normalize_quat_wxyz(self, q: torch.Tensor, out: torch.Tensor = None):
        q_norm = out
        if q_norm is None:
            q_norm = torch.empty_like(q)
        q_norm[:, :] = q
        q_norm /= torch.clamp(torch.norm(q_norm, dim=1, keepdim=True), min=1e-9)
        return q_norm

    def root_q_yaw_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):
        
        # Return quaternion with startup yaw removed while preserving current pitch/roll.
        if env_idxs is None:
            ws = self._root_q_yaw_rel_ws[robot_name]
            q_abs = self._root_q[robot_name]
            yaw_start = self._root_q_offsets_yaw[robot_name]

            self._normalize_quat_wxyz(q=q_abs, out=ws["q_abs_unit"])
            self._quat_to_yaw_wxyz(q=ws["q_abs_unit"], out=ws["yaw_abs"])

            torch.sub(ws["yaw_abs"], yaw_start, out=ws["yaw_rel"])
            torch.sin(ws["yaw_rel"], out=ws["yaw_sin"])
            torch.cos(ws["yaw_rel"], out=ws["yaw_cos"])
            torch.atan2(ws["yaw_sin"], ws["yaw_cos"], out=ws["yaw_rel"])

            # Build pure-yaw quaternions for:
            # 1) the current absolute heading and 2) the startup-relative heading.
            self._yaw_to_quat_wxyz(yaw=ws["yaw_abs"], like_q=ws["q_abs_unit"], out=ws["q_yaw_abs"])
            self._yaw_to_quat_wxyz(yaw=ws["yaw_rel"], like_q=ws["q_abs_unit"], out=ws["q_yaw_rel"])

            # Isolate pitch/roll by removing the absolute yaw from the current orientation.
            # For unit quaternions q_pr = q_yaw_abs^{-1} * q_abs.
            self._quat_conjugate_wxyz(q=ws["q_yaw_abs"], out=ws["q_yaw_abs_conj"])
            self._quat_multiply_wxyz(q1=ws["q_yaw_abs_conj"], q2=ws["q_abs_unit"], out=ws["q_pr"])
            # Recompose orientation with relative yaw + current pitch/roll.
            self._quat_multiply_wxyz(q1=ws["q_yaw_rel"], q2=ws["q_pr"], out=ws["q_rel"])

            return self._normalize_quat_wxyz(q=ws["q_rel"], out=ws["q_rel"])

        q_abs = self.root_q(robot_name=robot_name, env_idxs=env_idxs)
        q_abs = self._normalize_quat_wxyz(q=q_abs, out=q_abs)

        yaw_abs = self._quat_to_yaw_wxyz(q_abs)
        yaw_start = self._root_q_offsets_yaw[robot_name][env_idxs]
        yaw_rel = yaw_abs - yaw_start
        yaw_rel = torch.atan2(torch.sin(yaw_rel), torch.cos(yaw_rel))

        q_yaw_abs = self._yaw_to_quat_wxyz(yaw_abs, like_q=q_abs)
        q_yaw_rel = self._yaw_to_quat_wxyz(yaw_rel, like_q=q_abs)
        q_pr = self._quat_multiply_wxyz(self._quat_conjugate_wxyz(q_yaw_abs), q_abs)
        q_rel = self._quat_multiply_wxyz(q_yaw_rel, q_pr)

        return self._normalize_quat_wxyz(q_rel)
    
    def root_v(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_v=self._root_v[robot_name]
        if base_loc:
            root_v=self._root_v_base_loc[robot_name]
        if env_idxs is None:
            return root_v
        else:
            return root_v[env_idxs, :]
    
    def root_omega(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_omega=self._root_omega[robot_name]
        if base_loc:
            root_omega=self._root_omega_base_loc[robot_name]
        if env_idxs is None:
            return root_omega
        else:
            return root_omega[env_idxs, :]
    
    def root_a(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_a=self._root_a[robot_name]
        if base_loc:
            root_a=self._root_a_base_loc[robot_name]
        if env_idxs is None:
            return root_a
        else:
            return root_a[env_idxs, :]
    
    def root_alpha(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        root_alpha=self._root_alpha[robot_name]
        if base_loc:
            root_alpha=self._root_alpha_base_loc[robot_name]
        if env_idxs is None:
            return root_alpha
        else:
            return root_alpha[env_idxs, :]
        
    def gravity(self,
            robot_name: str,
            env_idxs: torch.Tensor = None,
            base_loc: bool = True):

        gravity_loc=self._gravity_normalized[robot_name]
        if base_loc:
            gravity_loc=self._gravity_normalized_base_loc[robot_name]
        if env_idxs is None:
            return gravity_loc
        else:
            return gravity_loc[env_idxs, :]
    
    def jnts_q(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):
        
        if env_idxs is None:
            return self._jnts_q[robot_name]
        else:
            return self._jnts_q[robot_name][env_idxs, :]

    def jnts_v(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._jnts_v[robot_name]
        else:
            return self._jnts_v[robot_name][env_idxs, :]

    def jnts_eff(self,
            robot_name: str,
            env_idxs: torch.Tensor = None): # (measured) efforts

        if env_idxs is None:
            return self._jnts_eff[robot_name]
        else:
            return self._jnts_eff[robot_name][env_idxs, :]

    def _wait_for_remote_step_req(self,
            robot_name: str):
        if not self._remote_steppers[robot_name].wait(self._timeout):
            Journal.log(self.__class__.__name__,
                "_wait_for_remote_step_req",
                "Didn't receive any remote step req within timeout!",
                LogType.EXCEP,
                throw_when_excep = False)
            return False
        return True
    
    def _process_remote_reset_req(self,
            robot_name: str):
        
        if not self._remote_resetters[robot_name].wait(self._timeout):
            Journal.log(self.__class__.__name__,
                "_process_remote_reset_req",
                "Didn't receive any remote reset req within timeout!",
                LogType.EXCEP,
                throw_when_excep = False)
            return False
            
        reset_requests = self._remote_reset_requests[robot_name]
        reset_requests.synch_all(read=True, retry=True) # read reset requests from shared mem
        to_be_reset = reset_requests.to_be_reset(gpu=self._use_gpu)
        if to_be_reset is not None:
            reset_ok=self._reset(env_indxs=to_be_reset,
                robot_name=robot_name,
                reset_cluster=True,
                reset_cluster_counter=False,
                randomize=True)
            if not reset_ok:
                return False
            self._set_startup_jnt_imp_gains(robot_name=robot_name,
                env_indxs=to_be_reset) # set gains to startup config (usually lower gains)
        
        control_cluster = self.cluster_servers[robot_name]
        control_cluster.activate_controllers(idxs=to_be_reset) # activate controllers
        # (necessary if failed)

        self._remote_resetters[robot_name].ack() # signal reset performed

        return True
    
    def _update_jnt_imp_cntrl_shared_data(self):
        if self._debug:
            for i in range(0, len(self._robot_names)):
                robot_name = self._robot_names[i]
                # updating all the jnt impedance data - > this may introduce some overhead
                imp_data = self._jnt_imp_cntrl_shared_data[robot_name].imp_data_view
                # set data
                imp_data.set(data_type="pos_err",
                        data=self._jnt_imp_controllers[robot_name].pos_err(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel_err",
                        data=self._jnt_imp_controllers[robot_name].vel_err(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="pos_gains",
                        data=self._jnt_imp_controllers[robot_name].pos_gains(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel_gains",
                        data=self._jnt_imp_controllers[robot_name].vel_gains(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="eff_ff",
                        data=self._jnt_imp_controllers[robot_name].eff_ref(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="pos",
                        data=self._jnt_imp_controllers[robot_name].pos(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="pos_ref",
                        data=self._jnt_imp_controllers[robot_name].pos_ref(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel",
                        data=self._jnt_imp_controllers[robot_name].vel(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="vel_ref",
                        data=self._jnt_imp_controllers[robot_name].vel_ref(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="eff",
                        data=self._jnt_imp_controllers[robot_name].eff(),
                        gpu=self._use_gpu)
                imp_data.set(data_type="imp_eff",
                        data=self._jnt_imp_controllers[robot_name].imp_eff(),
                        gpu=self._use_gpu)
                # copy from GPU to CPU if using gpu
                if self._use_gpu:
                    imp_data.synch_mirror(from_gpu=True,non_blocking=True)
                    # even if it's from GPU->CPu we can use non-blocking since it's just for db 
                    # purposes
                # write copies to shared memory
                imp_data.synch_all(read=False, retry=False)

    def _set_startup_jnt_imp_gains(self,
            robot_name:str, 
            env_indxs: torch.Tensor = None):
        
        startup_p_gains=self._jnt_imp_controllers[robot_name].startup_p_gains()
        startup_d_gains=self._jnt_imp_controllers[robot_name].startup_d_gains()

        if env_indxs is not None:
            self._jnt_imp_controllers[robot_name].set_gains(robot_indxs=env_indxs,
                pos_gains=startup_p_gains[env_indxs, :], 
                vel_gains=startup_d_gains[env_indxs, :])
        else:
            self._jnt_imp_controllers[robot_name].set_gains(robot_indxs=env_indxs,
                pos_gains=startup_p_gains[:, :], 
                vel_gains=startup_d_gains[:, :])
            
    def _write_state_to_jnt_imp(self,
        robot_name: str):
     
        # always update ,imp. controller internal state (jnt imp control is supposed to be
        # always running)
        self._jnt_imp_controllers[robot_name].update_state(pos=self.jnts_q(robot_name=robot_name), 
            vel = self.jnts_v(robot_name=robot_name),
            eff = self.jnts_eff(robot_name=robot_name))

    def _set_cluster_actions(self, 
        robot_name):
        control_cluster = self.cluster_servers[robot_name]
        actions=control_cluster.get_actions()
        active_controllers=control_cluster.get_active_controllers(gpu=self._use_gpu)
        
        if active_controllers is not None:
            self._jnt_imp_controllers[robot_name].set_refs(
                pos_ref=actions.jnts_state.get(data_type="q", gpu=self._use_gpu)[active_controllers, :], 
                vel_ref=actions.jnts_state.get(data_type="v", gpu=self._use_gpu)[active_controllers, :], 
                eff_ref=actions.jnts_state.get(data_type="eff", gpu=self._use_gpu)[active_controllers, :],
                robot_indxs=active_controllers)            
    
    def _jnt_imp_reset_overrride(self, robot_name:str):
        # to be overriden
        pass

    def _apply_cmds_to_jnt_imp_control(self, robot_name:str):

        self._jnt_imp_controllers[robot_name].apply_cmds()

    def _update_root_offsets(self, 
                    robot_name: str,
                    env_indxs: torch.Tensor = None):
        
        if self._debug:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):                
                    msg = "Provided env_indxs should be a torch tensor of indexes!"
                    Journal.log(self.__class__.__name__,
                        "update_root_offsets",
                        msg,
                        LogType.EXCEP,
                        throw_when_excep = True)    
                if self._use_gpu:
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
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
            if self._verbose:
                Journal.log(self.__class__.__name__,
                    "update_root_offsets",
                    f"updating root offsets " + for_robots,
                    LogType.STAT,
                    throw_when_excep = True)

        # only planar position used
        if env_indxs is None:
            self._root_pos_offsets[robot_name][:, 0:2]  = self._root_p[robot_name][:, 0:2]
            self._normalize_quat_wxyz(q=self._root_q[robot_name], out=self._root_q_offsets[robot_name])
            self._quat_to_yaw_wxyz(q=self._root_q_offsets[robot_name],
                out=self._root_q_offsets_yaw[robot_name])
            
        else:
            self._root_pos_offsets[robot_name][env_indxs, 0:2]  = self._root_p[robot_name][env_indxs, 0:2]
            q_root_norm=self._normalize_quat_wxyz(self._root_q[robot_name][env_indxs, :])
            self._root_q_offsets[robot_name][env_indxs, :]  = q_root_norm
            self._root_q_offsets_yaw[robot_name][env_indxs] = self._quat_to_yaw_wxyz(q=q_root_norm)

    def _reset_jnt_imp_control(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None):
        
        if self._debug:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):
                    Journal.log(self.__class__.__name__,
                        "reset_jnt_imp_control",
                        "Provided env_indxs should be a torch tensor of indexes!",
                        LogType.EXCEP,
                        throw_when_excep = True)
                if self._use_gpu:
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
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs)
                                
            if self._verbose:
                Journal.log(self.__class__.__name__,
                    "reset_jnt_imp_control",
                    f"resetting joint impedances " + for_robots,
                    LogType.STAT,
                    throw_when_excep = True)

        # resets all internal data, refs to defaults
        self._jnt_imp_controllers[robot_name].reset(robot_indxs=env_indxs)
        
        #restore jnt imp refs to homing            
        if env_indxs is None:                               
            self._jnt_imp_controllers[robot_name].set_refs(pos_ref=self._homing[:, :],
                robot_indxs = None)
        else:
            self._jnt_imp_controllers[robot_name].set_refs(pos_ref=self._homing[env_indxs, :],
                robot_indxs = env_indxs)

        # self._write_state_to_jnt_imp(robot_name=robot_name)
        # actually applies reset commands to the articulation
        self._write_state_to_jnt_imp(robot_name=robot_name)
        self._jnt_imp_reset_overrride(robot_name=robot_name)
        self._apply_cmds_to_jnt_imp_control(robot_name=robot_name)

    def _synch_default_root_states(self,
            robot_name: str,
            env_indxs: torch.Tensor = None):

        if self._debug:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):
                    msg = "Provided env_indxs should be a torch tensor of indexes!"
                    Journal.log(self.__class__.__name__,
                        "synch_default_root_states",
                        msg,
                        LogType.EXCEP,
                        throw_when_excep = True)  
                if self._use_gpu:
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
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
            if self._verbose:
                Journal.log(self.__class__.__name__,
                            "synch_default_root_states",
                            f"updating default root states " + for_robots,
                            LogType.STAT,
                            throw_when_excep = True)

        if env_indxs is None:
            self._root_p_default[robot_name][:, :] = self._root_p[robot_name]
            self._root_q_default[robot_name][:, :] = self._root_q[robot_name]
        else:
            self._root_p_default[robot_name][env_indxs, :] = self._root_p[robot_name][env_indxs, :]
            self._root_q_default[robot_name][env_indxs, :] = self._root_q[robot_name][env_indxs, :]

    def _generate_rob_descriptions(self, 
                    robot_name: str, 
                    urdf_path: str,
                    srdf_path: str):
        
        custom_xacro_args=extract_custom_xacro_args(self._env_opts)
        Journal.log(self.__class__.__name__,
                    "_generate_rob_descriptions",
                    "generating URDF for robot "+ f"{robot_name}, from URDF {urdf_path}...",
                    LogType.STAT,
                    throw_when_excep = True)
        xrdf_cmds=self._xrdf_cmds(robot_name=robot_name)
        xrdf_cmds=merge_xacro_cmds(prev_cmds=xrdf_cmds,
            new_cmds=custom_xacro_args)
        self._urdf_dump_paths[robot_name]=generate_urdf(robot_name=robot_name, 
            xacro_path=urdf_path,
            dump_path=self._descr_dump_path,
            xrdf_cmds=xrdf_cmds)
        Journal.log(self.__class__.__name__,
                    "_generate_rob_descriptions",
                    "generating SRDF for robot "+ f"{robot_name}, from SRDF {srdf_path}...",
                    LogType.STAT,
                    throw_when_excep = True)
        # we also generate SRDF files, which are useful for control
        self._srdf_dump_paths[robot_name]=generate_srdf(robot_name=robot_name, 
            xacro_path=srdf_path,
            dump_path=self._descr_dump_path,
            xrdf_cmds=xrdf_cmds)
    
    def _xrdf_cmds(self, robot_name:str):
        urdfpath=self._robot_urdf_paths[robot_name]
        # we assume directory tree of the robot package is like
        # robot-ros-pkg/robot_urdf/urdf/robot.urdf.xacro
        parts = urdfpath.split('/')
        urdf_descr_root_path = '/'.join(parts[:-2])
        cmds = get_xrdf_cmds(urdf_descr_root_path=urdf_descr_root_path) 
        return cmds
    
    @abstractmethod
    def current_tstep(self) -> int:
        pass
    
    @abstractmethod
    def world_time(self, robot_name: str) -> float:
        return self.cluster_sim_step_counters[robot_name]*self.physics_dt()
    
    @abstractmethod
    def is_running(self) -> bool:
        pass
    
    @abstractmethod
    def _get_contact_f(self, 
        robot_name: str, 
        contact_link: str,
        env_indxs: torch.Tensor) -> torch.Tensor:
        return None
    
    @abstractmethod
    def physics_dt(self) -> float:
        pass
    
    @abstractmethod
    def rendering_dt(self) -> float:
        pass
    
    @abstractmethod
    def set_physics_dt(self, physics_dt:float):
        pass

    @abstractmethod
    def set_rendering_dt(self, rendering_dt:float):
        pass

    @abstractmethod
    def _robot_jnt_names(self, robot_name: str) -> List[str]:
        pass
    
    @abstractmethod
    def _read_root_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):
        # IMPORTANT: Child interfaces should provide root quaternions in w, x, y, z convention.
        pass
    
    @abstractmethod
    def _read_jnts_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):
        pass

    @abstractmethod
    def _init_robots_state(self):
        pass

    @abstractmethod
    def _reset_state(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False):
        pass
    
    @abstractmethod
    def _init_world(self):
        pass

    @abstractmethod
    def _reset_sim(self) -> None:
        pass
    
    @abstractmethod
    def _set_jnts_to_homing(self, robot_name: str):
        pass

    @abstractmethod
    def _set_root_to_defconfig(self, robot_name: str):
        pass

    @abstractmethod
    def _parse_env_opts(self):
        pass
    
    @abstractmethod
    def _pre_setup(self):
        pass

    @abstractmethod
    def _generate_jnt_imp_control(self) -> JntImpCntrlChild:
        pass

    @abstractmethod
    def _render_sim(self, mode:str="human") -> None:
        pass

    @abstractmethod
    def _close(self) -> None:
        pass

    @abstractmethod
    def _step_world(self) -> None:
        pass
