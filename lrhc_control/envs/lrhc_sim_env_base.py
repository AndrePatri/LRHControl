from lrhc_control.controllers.rhc.lrhc_cluster_server import LRhcClusterServer
from lrhc_control.utils.shared_data.remote_stepping import RemoteStepperClnt
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetClnt
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetRequest

from control_cluster_bridge.utilities.homing import RobotHomer

from SharsorIPCpp.PySharsorIPC import VLevel, Journal, LogType

from typing import List, Union, Dict

import os
import signal
import time

import numpy as np
import torch

from abc import ABC, abstractmethod

class LRhcEnvBase():

    def __init__(self,
                robot_names: List[str],
                robot_pkg_names: List[str],
                robot_srdf_paths: List[str],
                cluster_dt: List[float],
                use_remote_stepping: List[bool],
                num_envs: int = 1,
                headless: bool = True, 
                debug = False,
                verbose: bool = False,
                vlevel: VLevel = VLevel.V1,
                n_init_step: int = 0,
                timeout_ms: int = 60000,
                custom_opts: Dict = None,
                use_gpu: bool = True,
                dtype: torch.dtype = torch.float32):

        # checks on input args
        # type checks
        if not isinstance(robot_names, List):
            exception = "robot_names must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        if not isinstance(robot_pkg_names, List):
            exception = "robot_pkg_names must be a list!"
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
        if not isinstance(robot_srdf_paths, List):
            exception = "robot_srdf_paths must be a list!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        # dim checks
        if not len(robot_pkg_names) == len(robot_names):
            exception = f"robot_pkg_names has len {len(robot_pkg_names)}" + \
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
            
        self._num_envs=num_envs
        self._headless=headless
        self._debug=debug
        self._verbose=verbose
        self._vlevel=vlevel
        self._force_reconnection=True
        self._timeout_ms=timeout_ms
        self._use_gpu=use_gpu
        self._device = "cuda" if self._use_gpu else "cpu"
        self._dtype=dtype
        self._robot_names=robot_names
        self._robot_pkg_names=robot_pkg_names
        self._custom_opts=custom_opts
            
        self.step_counter = 0 # global step counter
        self._init_steps_done = False
        self._n_init_steps = n_init_step # n steps to be performed before applying solutions from control clusters

        self._srdf_paths = robot_srdf_paths
        self._homers = {} 

        # control cluster data
        self.cluster_timers = {}
        self.cluster_step_counters = {}
        self.cluster_servers = {}
        self._trigger_sol = {}
        self._wait_sol = {}
        self._cluster_dt = {}

        # db data
        self.debug_data = {}
        self.debug_data["time_to_step_world"] = np.nan
        self.debug_data["time_to_get_states_from_sim"] = np.nan
        self.debug_data["cluster_sol_time"] = {}
        self.debug_data["cluster_state_update_dt"] = {}
        self.debug_data["sim_time"] = {}
        self.debug_data["cluster_time"] = {}
        
        self.env_timer = time.perf_counter()

        # remote sim stepping options
        self._timeout = timeout_ms # timeout for remote stepping
        self._use_remote_stepping = use_remote_stepping # whether the task associated with robot i 
        # should use remote stepping
        self._remote_steppers = {}
        self._remote_resetters = {}
        self._remote_reset_requests = {}
        self._is_first_trigger = {}

        self._closed = False
        # handle ctrl+c event
        signal.signal(signal.SIGINT, self.signal_handler)        

        self._setup()
            
    def signal_handler(self, sig, frame):
        self.close()
    
    def __del__(self):
        self.close()
    
    def close(self) -> None:
        if not self._closed:
            for i in range(len(self._robot_names)):
                self.cluster_servers[self._robot_names[i]].close()
                if self._use_remote_stepping[i]: # remote signaling
                    self._remote_reset_requests[self._robot_names[i]].close()
                    self._remote_resetters[self._robot_names[i]].close()
                    self._remote_steppers[self._robot_names[i]].close()
            self._close()
            self._closed=True
    
    def _setup(self) -> None:

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            self.cluster_step_counters[robot_name]=0
            self._is_first_trigger[robot_name] = True
            if not isinstance(self._cluster_dt[i], (float)):
                exception = f"cluster_dt[{i}] should be a float!"
                Journal.log(self.__class__.__name__,
                    "set_task",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            self._cluster_dt[robot_name] = self._cluster_dt[i]
            self._trigger_sol[robot_name] = True # allow first trigger
            self._wait_sol[robot_name] = False

            # initialize a lrhc cluster server for communicating with rhc controllers
            self.cluster_servers[robot_name] = LRhcClusterServer(cluster_size=self._num_envs, 
                        cluster_dt=self._cluster_dt[robot_name], 
                        control_dt=self.physics_dt(), 
                        jnt_names=self._robot_jnt_names(robot_name=robot_name), 
                        n_contacts=self._n_contacts(robot_name=robot_name),
                        contact_linknames=self._contact_names(robot_name=robot_name), 
                        verbose=self._verbose, 
                        vlevel=self._vlevel,
                        debug=self._debug, 
                        robot_name=robot_name,
                        use_gpu=self._use_gpu,
                        force_reconnection=self._force_reconnection,
                        timeout_ms=self._timeout)
            self.cluster_servers[robot_name].run()
            self.debug_data["cluster_sol_time"][robot_name] = np.nan
            self.debug_data["cluster_state_update_dt"][robot_name] = np.nan
            self.debug_data["sim_time"][robot_name] = np.nan
            if self._debug:
                self.cluster_timers[robot_name] = time.perf_counter()
            # remote sim stepping
            if self._use_remote_stepping[i]:
                self._remote_steppers[robot_name] = RemoteStepperClnt(namespace=robot_name,
                                                            verbose=self._verbose,
                                                            vlevel=self._vlevel)
                self._remote_resetters[robot_name] = RemoteResetClnt(namespace=robot_name,
                                                            verbose=self._verbose,
                                                            vlevel=self._vlevel)
                self._remote_reset_requests[robot_name] = RemoteResetRequest(namespace=robot_name,
                                                                    n_env=self.num_envs,
                                                                    is_server=True,
                                                                    verbose=self._verbose,
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

            self._homers[robot_name] = RobotHomer(srdf_path=self._srdf_paths[robot_name], 
                            jnt_names=self._robot_jnt_names(robot_name=robot_name),
                            filter=True)
            
            self._init_safe_cluster_actions(robot_name=robot_name)

    def step(self, actions=None) -> bool:
        success=True
        success = success and self._pre_sim_step()
        success = success and self._step_sim()
        success = success and self._post_sim_step()
        return success
    
    def render(self, mode:str) -> None:
        self._render(mode)

    def reset(self,
        env_indxs: torch.Tensor = None,
        robot_names: List[str] = None,
        reset_sim: bool = False,
        reset_cluster: bool = False,
        reset_cluster_counter = False,
        randomize: bool = False) -> None:

        if reset_cluster: # reset controllers remotely
            self._reset_cluster(env_indxs=env_indxs,
                    robot_names=robot_names)
        if reset_sim:
            self._reset_sim()
        if robot_names is None:
            robot_names = self.robot_names

        self._reset(env_indxs=env_indxs,
            robot_names=robot_names,
            randomize=randomize)
        
        if reset_cluster: # reset the state of clusters using the reset state
            for i in range(len(robot_names)):
                self._update_cluster_state(robot_name=robot_names[i],
                                env_indxs=env_indxs)
                if reset_cluster_counter:
                    self.cluster_step_counters[robot_names[i]] = 0                

    def _update_cluster_state(self, 
        robot_name: str, 
        env_indxs: torch.Tensor = None):
        
        if self._debug:
            if not isinstance(env_indxs, Union[torch.Tensor, None]):
                msg = "Provided env_indxs should be a torch tensor of indexes!"
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]: " + msg)
            
        control_cluster = self.cluster_servers[robot_name]
        # floating base
        rhc_state = control_cluster.get_state()
        rhc_state.root_state.set(data=self.root_p(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="p", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_q(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="q", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_v(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="v", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.root_state.set(data=self.root_omega(robot_name=robot_name, env_idxs=env_indxs), 
                data_type="omega", robot_idxs = env_indxs, gpu=self._use_gpu)
        # joints
        rhc_state.jnts_state.set(data=self.jnts_q(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="q", robot_idxs = env_indxs, gpu=self._use_gpu)
        rhc_state.jnts_state.set(data=self.jnts_v(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="v", robot_idxs = env_indxs, gpu=self._use_gpu) 
        rhc_state.jnts_state.set(data=self.jnts_eff(robot_name=robot_name, env_idxs=env_indxs), 
            data_type="eff", robot_idxs = env_indxs, gpu=self._use_gpu) 
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
                                contact_name=contact_link, robot_idxs = env_indxs, gpu=self.using_gpu)
                    
    def _init_safe_cluster_actions(self,
                            robot_name: str):

        # this does not actually write on shared memory, 
        # but it's enough to get safe actions for the simulator before the 
        # cluster starts to receive data from the controllers
        control_cluster = self.cluster_servers[robot_name]
        rhc_cmds = control_cluster.get_actions()
        n_jnts = rhc_cmds.n_jnts()
        
        robot_homing=None
        robot_homing=torch.from_numpy(self._homers[robot_name].get_homing().reshape(1,-1))
        if self._use_gpu:
            robot_homing=robot_homing.cuda()
        homing=robot_homing.repeat(self._num_envs, 1)
        null_action = torch.zeros((self.task.num_envs, n_jnts), 
                        dtype=self._dtype,
                        device=self._device)
        rhc_cmds.jnts_state.set(data=homing, data_type="q", gpu=self.using_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="v", gpu=self.using_gpu)
        rhc_cmds.jnts_state.set(data=null_action, data_type="eff", gpu=self.using_gpu)

    @abstractmethod
    def _close(self) -> None:
        pass

    @abstractmethod
    def _step_sim(self) -> bool:
        pass
    
    def _pre_sim_step(self) -> bool:
        self._update_state_from_sim()

        # cluster step logic here
        for i in range(len(self._robot_names)):
            
            robot_name = self._robot_names[i]
            control_cluster = self.cluster_servers[robot_name] # retrieve control
            # cluster states
            active = None 
            just_activated = None
            just_deactivated = None                
            failed = None
            
            # 1) this runs @cluster_dt: waits + retrieves latest solution
            if control_cluster.is_cluster_instant(self.cluster_step_counters[robot_name]) and \
                    self._init_steps_done:
                
                control_cluster.pre_trigger() # performs pre-trigger steps, like retrieving
                # values of some rhc flags on shared memory
                if not self._is_first_trigger[robot_name]: # first trigger we don't wait (there has been no trigger)
                    control_cluster.wait_for_solution() # this is blocking
                    if self._debug:
                        self.cluster_timers[robot_name] = time.perf_counter()
                    failed = control_cluster.get_failed_controllers(gpu=self.using_gpu)
                    if self._use_remote_stepping[i]:
                        self._remote_steppers[robot_name].ack() # signal cluster stepping is finished
                        if failed is not None: # deactivate robot completely (basically, deactivate jnt imp controller)
                            self._deactivate(env_indxs=failed,
                                robot_names=[robot_name])
                        self._process_remote_reset_req(robot_name=robot_name)
                    else:
                        # automatically reset and reactivate failed controllers if not running remotely
                        if failed is not None:
                            self.reset(env_indxs=failed,
                                    robot_names=[robot_name],
                                    reset_sim=False,
                                    reset_cluster=True,
                                    reset_cluster_counter=False,
                                    randomize=False)
                        # activate inactive controllers
                        control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

            active = control_cluster.get_active_controllers(gpu=self.using_gpu)
            # 2) this runs at @simulation dt i.e. the highest possible rate,
            # using the latest available RHC actions
            self._update_jnt_imp_cntrl_actions(robot_name<=robot_name, 
                        actions=control_cluster.get_actions(),
                        env_indxs=active)
            # 3) this runs at @cluster_dt: trigger cluster solution
            if control_cluster.is_cluster_instant(self.cluster_step_counters[robot_name]) and \
                    self._init_steps_done:
                if self._is_first_trigger[robot_name]:
                        # we update the all the default root state now. This state will be used 
                        # for future resets
                        self._synch_default_root_states(robot_name=robot_name)
                        self._is_first_trigger[robot_name] = False
                if self._use_remote_stepping[i]:
                        self._wait_for_remote_step_req(robot_name=robot_name)
                # handling controllers transition to running state
                just_activated = control_cluster.get_just_activated(gpu=self.using_gpu) 
                just_deactivated = control_cluster.get_just_deactivated(gpu=self.using_gpu)
                if just_activated is not None: # transition of some controllers from not active -> active
                    self._update_root_offsets(robot_name,
                                    env_indxs = just_activated) # we use relative state wrt to this state for the controllers
                    self._update_jnt_imp_control_gains(robot_name = robot_name, 
                                    jnt_stiffness = self.task.startup_jnt_stiffness, 
                                    jnt_damping = self.task.startup_jnt_damping, 
                                    wheel_stiffness = self.task.startup_wheel_stiffness, 
                                    wheel_damping = self.task.startup_wheel_damping,
                                    env_indxs = just_activated) # setting runtime state for jnt imp controller
                if just_deactivated is not None:
                    self._reset_jnt_imp_control(robot_name=robot_name,
                            env_indxs=just_deactivated)
                self._update_cluster_state(robot_name=robot_name, 
                                    env_indxs=active) # always update cluster state every cluster dt
                control_cluster.trigger_solution() # trigger only active controllers

    @abstractmethod
    def _post_sim_step(self) -> bool:
        self.step_counter +=1
        if self.step_counter==self._n_init_steps and \
                not self._init_steps_done:
            self._init_steps_done = True
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            self.cluster_step_counters[robot_name] += 1
            if self._debug:
                self.debug_data["sim_time"][robot_name]=self.cluster_step_counters[robot_name]*self.physics_dt()
                self.debug_data["cluster_sol_time"][robot_name] = \
                    self.cluster_servers[robot_name].solution_time()

    @abstractmethod
    def _render(self, mode:str) -> None:
        pass

    @abstractmethod
    def _reset(self,
        env_indxs: torch.Tensor,
        robot_names: List[str],
        randomize: bool) -> None:
        pass
    
    @abstractmethod
    def _reset_sim(self) -> None:
        pass
    
    @abstractmethod
    def _deactivate(self,
        env_indxs = torch.Tensor,
        robot_names = List[str]) -> None:
        pass
    
    @abstractmethod
    def _n_contacts(self, robot_name: str) -> List[int]:
        pass
    
    @abstractmethod
    def _contact_names(self, robot_name: str) -> List[str]:
        pass

    @abstractmethod
    def physics_dt(self) -> int:
        pass
    
    @abstractmethod
    def _robot_jnt_names(self, robot_name: str) -> List[str]:
        pass
    
    @abstractmethod
    def _update_state_from_sim(self):
        pass

    @abstractmethod
    def root_p(self, robot_name: str,
        env_idxs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def root_q(self, robot_name: str,
        env_idxs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def root_v(self, robot_name: str,
        env_idxs: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def root_omega(self, robot_name: str,
        env_idxs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def jnts_q(self, robot_name: str,
        env_idxs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def jnts_v(self, robot_name: str,
        env_idxs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def jnts_eff(self, robot_name: str,
        env_idxs: torch.Tensor) -> torch.Tensor:
        pass

    def _get_contact_f(self, 
        robot_name: str, 
        contact_link: str,
        env_indxs: torch.Tensor) -> torch.Tensor:
        return None

    def _wait_for_remote_step_req(self,
            robot_name: str):
        if not self._remote_steppers[robot_name].wait(self._timeout):
            self.close()
            Journal.log(self.__class__.__name__,
                "_wait_for_remote_step_req",
                "Didn't receive any remote step req within timeout!",
                LogType.EXCEP,
                throw_when_excep = True)
    
    def _process_remote_reset_req(self,
            robot_name: str):
        
        if not self._remote_resetters[robot_name].wait(self._timeout):
            self.close()
            Journal.log(self.__class__.__name__,
                "_process_remote_reset_req",
                "Didn't receive any remote reset req within timeout!",
                LogType.EXCEP,
                throw_when_excep = True)
            
        reset_requests = self._remote_reset_requests[robot_name]
        reset_requests.synch_all(read=True, retry=True) # read reset requests from shared mem
        to_be_reset = reset_requests.to_be_reset(gpu=self.using_gpu)
        if to_be_reset is not None:
            self.reset(env_indxs=to_be_reset,
                robot_names=[robot_name],
                reset_sim=False,
                reset_cluster=True,
                reset_cluster_counter=False,
                randomize=True)
        control_cluster = self.cluster_servers[robot_name]
        control_cluster.activate_controllers(idxs=to_be_reset) # activate controllers
        # (necessary if failed)

        self._remote_resetters[robot_name].ack() # signal reset performed

    @abstractmethod
    def _update_jnt_imp_cntrl_actions(self,
        robot_name: str, 
        actions = None,
        env_indxs: torch.Tensor = None):
        pass
    
    @abstractmethod
    def _reset_jnt_imp_control(self,
        robot_name: str, 
        env_indxs: torch.Tensor = None):
        pass

    @abstractmethod
    def _synch_default_root_states(self,
        robot_name: str):
        pass

    @abstractmethod
    def _update_root_offsets(self,
        robot_name: str, 
        env_indxs: torch.Tensor = None):
        pass
    
    @abstractmethod
    def _update_jnt_imp_control_gains(self,
        robot_name: str, 
        jnt_stiffness: torch.Tensor,
        jnt_damping: torch.Tensor,
        wheel_stiffness: torch.Tensor,
        wheel_damping: torch.Tensor,
        env_indxs: torch.Tensor = None):
        pass
