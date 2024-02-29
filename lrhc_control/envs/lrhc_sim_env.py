from omni_robo_gym.envs.isaac_env import IsaacSimEnv

from lrhc_control.controllers.rhc.lrhc_cluster_server import LRhcClusterServer
from lrhc_control.utils.shared_data.remote_env_stepper import RemoteEnvStepper

from SharsorIPCpp.PySharsorIPC import VLevel, Journal, LogType

from typing import List, Union

import torch 
import numpy as np

import time

from control_cluster_bridge.utilities.cpu_utils.core_utils import get_memory_usage

class LRhcIsaacSimEnv(IsaacSimEnv):

    def __init__(self,
                headless: bool = True, 
                sim_device: int = 0, 
                enable_livestream: bool = False, 
                enable_viewport: bool = False,
                debug = False):

        super().__init__(headless = headless, 
                sim_device = sim_device, 
                enable_livestream = enable_livestream, 
                enable_viewport = enable_viewport,
                debug = debug)

        # debug data
        self.debug_data = {}
        self.debug_data["time_to_step_world"] = np.nan
        self.debug_data["time_to_get_states_from_sim"] = np.nan
        self.debug_data["cluster_sol_time"] = {}
        self.debug_data["cluster_state_update_dt"] = {}

        self._is_training = [] # whether the i-th task is a training task or simply a simulation
        self._training_servers = {} # object in charge of handling connection with training env
        self._n_pre_training_steps = 0
        self._pre_training_step_counter = 0
        self._start_training = False

        self._is_first_trigger = {}
        self.cluster_timers = {}
        self.env_timer = time.perf_counter()

        self.step_counters = {}
        self.cluster_servers = {}
        self._trigger_cluster = {}
        self._cluster_dt = {}
                
    def set_task(self, 
                task, 
                cluster_dt: List[float], 
                is_training: List[bool],
                n_pre_training_steps = 0,
                backend="torch", 
                sim_params=None, 
                init_sim=True, 
                cluster_client_verbose = False, 
                cluster_client_debug = False) -> None:

        super().set_task(task, 
                backend=backend, 
                sim_params=sim_params, 
                init_sim=init_sim)
        
        self.robot_names = self.task.robot_names
        self.robot_pkg_names = self.task.robot_pkg_names
        self._is_training = is_training
        self._n_pre_training_steps = n_pre_training_steps
        
        if not isinstance(cluster_dt, List):
            
            exception = "cluster_dt must be a list!"

            Journal.log(self.__class__.__name__,
                "set_task",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

        if not (len(cluster_dt) == len(self.robot_names)):

            exception = f"cluster_dt length{len(cluster_dt)} does not match robot number {len(self.robot_names)}!"

            Journal.log(self.__class__.__name__,
                "set_task",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

        if not (len(is_training) == len(self.robot_names)):

            exception = f"is_training length{len(is_training)} does not match robot number {len(self.robot_names)}!"

            Journal.log(self.__class__.__name__,
                "set_task",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

        # now the task and the simulation is guaranteed to be initialized
        # -> we have the data to initialize the cluster client
        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            self.step_counters[robot_name] = 0
            self._is_first_trigger[robot_name] = True
                
            if not isinstance(cluster_dt[i], float):

                exception = f"cluster_dt should be a list of float values!"

                Journal.log(self.__class__.__name__,
                    "set_task",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            self._cluster_dt[robot_name] = cluster_dt[i]

            self._trigger_cluster[robot_name] = True # allow first trigger

            if robot_name in task.omni_contact_sensors:

                n_contact_sensors = task.omni_contact_sensors[robot_name].n_sensors
                contact_names = task.omni_contact_sensors[robot_name].contact_prims
            
            else:
                
                n_contact_sensors = -1
                contact_names = None

            self.cluster_servers[robot_name] = LRhcClusterServer(cluster_size=task.num_envs, 
                        cluster_dt=self._cluster_dt[robot_name], 
                        control_dt=task.integration_dt(), 
                        jnt_names = task.robot_dof_names[robot_name], 
                        n_contact_sensors = n_contact_sensors,
                        contact_linknames = contact_names, 
                        verbose = cluster_client_verbose, 
                        debug = cluster_client_debug, 
                        robot_name=robot_name,
                        use_gpu = task.using_gpu)
            
            self.debug_data["cluster_sol_time"][robot_name] = np.nan
            self.debug_data["cluster_state_update_dt"][robot_name] = np.nan

            if self.debug:

                self.cluster_timers[robot_name] = time.perf_counter()
            
            if self._is_training[i]:
                
                self._training_servers[robot_name] = RemoteEnvStepper(
                                            n_envs=self.task.num_envs,
                                            namespace = robot_name,
                                            is_server = True, 
                                            verbose = True,
                                            vlevel = VLevel.V2,
                                            force_reconnection = True,
                                            safe = True)
                        
            else:
                
                self._training_servers[robot_name] = None
        
        self.using_gpu = task.using_gpu

        self._world.add_physics_callback(callback_name="Sparapippo", callback_fn=self.Sparapippo)

    def Sparapippo(self, step_size):
            
        print('weweweewew')
            
    def close(self):

        for i in range(len(self.robot_names)):

            self.cluster_servers[self.robot_names[i]].close()
            
            if self._training_servers[self.robot_names[i]] is not None:
            
                self._training_servers[self.robot_names[i]].sim_env_not_ready() # signal for client

                self._training_servers[self.robot_names[i]].close()
            
        self.task.close() # performs closing steps for task

        super().close() # this has to be called last 
        # so that isaac's simulation is close properly
    
    def step(self, 
        actions = None):
        
        # Stepping order explained:
        # 1) we trigger the solution of the cluster, which will employ the previous state (i.e.
        #    the controller applies an action which is delayed of a cluster_dt wrt to the state it
        #    used -> this is realistic, since between state retrieval and action computation there 
        #    will always be at least a cluster_dt delay
        # 2) we then proceed to set the latest available cluster cmd to 
        #    the low-level impedance controller
        #    (the reference will actually change every cluster_dt/integration_dt steps)-> controllers in the cluster, in the meantime, 
        #    are solving in parallel between them and wrt to simulation stepping. Aside from being
        #    much more realistic, this also avoids serial evaluation of the controllers and the
        #    sim. stepping, which can cause significant rt-factor degradation.
        # 3) we step the simulation
        # 4) as soon as the simulation was stepped, we check the cluster solution status and
        #    wait for the solution if necessary. This means that the bottlneck of the step()
        #    will be the slowest between simulation stepping and cluster solution.
        # 5) we update the cluster state with the one reached after the sim stepping
        
        # print(f"Current RAM usage: {get_memory_usage()}")
        
        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            control_cluster = self.cluster_servers[robot_name]

            # running cluster client if not already running
            if not control_cluster.is_running():
                
                control_cluster.run()

                self._init_safe_cluster_actions(robot_name=robot_name)
                    
                if self._training_servers[robot_name] is not None:

                    self._training_servers[robot_name].run()

            # 1) this runs at a dt = cluster_clients[robot_name] dt (sol. triggering) 
            if control_cluster.is_cluster_instant(self.step_counters[robot_name]):
                
                if self._trigger_cluster[robot_name]:
                    
                    if self._is_first_trigger[robot_name]:
                        
                        # we update the all the default root state now. This state will be used 
                        # for future resets
                        self.task.synch_default_root_states(robot_name = robot_name)

                        self.task.update_jnt_imp_control_gains(robot_name = robot_name, 
                                        jnt_stiffness = self.task.startup_jnt_stiffness, 
                                        jnt_damping = self.task.startup_jnt_damping, 
                                        wheel_stiffness = self.task.startup_wheel_stiffness, 
                                        wheel_damping = self.task.startup_wheel_damping)
                            
                        self._is_first_trigger[robot_name] = False

                    if self._training_servers[robot_name] is not None and \
                        self._start_training:

                            self._training_servers[robot_name].wait() # blocking
                            
                            to_be_reset_remotely = self._training_servers[robot_name].get_stepper().get_resets()
                            
                            if to_be_reset_remotely is not None:
                                
                                print("Ueeeppaaa")
                                print(to_be_reset_remotely)

                                self.reset(env_indxs=to_be_reset_remotely,
                                    robot_names=[robot_name],
                                    reset_world=False,
                                    reset_cluster=True)

                            # when training controllers have to be kept always active
                            control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

                    control_cluster.pre_trigger_steps() # performs pre-trigger steps, like retrieving
                    # values of some activation flags
                        
                    just_activated = control_cluster.get_just_activated() # retrieves just 
                    # activated controllers
           
                    just_deactivated = control_cluster.get_just_deactivated() # retrieves just 
                    # deactivated controllers
                        
                    if just_activated is not None:
                        
                        # transition of some controllers to being triggered after being just activated

                        # we get the current absolute positions for the transitioned controllers and 
                        # use them as references
                        self.task.update_root_offsets(robot_name,
                                        env_indxs = just_activated)

                        # we initialize values of states for the activated controllers
                        self._update_cluster_state(robot_name = robot_name, 
                                        env_indxs = just_activated)

                        # some controllers transitioned to running state -> we set the 
                        # running gain state for the low-level imp. controller
                        # self.task.update_jnt_imp_control_gains(robot_name = robot_name, 
                        #                 jnt_stiffness = self.task.startup_jnt_stiffness, 
                        #                 jnt_damping = self.task.startup_jnt_damping, 
                        #                 wheel_stiffness = self.task.startup_wheel_stiffness, 
                        #                 wheel_damping = self.task.startup_wheel_damping,
                        #                 env_indxs = just_activated)

                    # if just_deactivated is not None:

                    #     # reset jnt imp. controllers for deactivated controllers
                        
                    #     self.task.reset_jnt_imp_control(robot_name=robot_name,
                    #             env_indxs=just_deactivated)

                    # every control_cluster_dt, trigger the solution of the active controllers in the cluster
                    # with the latest available state
                    control_cluster.trigger_solution()

            # 2) this runs at a dt = simulation dt i.e. the highest possible rate,
            #    using the latest available RHC solution (the new one is not available yet)
            # (sets low level cmds to jnt imp controller)
            active = control_cluster.get_active_controllers()

            self.task.pre_physics_step(robot_name = robot_name, 
                            actions = control_cluster.get_actions(),
                            env_indxs = active) # applies updated rhc actions to low-level
            # joint imp. control only for active controllers     

        # 3) simulation stepping (@ integration_dt) (for all robots and all environments)
        self._step_world()

        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            # this runs at a dt = control_cluster dt
            if control_cluster.is_cluster_instant(self.step_counters[robot_name]):
            
                if not self._trigger_cluster[robot_name]:                        
            
                    # we reach the next control instant -> we get the solution
                    # we also reset the flag, so next call to step() will trigger again the
                    # cluster

                    # 3) wait for solution (will also read latest computed cmds)
                    control_cluster.wait_for_solution() # this is blocking
                        
                    self._trigger_cluster[robot_name] = True # this allows for the next trigger 

                    # 4) update cluster state
                    if self.debug:

                        self.cluster_timers[robot_name] = time.perf_counter()

                    # update cluster state 
                    self._update_cluster_state(robot_name = robot_name, 
                                    env_indxs = active)
                    
                    if self._training_servers[robot_name] is None:
                        
                        # automatically reset failed controllers if not running training
                        failed = control_cluster.get_failed_controllers()

                        if failed is not None:
        
                            self.reset(env_indxs=failed,
                                    robot_names=[robot_name],
                                    reset_world=False,
                                    reset_cluster=True)
                    
                        control_cluster.activate_controllers(idxs=control_cluster.get_inactive_controllers())

                    if self._training_servers[robot_name] is not None:

                        if self._start_training:
                            
                            self._training_servers[robot_name].step() # signal stepping is finished
                            
                        if self._pre_training_step_counter < self._n_pre_training_steps and \
                                not self._start_training:
                    
                            self._pre_training_step_counter += 1
                    
                        if self._pre_training_step_counter >= self._n_pre_training_steps and \
                                not self._start_training:
                            
                            self._start_training = True # next cluster step we wait for connection to training client

                            self._training_servers[robot_name].sim_env_ready() # signal training client sim is ready
                        
                    if self.debug:

                        self.debug_data["cluster_state_update_dt"][robot_name] = \
                            time.perf_counter() - self.cluster_timers[robot_name]
                        
                        self.debug_data["cluster_sol_time"][robot_name] = \
                            control_cluster.solution_time
                        
                else: # we are in the same step() call as the cluster trigger

                    self._trigger_cluster[robot_name] = False # -> next cluster instant we get/wait the solution
                    # from the cluster                

            self.step_counters[robot_name] += 1

        if self.debug:
            
            self.env_timer = time.perf_counter()

        self.task.get_states() # gets data from simulation (jnt imp control always needs updated state)

        if self.debug:
                        
            self.debug_data["time_to_get_states_from_sim"] = time.perf_counter() - self.env_timer

    def reset_cluster(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str]=None):
        
        rob_names = robot_names
        
        if rob_names is None:
            
            rob_names = self.robot_names

        for i in range(len(rob_names)):
            
            robot_name = rob_names[i]

            control_cluster = self.cluster_servers[robot_name]

            control_cluster.reset_controllers(idxs=env_indxs)
    
    def full_reset(self,
            robot_name: str = None):

        # resets EVERYTHING
        self.reset(env_indxs=None,# all env
                robot_names=[robot_name], # potenially not all robots
                reset_world=False,
                reset_cluster=True,
                reset_counter=True)

    def reset(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str]=None,
            reset_world: bool = False,
            reset_cluster: bool = False,
            reset_counter = False):

        if reset_cluster:

            # reset clusters
            self.reset_cluster(env_indxs=env_indxs,
                    robot_names=robot_names)
            
        if reset_world:

            self._world.reset()

        self.task.reset(env_indxs = env_indxs,
            robot_names = robot_names)
        
        # perform a simulation step
        
        # self._world.step(render=self._render)

        # self.task.get_states(env_indxs=env_indxs,
        #                 robot_names=robot_names) # updates states from sim 

        self.task.after_reset(env_indxs = env_indxs,
            robot_names = robot_names)
        
        rob_names = robot_names
        
        if rob_names is None:
            
            rob_names = self.robot_names

        if reset_cluster:

            for i in range(len(rob_names)):

                self._update_cluster_state(robot_name=rob_names[i],
                                env_indxs=env_indxs)

        if reset_counter:
            
            for i in range(len(rob_names)):
                
                self.step_counters[rob_names[i]] = 0
    
    def _init_safe_cluster_actions(self,
                            robot_name: str):

        # this does not actually write on shared memory, 
        # but it's enough to get safe actions before the 
        # cluster 
        control_cluster = self.cluster_servers[robot_name]

        rhc_cmds = control_cluster.get_actions()

        n_jnts = rhc_cmds.n_jnts()
        
        homing = self.task.homers[robot_name].get_homing()

        null_action = torch.zeros((self.task.num_envs, n_jnts), 
                        dtype=self.task.torch_dtype)

        rhc_cmds.jnts_state.set_q(q = homing, gpu = self.using_gpu)

        rhc_cmds.jnts_state.set_v(v = null_action, gpu = self.using_gpu)

        rhc_cmds.jnts_state.set_eff(eff = null_action, gpu = self.using_gpu)

        rhc_cmds.synch_to_shared_mem() # write init to shared mem

    def _step_world(self):

        if self.debug:
            
            self.env_timer = time.perf_counter()

        self._world.step(render=self._render)

        if self.debug:
            
            self.debug_data["time_to_step_world"] = time.perf_counter() - self.env_timer

    def _update_contact_state(self, 
                    robot_name: str, 
                    env_indxs: torch.Tensor = None):

        for i in range(0, self.cluster_servers[robot_name].n_contact_sensors):
            
            contact_link = self.cluster_servers[robot_name].contact_linknames[i]
            
            if not self.using_gpu:
                
                f_contact = self.task.omni_contact_sensors[robot_name].get(dt = self.task.integration_dt(), 
                                                            contact_link = contact_link,
                                                            env_indxs = env_indxs,
                                                            clone = False)
                # assigning measured net contact forces
                self.cluster_servers[robot_name].get_state().contact_wrenches.set_f_contact(f=f_contact,
                                            contact_name=contact_link,
                                            robot_idxs = env_indxs,
                                            gpu=self.using_gpu)
                    
            else:
                
                if ((self.step_counters[robot_name] + 1) % 10000) == 0:
                    
                    # sporadic warning
                    warning = f"Contact state from link {contact_link} cannot be retrieved in IsaacSim if using use_gpu_pipeline is set to True!"

                    Journal.log(self.__class__.__name__,
                        "_update_contact_state",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)
            
    def _update_cluster_state(self, 
                    robot_name: str, 
                    env_indxs: torch.Tensor = None):
        
        if not isinstance(env_indxs, Union[torch.Tensor, None]):
            
            msg = "Provided env_indxs should be a torch tensor of indexes!"
        
            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]: " + msg)
        
        if env_indxs is not None:

            if not len(env_indxs.shape) == 1:

                msg = "Provided env_indxs should be a 1D torch tensor!"
            
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]: " + msg)

        control_cluster = self.cluster_servers[robot_name]

        # floating base
        relative_pos = self.task.root_p_rel(robot_name=robot_name,
                                            env_idxs=env_indxs)

        control_cluster.get_state().root_state.set_p(p = relative_pos,
                                                            robot_idxs = env_indxs,
                                                            gpu = self.using_gpu) # we only set the relative position
        # w.r.t. the initial spawning pose
        control_cluster.get_state().root_state.set_q(q = self.task.root_q(robot_name=robot_name,
                                                                                            env_idxs=env_indxs),
                                                                robot_idxs = env_indxs,
                                                                gpu = self.using_gpu)
        # control_cluster.get_state().root_state.set_q(q = self.task.root_q_rel(robot_name=robot_name,
        #                                                                                     env_idxs=env_indxs),
        #                                                         robot_idxs = env_indxs,
        #                                                         gpu = self.using_gpu)

        control_cluster.get_state().root_state.set_v(v=self.task.root_v(robot_name=robot_name,
                                                                                            env_idxs=env_indxs),
                                                                robot_idxs = env_indxs,
                                                                gpu = self.using_gpu) 
        # control_cluster.get_state().root_state.set_v(v=self.task.root_v_rel(robot_name=robot_name,
        #                                                                                     env_idxs=env_indxs),
        #                                                         robot_idxs = env_indxs,
        #                                                         gpu = self.using_gpu) 

        control_cluster.get_state().root_state.set_omega(gpu = self.using_gpu,
                                                                    robot_idxs = env_indxs,
                                                                    omega=self.task.root_omega(robot_name=robot_name,
                                                                                            env_idxs=env_indxs)) 
        # control_cluster.get_state().root_state.set_omega(gpu = self.using_gpu,
        #                                                             robot_idxs = env_indxs,
        #                                                             omega=self.task.root_omega_rel(robot_name=robot_name,
        #                                                                                     env_idxs=env_indxs)) 

        # joints
        control_cluster.get_state().jnts_state.set_q(q=self.task.jnts_q(robot_name=robot_name,
                                                                                            env_idxs=env_indxs), 
                                                                    robot_idxs = env_indxs,
                                                                    gpu = self.using_gpu)

        control_cluster.get_state().jnts_state.set_v(v=self.task.jnts_v(robot_name=robot_name,
                                                                                            env_idxs=env_indxs),
                                                                    robot_idxs = env_indxs,
                                                                    gpu = self.using_gpu) 

        # Updating contact state for selected contact links
        self._update_contact_state(robot_name=robot_name,
                            env_indxs=env_indxs)
