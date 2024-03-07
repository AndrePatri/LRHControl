from gymnasium import spaces

import torch

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus

from lrhc_control.utils.shared_data.remote_env_stepper import RemoteEnvStepper
from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import SharedTrainingEnvInfo

from lrhc_control.utils.shared_data.training_env import Observations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Rewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations
from lrhc_control.utils.shared_data.training_env import Truncations
from lrhc_control.utils.shared_data.training_env import TimeUnlimitedTasksEpCounter

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from perf_sleep.pyperfsleep import PerfSleep

from abc import abstractmethod

import os

class LRhcTrainingEnvBase():

    """Base class for a remote training environment tailored to Learning-based Receding Horizon Control"""

    def __init__(self,
            namespace: str,
            obs_dim: int,
            actions_dim: int,
            time_limit_nsteps: int,
            env_name: str = "",
            n_preinit_steps: int = 1,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            debug: bool = True,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32):
        
        self._this_path = os.path.abspath(__file__)

        self._env_index = 0

        self._time_limit_nsteps = time_limit_nsteps

        self._namespace = namespace
        self._with_gpu_mirror = True
        self._safe_shared_mem = False

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim

        self._use_gpu = use_gpu

        self._dtype = dtype

        self._verbose = verbose
        self._vlevel = vlevel

        self._is_debug = debug

        self._env_name = env_name

        self._robot_state = None
        self._rhc_refs = None
        self._rhc_status = None

        self._remote_stepper = None

        self._agent_refs = None

        self._n_envs = 0

        self._n_preinit_steps = n_preinit_steps
        
        self._is_first_step = True

        self._perf_timer = PerfSleep()

        self._episode_counters = None
        self._obs = None
        self._actions = None
        self._tot_rewards = None
        self._rewards = None
        self._terminations = None
        self._truncations = None

        self._obs_threshold = 1 # used for clipping observations

        self._base_info = {}
        self._base_info["final_info"] = None
        self._infos = []
        
        self._attach_to_shared_mem()

        self._init_obs()
        self._init_actions(actions_dim)
        self._init_rewards()
        self._init_infos()
        self._init_terminations()
        self._init_truncations()
        
        self._wait_for_sim_env()

        self._init_step()
    
    def _get_this_file_path(self):

        return self._this_path

    def _first_step(self):

        self._activate_rhc_controllers()

        self._is_first_step = False
    
    def _init_step(self):
    
        self.reset()

        for i in range(self._n_preinit_steps): # perform some
            # dummy remote env stepping to make sure to have meaningful 
            # initializations (doesn't increment step counter)
        
            self._check_controllers_registered()

            if self._is_first_step:

                self._first_step()
            
            self._remote_stepper.step() 

            self._remote_stepper.wait()
    
    def _debug(self):
        
        if self._use_gpu:

            self._obs.synch_mirror(from_gpu=True) # copy data from gpu to cpu view
            self._actions.synch_mirror(from_gpu=True)
            self._tot_rewards.synch_mirror(from_gpu=True)
            self._rewards.synch_mirror(from_gpu=True)

        self._obs.synch_all(read=False, wait=True) # copies data on CPU shared mem
        self._actions.synch_all(read=False, wait=True) 
        self._tot_rewards.synch_all(read=False, wait=True)
        self._rewards.synch_all(read=False, wait=True)
    
    def step(self, 
            action, 
            reset: bool = False):
        
        if reset:

            self.reset()

        self._check_controllers_registered() # does not make sense to run training
        # if we lost some controllers

        actions = self._actions.get_torch_view(gpu=self._use_gpu)
        actions[:, :] = action # writes actions
        
        self._apply_actions_to_rhc() # apply agent actions to rhc controller

        self._remote_stepper.step() # triggers simulation + RHC stepping

        self._remote_stepper.wait() # wait for step completion

        self._get_observations()
        self._clamp_obs() # to avoid bad things

        self._compute_rewards()

        # truncated = None
        # info = {}

        self._post_step() # post step operations
    
    def randomize_refs(self,
                env_indxs: torch.Tensor = None):

        self._randomize_refs(env_indxs=env_indxs)

    def reset(self):
        
        self.randomize_refs(env_indxs=None) # randomize all refs across envs

        self._actions.reset()
        self._obs.reset()
        self._rewards.reset()
        self._tot_rewards.reset()

        self._terminations.reset()
        self._truncations.reset()

        self._episode_counters.reset()

        self._remote_reset()

        # self._get_observations()
        # self._clamp_obs() # to avoid bad things
    
    def _remote_reset(self):

        pass

    def close(self):
        
        # close all shared mem. clients
        self._robot_state.close()
        self._rhc_refs.close()
        self._rhc_status.close()
        
        self._remote_stepper.close()
        
        self._episode_counters.close()

        # closing env.-specific shared data
        self._obs.close()
        self._actions.close()
        self._rewards.close()
        self._tot_rewards.close()

        self._terminations.close()
        self._truncations.close()

    def get_last_obs(self):
    
        return self._obs.get_torch_view(gpu=self._use_gpu)

    def get_last_actions(self):
    
        return self._actions.get_torch_view(gpu=self._use_gpu)
    
    def get_last_rewards(self):

        return self._tot_rewards.get_torch_view(gpu=self._use_gpu)

    def get_last_terminations(self):
        
        return self._terminations.get_torch_view(gpu=self._use_gpu)

    def get_last_truncations(self):
                                 
        return self._truncations.get_torch_view(gpu=self._use_gpu)
                            
    def obs_dim(self):

        return self._obs_dim
    
    def actions_dim(self):

        return self._actions_dim
    
    def using_gpu(self):

        return self._use_gpu

    def name(self):

        return self._env_name

    def n_envs(self):

        return self._n_envs

    def dtype(self):
                                    
        return self._dtype 
    
    def _get_obs_names(self):

        # to be overridden by child class

        return None
    
    def _get_action_names(self):

        # to be overridden by child class

        return None
    
    def _get_rewards_names(self):

        # to be overridden by child class

        return None
    
    def _init_obs(self):
        
        self._obs = Observations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=self._get_obs_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        self._obs.run()
        
    def _init_actions(self, actions_dim: int):
        
        self._actions = Actions(namespace=self._namespace,
                            n_envs=self._n_envs,
                            action_dim=self._actions_dim,
                            action_names=self._get_action_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)

        self._actions.run()

    def _init_rewards(self):
        
        self._rewards = Rewards(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_rewards=len(self._get_rewards_names()),
                            reward_names=self._get_rewards_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._tot_rewards = TotRewards(namespace=self._namespace,
                            n_envs=self._n_envs,
                            reward_names=["total_reward"],
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._rewards.run()
        self._tot_rewards.run()
    
    def _init_terminations(self):

        # Boolean array indicating whether each environment episode has terminated after 
        # the current step. An episode termination could occur based on predefined conditions
        # in the environment, such as reaching a goal or exceeding a time limit.

        self._terminations = Terminations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        
        self._terminations.run()
    
    def _init_truncations(self):

        # Boolean array indicating whether each environment episode has been truncated 
        # after the current step. Truncation usually means that the episode has been 
        # forcibly ended before reaching a natural termination point.
        
        self._truncations = Truncations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        
        self._truncations.run()
    
    def _init_infos(self):
        
        # Additional information about the environment's response. It can include various 
        # details such as diagnostics, statistics, or any custom information provided by the 
        # environment

        device = "cuda" if self._use_gpu else "cpu"

        for i in range(self._n_envs):
            self._infos.append(self._base_info)
        
    def _attach_to_shared_mem(self):

        # runs shared mem clients for getting observation and setting RHC commands
        self._robot_state = RobotState(namespace=self._namespace,
                                is_server=False, 
                                with_gpu_mirror=self._use_gpu,
                                safe=self._safe_shared_mem,
                                verbose=self._verbose,
                                vlevel=self._vlevel)
        
        self._rhc_refs = RhcRefs(namespace=self._namespace,
                            is_server=False,
                            with_gpu_mirror=self._use_gpu,
                            safe=self._safe_shared_mem,
                            verbose=self._verbose,
                            vlevel=self._vlevel)

        self._rhc_status = RhcStatus(namespace=self._namespace,
                                is_server=False,
                                with_gpu_mirror=self._use_gpu,
                                verbose=self._verbose,
                                vlevel=self._vlevel)
        
        self._robot_state.run()
        self._rhc_refs.run()
        self._rhc_status.run()

        self._n_envs = self._robot_state.n_robots()

        # run server for agent commands
        self._agent_refs = AgentRefs(namespace=self._namespace,
                                is_server=True,
                                n_robots=self._n_envs,
                                n_jnts=self._robot_state.n_jnts(),
                                n_contacts=self._robot_state.n_contacts(),
                                contact_names=self._robot_state.contact_names(),
                                q_remapping=None,
                                with_gpu_mirror=True,
                                force_reconnection=True,
                                safe=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                fill_value=0)
        self._agent_refs.run()
        
        # remote stepper for coordination with sim environment
        self._remote_stepper = RemoteEnvStepper(namespace=self._namespace,
                            is_server=False,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe_shared_mem)
        self._remote_stepper.run()
        self._remote_stepper.training_env_ready()
        
        # episode steps counters 
        self._episode_counters = TimeUnlimitedTasksEpCounter(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_steps_limit=self._time_limit_nsteps,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=False) # handles step counter through episodes and through envs
        self._episode_counters.run()
        self._episode_counters.reset()

        # debug data servers
        traing_env_param_dict = {}
        traing_env_param_dict["use_gpu"] = self._use_gpu
        traing_env_param_dict["debug"] = self._is_debug
        traing_env_param_dict["n_preinit_steps"] = self._n_preinit_steps
        traing_env_param_dict["n_preinit_steps"] = self._n_envs
        
        self._training_sim_info = SharedTrainingEnvInfo(namespace=self._namespace,
                is_server=True, 
                training_env_params_dict=traing_env_param_dict,
                safe=False,
                force_reconnection=True,
                verbose=self._verbose,
                vlevel=self._vlevel)
        self._training_sim_info.run()
        
    def _activate_rhc_controllers(self):

        self._rhc_status.activation_state.torch_view[:, :] = True

        self._rhc_status.activation_state.synch_all(read=False, wait=True) # activates all controllers
    
    def _synch_obs(self,
            gpu=True):

        # read from shared memory on CPU
        # root link state
        self._robot_state.root_state.synch_all(read = True, wait = True)
        # refs for root link
        self._rhc_refs.rob_refs.root_state.synch_all(read = True, wait = True)
        # rhc cost
        self._rhc_status.rhc_cost.synch_all(read = True, wait = True)
        # rhc constr. violations
        self._rhc_status.rhc_constr_viol.synch_all(read = True, wait = True)
        # failure states
        self._rhc_status.fails.synch_all(read = True, wait = True)

        if gpu:

            # copies data to "mirror" on GPU
            self._robot_state.root_state.synch_mirror(from_gpu=False) # copies shared data on GPU
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_cost.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_constr_viol.synch_mirror(from_gpu=False)
            self._rhc_status.fails.synch_mirror(from_gpu=False)

            #torch.cuda.synchronize() # ensuring that all the streams on the GPU are completed \
            # before the CPU continues execution

    def _synch_refs(self,
            gpu=True):

        if gpu:
            # copies latest refs from GPU to CPU shared mem for debugging
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=True) 

        self._agent_refs.rob_refs.root_state.synch_all(read=False, wait = True) # write on shared mem

    def _clamp_obs(self):

        self._obs.get_torch_view(gpu=self._use_gpu).clamp_(-self._obs_threshold, self._obs_threshold)
    
    def _wait_for_sim_env(self):

        while not self._remote_stepper.is_sim_env_ready():
    
            warning = f"Waiting for sim env to be ready..."

            Journal.log(self.__class__.__name__,
                "_wait_for_sim_env",
                warning,
                LogType.WARN,
                throw_when_excep = True)
            
            self._perf_timer.clock_sleep(2000000000) # nanoseconds 
        
        info = f"Sim. env ready."

        Journal.log(self.__class__.__name__,
            "_wait_for_sim_env",
            info,
            LogType.INFO,
            throw_when_excep = True)
    
    def _check_controllers_registered(self):

        self._rhc_status.controllers_counter.synch_all(read=True, wait=True)

        n_connected_controllers = self._rhc_status.controllers_counter.torch_view[0, 0].item()

        if not n_connected_controllers == self._n_envs:

            exception = f"Expected {self._n_envs} controllers to be active during training, " + \
                f"but got {n_connected_controllers}"

            Journal.log(self.__class__.__name__,
                "_check_controllers_registered",
                exception,
                LogType.EXCEP,
                throw_when_excep = False)
            
            self.close()

            exit()
    
    def _check_truncations(self):

        truncations = self._truncations.get_torch_view(gpu=self._use_gpu)

        # time unlimited episodes, using time limits just for diversifying 
        # experience
        truncations[:, :] = self._episode_counters.time_limits_reached() 
        
        self.randomize_refs(env_indxs=truncations.flatten()) # randomize 
        # refs of envs that reached the time limit

        if self._use_gpu:
            # from GPU to CPU 
            self._truncations.synch_mirror(from_gpu=True) 

        self._truncations.synch_all(read=False, wait = True) # writes on shared mem
    
    def _check_terminations(self):

        terminations = self._terminations.get_torch_view(gpu=self._use_gpu)
        # handle episodes termination
        terminations[:, :] = self._rhc_status.fails.get_torch_view(gpu=self._use_gpu)

        if self._use_gpu:
            # from GPU to CPU 
            self._terminations.synch_mirror(from_gpu=True) 
        
        self._terminations.synch_all(read=False, wait = True) # writes on shared mem

    def _post_step(self):
        
        self._episode_counters.increment() # first increment counters

        self._check_truncations() 
        self._check_terminations()

        terminated = self._terminations.get_torch_view(gpu=self._use_gpu)
        truncated = self._truncations.get_torch_view(gpu=self._use_gpu)

        # only ending episode if termination reached (
        # see "Time Limits in Reinforcement Learning" by F. Pardo)

        episode_finished = torch.logical_or(terminated,
                                        truncated)
        self._episode_counters.reset(to_be_reset=episode_finished)
        
        stepper = self._remote_stepper.get_stepper()
        stepper.reset(env_mask=episode_finished) # remotely reset envs for which 
        # the episode is terminated

        # reset counter if either terminated
        if self._is_debug:
            
            self._debug()

    @abstractmethod
    def _apply_actions_to_rhc(self):

        pass

    @abstractmethod
    def _compute_rewards(self):
        
        pass

    @abstractmethod
    def _get_observations(self):
                
        pass
    
    @abstractmethod
    def _randomize_refs(self,
                env_indxs: torch.Tensor = None):
        
        pass
