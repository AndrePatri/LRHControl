from gymnasium import spaces

import torch

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus

from lrhc_control.utils.shared_data.remote_env_stepper import RemoteEnvStepper
from lrhc_control.utils.shared_data.agent_refs import AgentRefs

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from perf_sleep.pyperfsleep import PerfSleep

from abc import abstractmethod
    
class LRhcTrainingEnvBase():

    """Base class for a remote training environment tailored to Learning-based Receding Horizon Control"""

    def __init__(self,
            namespace: str,
            obs_dim: int,
            actions_dim: int,
            env_name: str = "",
            n_preinit_steps: int = 1,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32):
        
        self._env_index = 0

        self._namespace = namespace
        self._with_gpu_mirror = True
        self._safe_shared_mem = False

        self._use_gpu = use_gpu

        self._dtype = dtype

        self._verbose = verbose
        self._vlevel = vlevel

        self._env_name = env_name

        self._robot_state = None
        self._rhc_refs = None
        self._rhc_status = None

        self._remote_stepper = None

        self._agent_refs = None

        self._step_counter = 0

        self._n_envs = 0

        self._n_preinit_steps = n_preinit_steps
        
        self._is_first_step = True

        self._perf_timer = PerfSleep()

        self._obs = None
        self._actions = None
        self._rewards = None
        self._terminations = None
        self._truncations = None

        self._base_info = {}
        self._base_info["final_info"] = None
        self._infos = []
        
        self._attach_to_shared_mem()

        self._init_obs(obs_dim)
        self._init_actions(actions_dim)
        
        self._wait_for_sim_env()

        self._init_step()
            
    def _first_step(self):

        self._activate_rhc_controllers()

        self._is_first_step = False
    
    def _init_step(self):
        
        for i in range(self._n_preinit_steps): # perform some
            # dummy remote env stepping to make sure to have meaningful 
            # initializations (doesn't increment step counter)
        
            self._check_controllers_registered()

            if self._is_first_step:

                self._first_step()
            
            self._remote_stepper.step() 

            self._remote_stepper.wait()
        
        self._get_observations() # initializes observations
    
    def step(self, action):
        
        self._check_controllers_registered() # does not make sense to run training
        # if we lost some controllers

        # self._apply_rhc_actions(agent_action = action) # first apply actions to rhc controller

        self._remote_stepper.step() # trigger simulation stepping

        self._remote_stepper.wait() # blocking

        # observations = self._get_observations()
        # rewards = self._compute_reward()

        # truncated = None
        # info = {}

        self._step_counter +=1
    
    def reset(self, seed=None, options=None):
        
        self._step_counter = 0

        self._randomize_agent_refs()

        self._actions.zero_()
        self._rewards.zero_()
        self._obs.zero_()

        self._get_observations()

    def close(self):
        
        # close all shared mem. clients
        self._robot_state.close()
        self._rhc_refs.close()
        self._rhc_status.close()

        self._remote_stepper.close()

    def get_last_obs(self):
    
        return self._obs

    def get_last_actions(self):
    
        return self._actions
    
    def get_last_rewards(self):

        return self._rewards

    def get_last_terminations(self):
        
        return self._terminations

    def get_last_truncations(self):
                                 
        return self._truncations
                            
    def obs_dim(self):

        return self._obs.shape[1]
    
    def actions_dim(self):

        return self._actions.shape[1]
    
    def using_gpu(self):

        return self._use_gpu

    def name(self):

        return self._env_name

    def n_envs(self):

        return self._n_envs

    def dtype(self):
                                    
        return self._dtype 
    
    def _init_obs(self, obs_dim: int):
        
        device = "cuda" if self._use_gpu else "cpu"

        self._obs = torch.full(size=(self._n_envs, obs_dim), 
                                    fill_value=0,
                                    dtype=torch.float32,
                                    device=device)
        
    def _init_actions(self, actions_dim: int):
        
        device = "cuda" if self._use_gpu else "cpu"

        self._actions = torch.full(size=(self._n_envs, actions_dim), 
                                    fill_value=0,
                                    dtype=torch.float32,
                                    device=device)

    def _init_rewards(self):
        
        device = "cuda" if self._use_gpu else "cpu"

        self._rewards = torch.full(size=(self._n_envs, 1), 
                                    fill_value=0,
                                    dtype=torch.float32,
                                    device=device)
    
    def _init_terminations(self):

        # Boolean array indicating whether each environment episode has terminated after 
        # the current step. An episode termination could occur based on predefined conditions
        # in the environment, such as reaching a goal or exceeding a time limit.

        device = "cuda" if self._use_gpu else "cpu"

        self._terminations = torch.full(size=(self._n_envs, 1), 
                                    fill_value=False,
                                    dtype=torch.bool,
                                    device=device)
    
    def _init_truncations(self):

        # Boolean array indicating whether each environment episode has been truncated 
        # after the current step. Truncation usually means that the episode has been 
        # forcibly ended before reaching a natural termination point.

        device = "cuda" if self._use_gpu else "cpu"

        self._truncations = torch.full(size=(self._n_envs, 1), 
                                    fill_value=False,
                                    dtype=torch.bool,
                                    device=device)
    
    def _init_infos(self):
        
        # Additional information about the environment's response. It can include various 
        # details such as diagnostics, statistics, or any custom information provided by the 
        # environment

        device = "cuda" if self._use_gpu else "cpu"

        for i in range(self._n_envs):
            self.info.append(self._base_info)
        
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
                                force_reconnection=False,
                                safe=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                fill_value=0)
        self._agent_refs.run()

        self._remote_stepper = RemoteEnvStepper(namespace=self._namespace,
                            is_server=False,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe_shared_mem)
        self._remote_stepper.run()
        self._remote_stepper.training_env_ready()

    def _activate_rhc_controllers(self):

        self._rhc_status.activation_state.torch_view[:, :] = True

        self._rhc_status.activation_state.synch_all(read=False, wait=True) # activates all controllers
    
    def _synch_data(self):

        # root link state
        self._robot_state.root_state.synch_all(read = True, wait = True)
        self._robot_state.root_state.synch_mirror(from_gpu=False) # copies shared data on GPU
        # refs for root link
        self._rhc_refs.rob_refs.root_state.synch_all(read = True, wait = True)
        self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=False)
        # rhc cost
        self._rhc_status.rhc_cost.synch_all(read = True, wait = True)
        self._rhc_status.rhc_cost.synch_mirror(from_gpu=False)
        # rhc constr. violations
        self._rhc_status.rhc_constr_viol.synch_all(read = True, wait = True)
        self._rhc_status.rhc_constr_viol.synch_mirror(from_gpu=False)
        
        # copies latest agent refs to shared mem on CPU (for debugging)
        self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=True) 
        self._agent_refs.rob_refs.root_state.synch_all(read=False, wait = True)

        torch.cuda.synchronize() # this way we ensure that after this the state on GPU
        # is fully updated

    def _check_termination(self):

        return None
    
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
   
    @abstractmethod
    def _apply_rhc_actions(self,
                agent_action):

        pass

    @abstractmethod
    def _compute_reward(self):
        
        pass

    @abstractmethod
    def _get_observations(self):
                
        pass
    
    @abstractmethod
    def _randomize_agent_refs(self):
        
        pass
