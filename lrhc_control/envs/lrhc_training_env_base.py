from gymnasium import spaces

import numpy as np
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
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True):
        
        self._env_index = 0

        self._namespace = namespace
        self._with_gpu_mirror = True
        self._safe_shared_mem = False

        self._use_gpu = use_gpu

        self._verbose = verbose
        self._vlevel = vlevel

        self._robot_state = None
        self._rhc_refs = None
        self._rhc_status = None

        self._remote_stepper = None

        self._agent_refs = None

        self._step_counter = 0

        self._n_envs = 0

        self._is_first_step = True

        self._perf_timer = PerfSleep()

        self._torch_obs = None
        self._torch_actions = None
        
        self._attach_to_shared_mem()

        self._init_obs(obs_dim)
        self._init_actions(actions_dim)
        
        self._wait_for_sim_env()

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, obs_dim), dtype=np.float32)
    
        self.action_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, actions_dim), dtype=np.float32)
    
    def step(self, action):
        
        self._check_controllers_registered() # does not make sense to run training
        # if we lost some controllers

        if self._is_first_step:

            self._activate_rhc_controllers()

            self._is_first_step = False

        # self._apply_rhc_actions(agent_action = action) # first apply actions to rhc controller

        self._remote_stepper.step() # trigger simulation stepping

        self._remote_stepper.wait() # blocking

        # observations = self._get_observations()
        # rewards = self._compute_reward()

        # truncated = None
        # info = {}

        self._step_counter +=1

        # return observations, rewards, self._check_termination(), truncated, info
    
        return None, None, None, None, None

    def reset(self, seed=None, options=None):
        
        self._step_counter = 0

        self._randomize_agent_refs()

        # observation = self._get_observations()
        observation = None

        info = {}
        
        return observation, info

    def close(self):
        
        # close all shared mem. clients
        self._robot_state.close()
        self._rhc_refs.close()
        self._rhc_status.close()

        self._remote_stepper.close()

    def obs_dim(self):

        return self._torch_obs.shape[1]
    
    def actions_dim(self):

        return self._torch_actions.shape[1]
 
    def _init_obs(self, obs_dim: int):
        
        device = "cuda" if self._use_gpu else "cpu"

        self._torch_obs = torch.full(size=(self._n_envs, obs_dim), 
                                    fill_value=0,
                                    dtype=torch.float32,
                                    device=device)
        
    def _init_actions(self, actions_dim: int):
        
        device = "cuda" if self._use_gpu else "cpu"

        self._torch_actions = torch.full(size=(self._n_envs, actions_dim), 
                                    fill_value=0,
                                    dtype=torch.float32,
                                    device=device)

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

if __name__ == '__main__':

    from stable_baselines3.common.env_checker import check_env

    env = LRhcTrainingVecEnv()

    check_env(env) # checks custom environment and output additional warnings if needed
