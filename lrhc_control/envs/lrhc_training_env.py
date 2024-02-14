import gymnasium as gym
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

class LRhcTrainingVecEnv(gym.Env):

    """Remote training environment for Learning-based Receding Horizon Control"""

    def __init__(self,
            namespace: str,
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

        self._attach_to_shared_mem()
        
        self._wait_for_sim_env()

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 4), dtype=np.float32)
    
        self.action_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 1), dtype=np.float32)

        super().__init__()
    
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

    def _apply_rhc_actions(self,
                agent_action):

        # agent_action = torch.tensor(agent_action)

        # rhc_current_ref = self._rhc_refs.rob_refs.root_state.get_p(gpu=True)
        # rhc_current_ref[:, 2:3] = agent_action[0] # overwrite z ref

        # if self._use_gpu:

        #     self._rhc_refs.rob_refs.root_state.set_p(p=rhc_current_ref,
        #                                     gpu=True) # write ref on gpu
        #     self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True) # write from gpu to cpu mirror
        #     self._rhc_refs.rob_refs.root_state.synch_all(read=False, wait=True) # write mirror to shared mem
            
        # else:

        #     self._rhc_refs.rob_refs.root_state.set_p(p=rhc_current_ref,
        #                                     gpu=False) # write ref on cpu mirror
        #     self._rhc_refs.rob_refs.root_state.synch_all(read=False, wait=True) # write mirror to shared mem

        a = None

    def _compute_reward(self):
        
        return None
        
        self._synch_data()

        rhc_h_ref = self._rhc_refs.rob_refs.root_state.get_p(gpu=True)[:, 2:3] # getting z ref
        robot_h = self._robot_state.root_state.get_p(gpu=True)[:, 2:3]

        h_error = (rhc_h_ref - robot_h)

        # rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=True)
        # rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=True)
        
        # reward = torch.norm(h_error, p=2) + rhc_cost + rhc_const_viol
        reward = torch.norm(h_error, p=2)

        return reward.item()

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

    def _get_observations(self):
                
        self._synch_data()

        if self._use_gpu:
            
            agent_h_ref = self._agent_refs.rob_refs.root_state.get_p(gpu=True)[:, 2:3] # getting z ref
            robot_h = self._robot_state.root_state.get_p(gpu=True)[:, 2:3]
            rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=True)
            rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=True)

            return torch.cat((robot_h, rhc_cost, rhc_const_viol, 
                                agent_h_ref),dim=1)
        
        else:

            agent_h_ref = self._agent_refs.rob_refs.root_state.get_p(gpu=False)[:, 2:3] # getting z ref
            robot_h = self._robot_state.root_state.get_p(gpu=False)[:, 2:3]
            rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=False)
            rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=False)

            return torch.cat((robot_h, rhc_cost, rhc_const_viol, 
                                agent_h_ref),dim=1)
    
    def _randomize_agent_refs(self):
        
        agent_p_ref_current = self._agent_refs.rob_refs.root_state.get_p(gpu=True)

        agent_p_ref_current[:, 2:3] = (1.0 - 0.2) * torch.rand_like(agent_p_ref_current[:, 2:3]) + 0.2 # randomize h ref

        self._agent_refs.rob_refs.root_state.set_p(p=agent_p_ref_current,
                                        gpu=True)

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
            
            self._perf_timer.clock_sleep(1000000000) # nanoseconds 
        
        info = f"Sim. env ready."

        Journal.log(self.__class__.__name__,
            "_wait_for_sim_env",
            info,
            LogType.INFO,
            throw_when_excep = True)
    
    def step(self, action):
        
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

        observation = self._get_observations()
        info = {}
        
        return observation, info

    def render(self):
        
        pass # no need for rendering

    def close(self):
        
        # close all shared mem. clients
        self._robot_state.close()
        self._rhc_refs.close()
        self._rhc_status.close()

        self._remote_stepper.close()

if __name__ == '__main__':

    from stable_baselines3.common.env_checker import check_env

    env = LRhcTrainingVecEnv()

    check_env(env) # checks custom environment and output additional warnings if needed
