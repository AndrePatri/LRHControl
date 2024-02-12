import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus

from lrhc_control.utils.shared_data.remote_stepping import RemoteStepper

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

class LRhcTrainingEnv(gym.vector.VectorEnv):

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

        self._step_counter = 0

        self._n_envs = 0

        self._is_first_step = True

        self._attach_to_shared_mem()
        
        observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 4), dtype=np.float32)
    
        action_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 1), dtype=np.float32)

        super().__init__(num_envs=self._n_envs,
                observation_space=observation_space,
                action_space=action_space)
    
    def _attach_to_shared_mem(self):

        # runs shared mem clients
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
        
        self._remote_stepper = RemoteStepper(namespace=self._namespace,
                                    is_server=False,
                                    verbose=self._verbose,
                                    vlevel=self._vlevel,
                                    safe=self._safe_shared_mem)
        
        self._robot_state.run()
        self._rhc_refs.run()
        self._rhc_status.run()

        self._remote_stepper.run()
        
        self._n_envs = self._robot_state.n_robots()

    def _activate_rhc_controllers(self):

        self._rhc_status.activation_state.torch_view[:, :] = True

        self._rhc_status.activation_state.synch_all(read=False, wait=True) # activates all controllers

    def _apply_rhc_actions(self,
                agent_action):

        a = 1

    def _compute_reward(self,
                observation):
        
        reward = None

        return reward

    def _get_observations(self):
        
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

        # torch.cuda.synchronize() # this way we ensure that after this the state on GPU
        # is fully updated

        observations = None

        return observations
    
    def _check_termination(self):

        return None
    
    def step(self, action):
        
        # if self._is_first_step:

        #     self._activate_rhc_controllers()

        #     self._is_first_step = False

        self._apply_rhc_actions(agent_action = action) # first apply actions to rhc controller

        self._remote_stepper.step() # trigger simulation stepping

        print("AAAAAAAAAAAAAAAAAA")
        self._remote_stepper.wait_for_step_done() # blocking
        print("UUUUUUUUUUUUUUUUUUUU")

        self._step_counter +=1

        observation = self._get_observations()

        reward = self._compute_reward(observation)

        truncated = None
        info = None

        return observation, reward, self._check_termination(), truncated, info

    def reset(self, seed=None, options=None):
        
        self._step_counter = 0

        observation = self._get_observations()
        info = None

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

    env = LRhcTrainingEnv()

    check_env(env) # checks custom environment and output additional warnings if needed
