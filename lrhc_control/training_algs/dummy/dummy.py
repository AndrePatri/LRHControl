from lrhc_control.training_algs.dummy.dummy_test_algo_base import DummyTestAlgoBase

import torch 
import torch.nn as nn
import torch.nn.functional as F

import os

import time

class Dummy(DummyTestAlgoBase):

    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    remote_db=remote_db,
                    seed=seed)

        self._this_child_path = os.path.abspath(__file__) # overrides parent
    
    def _collect_transition(self):
        
        # experience collection
        self._switch_training_mode(train=False)

        obs = self._env.get_obs(clone=True) # also accounts for resets when envs are 
        # either terminated or truncated. CRUCIAL: we need to clone, 
        # otherwise obs is be a view and will be overridden in the call to step
        # with next_obs!!!
        actions, _, mean = self._agent.get_action(x=obs)
        actions = actions.detach()
                
        # perform a step of the (vectorized) env and retrieve trajectory
        env_step_ok = self._env.step(actions)

        return env_step_ok