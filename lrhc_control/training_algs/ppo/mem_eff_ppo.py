from lrhc_control.training_algs.ppo.actor_critic_algo import ActorCriticAlgoBase

import torch 
import torch.nn as nn

import os

import time

class MemEffPPO(ActorCriticAlgoBase):

    # "memory efficient" implementation of PPO. W.r.t cleanrl's implementation, 
    # this class correctly handles truncations during bootstrap and it does so 
    # efficiently (GPU memory-wise), at the expense of code readability
     
    # TO BE IMPLEMENTED