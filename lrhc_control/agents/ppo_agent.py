
import torch.nn as nn
import torch
from torch.distributions.normal import Normal

class Agent(nn.Module):

    def __init__(self,
            obs_dim: int, 
            actions_dim: int):

        super().__init__()
        
        self._obs_dim = obs_dim
        self._actiosn_dim = actions_dim

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(self._obs_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(self._obs_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, self._actiosn_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self._actiosn_dim))

    def get_value(self, x):

        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def _layer_init(layer, std=torch.sqrt(2), bias_const=0.0):

        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

