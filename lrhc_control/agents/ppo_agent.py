
import torch.nn as nn
import torch
from torch.distributions.normal import Normal

class Agent(nn.Module):

    def __init__(self, envs):

        super().__init__()
        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(torch.tensor(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            self._layer_init(nn.Linear(torch.tensor(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, torch.prod(torch.tensor(envs.single_action_space.shape))), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, torch.prod(torch.tensor(envs.single_action_space.shape))))

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
    
    def _layer_init(self, layer, std=torch.sqrt(torch.tensor(2)), bias_const=0.0):

        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

