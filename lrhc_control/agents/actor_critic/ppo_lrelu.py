
import torch.nn as nn
import torch
from torch.distributions.normal import Normal

class ActorCriticLRelu(nn.Module):

    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            actor_std: float = 0.01, 
            critic_std: float = 1.0):

        self._actor_std = actor_std
        self._critic_std = critic_std
        
        self._lrelu_slope = 0.01

        super().__init__()
            
        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
            
        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(self._obs_dim, 64)),
            nn.LeakyReLU(self._lrelu_slope),
            self._layer_init(nn.Linear(64, 64)),
            nn.LeakyReLU(self._lrelu_slope),
            self._layer_init(nn.Linear(64, 1), std=self._critic_std),
        ) # (stochastic critic)
        self.actor_mean = nn.Sequential(
            self._layer_init(nn.Linear(self._obs_dim, 64)),
            nn.LeakyReLU(self._lrelu_slope),
            self._layer_init(nn.Linear(64, 64)),
            nn.LeakyReLU(self._lrelu_slope),
            self._layer_init(nn.Linear(64, self._actions_dim), std=self._actor_std),
            nn.Tanh()
        ) # (stochastic actor)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self._actions_dim))

    def get_impl_path(self):

        import os 
        
        return os.path.abspath(__file__)
    
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