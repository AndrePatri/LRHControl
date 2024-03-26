
import torch.nn as nn
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F

class MixedTanh(nn.Module):
    def __init__(self, input_dim: int, 
            n_tanh_outputs: int,
            tanh_lb=-1, tanh_ub=1):
        
        super().__init__()

        self._tanh_lb = tanh_lb
        self._tanh_ub = tanh_ub

        self._input_dim = input_dim
        self._output_dim = input_dim
        self._n_tanh_outputs = n_tanh_outputs
        self._n_identity_outputs = self._output_dim - self._n_tanh_outputs

    def forward(self, input):
        identity_part = input[:, :self._n_identity_outputs]
        aux = F.tanh(input[:, self._n_identity_outputs:]) 
        tanh_part = self._tanh_lb + 0.5 * (aux + 1.0) * (self._tanh_ub - self._tanh_lb)
        print("iiiiiiii")
        print(identity_part)
        print(tanh_part)
        return torch.cat((identity_part, tanh_part), dim=1)
    
class ActorCriticThB(nn.Module):

    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            n_tanh_outputs: int = 0,
            actor_std: float = 0.01, 
            critic_std: float = 1.0,
            tanh_lb=-1, tanh_ub=1):

        self._actor_std = actor_std
        self._critic_std = critic_std
        
        super().__init__()
            
        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        self._n_tanh_outputs = n_tanh_outputs
        self._tanh_lb = tanh_lb
        self._tanh_ub = tanh_ub

        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(self._obs_dim, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 1), std=self._critic_std),
        ) # (stochastic critic)
        if self._n_tanh_outputs > 0:
            self.actor_mean = nn.Sequential(
                self._layer_init(nn.Linear(self._obs_dim, 64)),
                nn.Tanh(),
                self._layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                self._layer_init(nn.Linear(64, self._actions_dim), std=self._actor_std),
                MixedTanh(input_dim=self._actions_dim, 
                    n_tanh_outputs=self._n_tanh_outputs,
                    tanh_lb=self._tanh_lb, tanh_ub=self._tanh_ub)
            ) # (stochastic actor)
        else:
            self.actor_mean = nn.Sequential(
                self._layer_init(nn.Linear(self._obs_dim, 64)),
                nn.Tanh(),
                self._layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                self._layer_init(nn.Linear(64, self._actions_dim), std=self._actor_std)
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

        # print("action_mean")
        # print(torch.isfinite(action_mean).all())

        # print("Log std")
        # print(torch.isfinite(action_std).all())
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def get_action(self, x):

        action_mean = self.actor_mean(x)

        action_logstd = self.actor_logstd.expand_as(action_mean)

        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def _layer_init(self, layer, std=torch.sqrt(torch.tensor(2)), bias_const=0.0):

        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        
        return layer