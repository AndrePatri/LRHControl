import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from lrhc_control.utils.nn.normalization_utils import RunningNormalizer 

from typing import List

from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import VLevel

class SACAgent(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            norm_obs: bool = True,
            device:str="cuda",
            dtype=torch.float32,
            is_eval:bool=False,
            epsilon:float=1e-8,
            debug:bool=False,
            layer_size_actor:int=256,
            layer_size_critic:int=256):

        super().__init__()

        self._normalize_obs = norm_obs

        self._debug = debug
        layer_size_actor=layer_size_actor
        layer_size_critic=layer_size_critic

        self.actor = None
        self.qf1 = None
        self.qf1_target = None
        self.qf2 = None
        self.qf2_target = None

        self._torch_device = device
        self._torch_dtype = dtype

        self.actor = Actor(obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    actions_ub=actions_ub,
                    actions_lb=actions_lb,
                    device=device,
                    dtype=dtype,
                    layer_size=layer_size_actor
                    )
        self.qf1 = CriticQ(obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    device=device,
                    dtype=dtype,
                    layer_size=layer_size_critic)

        if not is_eval: # just needed for training
            self.qf1_target = CriticQ(obs_dim=obs_dim,
                        actions_dim=actions_dim,
                        device=device,
                        dtype=dtype,
                        layer_size=layer_size_critic)
            self.qf2 = CriticQ(obs_dim=obs_dim,
                        actions_dim=actions_dim,
                        device=device,
                        dtype=dtype,
                        layer_size=layer_size_critic)
            self.qf2_target = CriticQ(obs_dim=obs_dim,
                        actions_dim=actions_dim,
                        device=device,
                        dtype=dtype,
                        layer_size=layer_size_critic)
        
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.running_norm = None
        if self._normalize_obs:
            self.running_norm = RunningNormalizer((obs_dim,), epsilon=epsilon, 
                                    device=device, dtype=dtype, 
                                    freeze_stats=is_eval,
                                    debug=self._debug)
            self.running_norm.type(dtype) # ensuring correct dtype for whole module

    def get_impl_path(self):
        import os 
        return os.path.abspath(__file__)
    
    def get_action(self, x):
        if self.running_norm is not None:
            x = self.running_norm(x)
        return self.actor.get_action(x)
    
    def get_qf1_val(self, x, a):
        if self.running_norm is not None:
            x = self.running_norm(x)
        return self.qf1(x, a)

    def get_qf2_val(self, x, a):
        if self.running_norm is not None:
            x = self.running_norm(x)
        return self.qf2(x, a)
    
    def get_qf1t_val(self, x, a):
        if self.running_norm is not None:
            x = self.running_norm(x)
        return self.qf1_target(x, a)
    
    def get_qf2t_val(self, x, a):
        if self.running_norm is not None:
            x = self.running_norm(x)
        return self.qf2_target(x, a)

    def load_state_dict(self, param_dict):

        missing, unexpected = super().load_state_dict(param_dict,
            strict=False)
        if not len(missing)==0:
            Journal.log(self.__class__.__name__,
                "load_state_dict",
                f"These parameters are missing from the provided state dictionary: {str(missing)}\n",
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(unexpected)==0:
            Journal.log(self.__class__.__name__,
                "load_state_dict",
                f"These parameters present in the provided state dictionary are not needed: {str(unexpected)}\n",
                LogType.WARN)
            
class CriticQ(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            device:str="cuda",
            dtype=torch.float32,
            layer_size:int=256):

        super().__init__()

        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        self._q_net_dim = self._obs_dim+self._actions_dim

        size_internal_layer=layer_size
    
        self._q_net = nn.Sequential(
            self._layer_init(layer=nn.Linear(self._q_net_dim, size_internal_layer), device=self._torch_device,dtype=self._torch_dtype),
            nn.ReLU(),
            self._layer_init(layer=nn.Linear(size_internal_layer, size_internal_layer), device=self._torch_device,dtype=self._torch_dtype),
            nn.ReLU(),
            self._layer_init(layer=nn.Linear(size_internal_layer, 1), device=self._torch_device,dtype=self._torch_dtype),
        )
        self._q_net.type(self._torch_dtype) # ensuring compatible dtypes

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _layer_init(self, 
            layer, 
            std=torch.sqrt(torch.tensor(2)), 
            bias_const=0.0,
            device: str = "cuda",
            dtype = torch.float32):
        # device
        layer.to(device)
        # dtype
        layer.weight.data.type(dtype)
        layer.bias.data.type(dtype)
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self._q_net(x)

class Actor(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            device:str="cuda",
            dtype=torch.float32,
            layer_size:int=256):
        super().__init__()

        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        
        # action scale and bias
        if actions_ub is None:
            actions_ub = [1] * actions_dim
        if actions_lb is None:
            actions_lb = [-1] * actions_dim
        if (len(actions_ub) != actions_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Actions ub list length should be equal to {actions_dim}, but got {len(actions_ub)}",
                LogType.EXCEP,
                throw_when_excep = True)
        if (len(actions_lb) != actions_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Actions lb list length should be equal to {actions_dim}, but got {len(actions_lb)}",
                LogType.EXCEP,
                throw_when_excep = True)
        self._actions_ub = torch.tensor(actions_ub, dtype=self._torch_dtype, 
                                device=self._torch_device)
        self._actions_lb = torch.tensor(actions_lb, dtype=self._torch_dtype,
                                device=self._torch_device)
        action_scale = torch.full((actions_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        action_scale[:] = (self._actions_ub-self._actions_lb)/2.0
        self.register_buffer(
            "action_scale", action_scale
        )
        actions_bias = torch.full((actions_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        actions_bias[:] = (self._actions_ub+self._actions_lb)/2.0
        self.register_buffer(
            "action_bias", actions_bias
        )

        size_internal_layer = layer_size
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

        self._fc12 = nn.Sequential(
            self._layer_init(layer=nn.Linear(self._obs_dim, size_internal_layer), device=self._torch_device,dtype=self._torch_dtype),
            nn.ReLU(),
            self._layer_init(layer=nn.Linear(size_internal_layer, size_internal_layer), device=self._torch_device,dtype=self._torch_dtype),
            nn.ReLU()
        )
        self.fc_mean = self._layer_init(layer=nn.Linear(size_internal_layer, self._actions_dim), device=self._torch_device,dtype=self._torch_dtype)
        self.fc_logstd = self._layer_init(layer=nn.Linear(size_internal_layer, self._actions_dim), device=self._torch_device,dtype=self._torch_dtype)

        # ensuring correct dtypes
        self._fc12.type(self._torch_dtype)
        self.fc_mean.type(self._torch_dtype)
        self.fc_logstd.type(self._torch_dtype)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_impl_path(self):
        import os 
        return os.path.abspath(__file__)
    
    def _layer_init(self, 
            layer, 
            std=torch.sqrt(torch.tensor(2)), 
            bias_const=0.0,
            device: str = "cuda",
            dtype = torch.float32):
        # device
        layer.to(device)
        # dtype
        layer.weight.data.type(dtype)
        layer.bias.data.type(dtype)
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, x):
        x = self._fc12(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX -self. LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

if __name__ == "__main__":  
    
    device = "cuda"
    dummy_obs = torch.full(size=(2, 5),dtype=torch.float32,device=device,fill_value=0) 

    sofqn = CriticQ(obs_dim=5,actions_dim=3,
            norm_obs=True,
            device=device,
            dtype=torch.float32,
            is_eval=False)
    
    print("Db prints Q")
    print(f"N. params: {sofqn.get_n_params()}")
    
    dummy_a = torch.full(size=(2, 3),dtype=torch.float32,device=device,fill_value=0)
    q_v = sofqn.forward(x=dummy_obs,a=dummy_a)
    print(q_v)

    actor = Actor(obs_dim=5,actions_dim=3,
            actions_lb=[-1.0, -1.0, -1.0],actions_ub=[1.0, 1.0, 1.0],
            norm_obs=True,
            device=device,
            dtype=torch.float32,
            is_eval=False)
    
    print("Db prints Actor")
    print(f"N. params: {actor.get_n_params()}")
    output=actor.forward(x=dummy_obs)
    print(output)
    print(actor.get_action(x=dummy_obs))