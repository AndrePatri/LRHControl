import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from lrhc_control.utils.nn.normalization_utils import RunningNormalizer 

class SoftQNetwork(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            norm_obs: bool = True,
            device:str="cuda",
            dtype=torch.float32,
            is_eval:bool=False):
        
        self._normalize_obs = norm_obs
        self._is_eval = is_eval

        self._torch_device = device
        self._torch_dtype = dtype

        super().__init__()

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        self._q_net_dim = self._obs_dim+self._actions_dim

        size_internal_layer = 256
        if self._normalize_obs:
            self._q_net = nn.Sequential(
                RunningNormalizer((self._q_net_dim,), epsilon=1e-8, device=self._torch_device, dtype=self._torch_dtype, is_training=not self._is_eval),
                self._layer_init(layer=nn.Linear(self._q_net_dim, size_internal_layer),device=self._torch_device,dtype=self._torch_dtype),
                nn.ReLU(),
                self._layer_init(layer=nn.Linear(size_internal_layer, size_internal_layer), device=self._torch_device,dtype=self._torch_dtype),
                nn.ReLU(),
                self._layer_init(layer=nn.Linear(size_internal_layer, 1), device=self._torch_device,dtype=self._torch_dtype),
            )
        else:
            self._q_net = nn.Sequential(
                self._layer_init(layer=nn.Linear(self._q_net_dim, size_internal_layer),device=self._torch_device,dtype=self._torch_dtype),
                nn.ReLU(),
                self._layer_init(layer=nn.Linear(size_internal_layer, size_internal_layer), device=self._torch_device,dtype=self._torch_dtype),
                nn.ReLU(),
                self._layer_init(layer=nn.Linear(size_internal_layer, 1), device=self._torch_device,dtype=self._torch_dtype),
            )

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
    
    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self._q_net(x)

if __name__ == "__main__":  
    
    device = "cuda"
    sofqn = SoftQNetwork(obs_dim=5,actions_dim=3,
            norm_obs=True,
            device=device,
            dtype=torch.float32,
            is_eval=False)
    
    print("Db prints")
    print(f"N. params: {sofqn.get_n_params()}")
    dummy_obs = torch.full(size=(2, 5),dtype=torch.float32,device=device,fill_value=0) 
    dummy_a = torch.full(size=(2, 3),dtype=torch.float32,device=device,fill_value=0)
    q_v = sofqn.forward(x=dummy_obs,a=dummy_a)
    print(q_v)