import torch 
import torch.nn as nn
from torch.distributions.normal import Normal
import math

from aug_mpc.utils.nn.normalization_utils import RunningNormalizer 
from aug_mpc.utils.nn.layer_utils import llayer_init 

from typing import List

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import VLevel

class SACAgent(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            obs_ub: List[float] = None,
            obs_lb: List[float] = None,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            rescale_obs: bool = False,
            norm_obs: bool = True,
            use_action_rescale_for_critic: bool = True,
            device:str="cuda",
            dtype=torch.float32,
            is_eval:bool=False,
            load_qf:bool=False,
            epsilon:float=1e-8,
            debug:bool=False,
            compression_ratio:float=-1.0, # > 0; if [0, 1] compression, >1 "expansion"
            layer_width_actor:int=256,
            n_hidden_layers_actor:int=2,
            layer_width_critic:int=512,
            n_hidden_layers_critic:int=4,
            torch_compile: bool = False,
            add_weight_norm: bool = False,
            add_layer_norm: bool = False,
            add_batch_norm: bool = False):

        super().__init__()

        self._use_torch_compile=torch_compile

        self._layer_width_actor=layer_width_actor
        self._layer_width_critic=layer_width_critic
        self._n_hidden_layers_actor=n_hidden_layers_actor
        self._n_hidden_layers_critic=n_hidden_layers_critic

        self._obs_dim=obs_dim
        self._actions_dim=actions_dim
        self._actions_ub=actions_ub
        self._actions_lb=actions_lb

        self._add_weight_norm=add_weight_norm
        self._add_layer_norm=add_layer_norm
        self._add_batch_norm=add_batch_norm

        self._is_eval=is_eval
        self._load_qf=load_qf

        self._epsilon=epsilon

        if compression_ratio > 0.0:
            self._layer_width_actor=int(compression_ratio*obs_dim)
            self._layer_width_critic=int(compression_ratio*(obs_dim+actions_dim))
        
        self._normalize_obs = norm_obs
        self._rescale_obs=rescale_obs
        if self._rescale_obs and self._normalize_obs:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Both running normalization and obs rescaling is enabled. Was this intentional?",
                LogType.WARN,
                throw_when_excep = True)
        
        self._use_action_rescale_for_critic=use_action_rescale_for_critic

        self._rescaling_epsi=1e-9

        self._debug = debug

        self._torch_device = device
        self._torch_dtype = dtype

        # obs scale and bias
        if obs_ub is None:
            obs_ub = [1] * obs_dim
        if obs_lb is None:
            obs_lb = [-1] * obs_dim
        if (len(obs_ub) != obs_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Observations ub list length should be equal to {obs_dim}, but got {len(obs_ub)}",
                LogType.EXCEP,
                throw_when_excep = True)
        if (len(obs_lb) != obs_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Observations lb list length should be equal to {obs_dim}, but got {len(obs_lb)}",
                LogType.EXCEP,
                throw_when_excep = True)

        self._obs_ub = torch.tensor(obs_ub, dtype=self._torch_dtype, 
                                device=self._torch_device)
        self._obs_lb = torch.tensor(obs_lb, dtype=self._torch_dtype,
                                device=self._torch_device)
        obs_scale = torch.full((obs_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        obs_scale[:] = (self._obs_ub-self._obs_lb)/2.0
        self.register_buffer(
            "obs_scale", obs_scale
        )
        obs_bias = torch.full((obs_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        obs_bias[:] = (self._obs_ub+self._obs_lb)/2.0
        self.register_buffer(
            "obs_bias", obs_bias)
        
        self._build_nets()
        
        self._init_obs_norm()

        msg=f"Created SAC agent with actor [{self._layer_width_actor}, {self._n_hidden_layers_actor}]\
        and critic [{self._layer_width_critic}, {self._n_hidden_layers_critic}] sizes.\
        \n Runningobs normalizer: {type(self.obs_running_norm)} \
        \n Batch normalization: {self._add_batch_norm} \
        \n Layer normalization: {self._add_layer_norm} \
        \n Weight normalization: {self._add_weight_norm} \
        \n Critic input actions are descaled: {self._use_action_rescale_for_critic}"
        Journal.log(self.__class__.__name__,
            "__init__",
            msg,
            LogType.INFO)
    
    def _init_obs_norm(self):
        
        self.obs_running_norm=None
        if self._normalize_obs:
            self.obs_running_norm = RunningNormalizer((self._obs_dim,), 
                                        epsilon=self._epsilon, 
                                        device=self._torch_device, dtype=self._torch_dtype, 
                                        freeze_stats=True, # always start with freezed stats
                                        debug=self._debug)
            self.obs_running_norm.type(self._torch_dtype) # ensuring correct dtype for whole module

    def _build_nets(self):

        if self._add_weight_norm:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Will use weight normalization reparametrization\n",
                LogType.INFO)

        self.actor=None
        self.qf1=None
        self.qf2=None
        self.qf1_target=None
        self.qf2_target=None
        
        self.actor = Actor(obs_dim=self._obs_dim,
                    actions_dim=self._actions_dim,
                    actions_ub=self._actions_ub,
                    actions_lb=self._actions_lb,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    layer_width=self._layer_width_actor,
                    n_hidden_layers=self._n_hidden_layers_actor,
                    add_weight_norm=self._add_weight_norm,
                    add_layer_norm=self._add_layer_norm,
                    add_batch_norm=self._add_batch_norm,
                    )

        if (not self._is_eval) or self._load_qf: # just needed for training or during eval
            # for debug, if enabled
            self.qf1 = CriticQ(obs_dim=self._obs_dim,
                    actions_dim=self._actions_dim,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=self._add_weight_norm,
                    add_layer_norm=self._add_layer_norm,
                    add_batch_norm=self._add_batch_norm)
            self.qf1_target = CriticQ(obs_dim=self._obs_dim,
                    actions_dim=self._actions_dim,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=self._add_weight_norm,
                    add_layer_norm=self._add_layer_norm,
                    add_batch_norm=self._add_batch_norm)
            
            self.qf2 = CriticQ(obs_dim=self._obs_dim,
                    actions_dim=self._actions_dim,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=self._add_weight_norm,
                    add_layer_norm=self._add_layer_norm,
                    add_batch_norm=self._add_batch_norm)
            self.qf2_target = CriticQ(obs_dim=self._obs_dim,
                    actions_dim=self._actions_dim,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=self._add_weight_norm,
                    add_layer_norm=self._add_layer_norm,
                    add_batch_norm=self._add_batch_norm)
        
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())

        if self._use_torch_compile:
            self.obs_running_norm=torch.compile(self.obs_running_norm)
            self.actor = torch.compile(self.actor)
            if (not self._is_eval) or self._load_qf:
                self.qf1 = torch.compile(self.qf1)
                self.qf2 = torch.compile(self.qf2)
                self.qf1_target = torch.compile(self.qf1_target)
                self.qf2_target = torch.compile(self.qf2_target)
            
    def reset(self, reset_stats: bool = False):
        # we should just reinitialize the parameters, but for easiness
        # we recreate the networks

        # force deallocation of objects
        import gc
        del self.actor
        del self.qf1
        del self.qf2
        del self.qf1_target
        del self.qf2_target
        gc.collect()

        self._build_nets()

        if reset_stats: # we also reinitialize obs norm
            self._init_obs_norm()

        # self.obs_running_norm.reset()

    def layer_width_actor(self):
        return self._layer_width_actor

    def n_hidden_layers_actor(self):
        return self._n_hidden_layers_actor

    def layer_width_critic(self):
        return self._layer_width_critic

    def n_hidden_layers_critic(self):
        return self._n_hidden_layers_critic

    def get_impl_path(self):
        import os 
        return os.path.abspath(__file__)
    
    def update_obs_bnorm(self, x):
        self.obs_running_norm.unfreeze()
        self.obs_running_norm.manual_stat_update(x)
        self.obs_running_norm.freeze()

    def _obs_scaling_layer(self, x):
        x=(x-self.obs_bias)
        x=x/(self.obs_scale+self._rescaling_epsi)
        return x
    
    def _preprocess_obs(self, x):
        if self._rescale_obs:
            x = self._obs_scaling_layer(x)
        if self.obs_running_norm is not None:
            x = self.obs_running_norm(x)
        return x

    def _preprocess_actions(self, a):
        if self._use_action_rescale_for_critic:
            a=self.actor.remove_scaling(a=a) # rescale to be in range [-1, 1]
        return a

    def get_action(self, x):
        x = self._preprocess_obs(x)
        return self.actor.get_action(x)
    
    def get_qf1_val(self, x, a):
        x = self._preprocess_obs(x)
        a = self._preprocess_actions(a)
        return self.qf1(x, a)

    def get_qf2_val(self, x, a):
        x = self._preprocess_obs(x)
        a = self._preprocess_actions(a)
        return self.qf2(x, a)
    
    def get_qf1t_val(self, x, a):
        x = self._preprocess_obs(x)
        a = self._preprocess_actions(a)
        return self.qf1_target(x, a)
    
    def get_qf2t_val(self, x, a):
        x = self._preprocess_obs(x)
        a = self._preprocess_actions(a)
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
        
        # sanity check on running normalizer
        import re
        running_norm_pattern = r"running_norm"
        error=f"Found some keys in model state dictionary associated with a running normalizer. Are you running the agent with norm_obs=True?\n"
        if any(re.match(running_norm_pattern, key) for key in unexpected):
            Journal.log(self.__class__.__name__,
                "load_state_dict",
                error,
                LogType.EXCEP,
                throw_when_excep=True)

class CriticQ(nn.Module):
    def __init__(self,
        obs_dim: int, 
        actions_dim: int,
        device: str = "cuda",
        dtype = torch.float32,
        layer_width: int = 512,
        n_hidden_layers: int = 4,
        add_weight_norm: bool = False,
        add_layer_norm: bool = False,
        add_batch_norm: bool = False):

        super().__init__()

        self._lrelu_slope=0.01

        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        self._q_net_dim = self._obs_dim + self._actions_dim

        self._first_hidden_layer_width=self._q_net_dim # fist layer fully connected and of same dim
        # as input

        init_type="kaiming_uniform" # maintains the variance of activations throughout the network
        nonlinearity="leaky_relu" # suited for kaiming

        # Input layer        
        layers=llayer_init(
            layer=nn.Linear(self._q_net_dim, self._first_hidden_layer_width),
            init_type=init_type,
            nonlinearity=nonlinearity,
            a_leaky_relu=self._lrelu_slope,
            device=self._torch_device,
            dtype=self._torch_dtype,
            add_weight_norm=add_weight_norm,
            add_layer_norm=add_layer_norm,
            add_batch_norm=add_batch_norm,
            uniform_biases=False, # constant bias init
            bias_const=0.0
        )
        layers.extend([nn.LeakyReLU(negative_slope=self._lrelu_slope)])
        
        # Hidden layers
        layers.extend(
            llayer_init(
                layer=nn.Linear(self._first_hidden_layer_width, layer_width),
                init_type=init_type,
                nonlinearity=nonlinearity,
                a_leaky_relu=self._lrelu_slope,
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=add_weight_norm,
                add_layer_norm=add_layer_norm,
                add_batch_norm=add_batch_norm,
                uniform_biases=False, # constant bias init
                bias_const=0.0
            )
        )
        layers.extend([nn.LeakyReLU(negative_slope=self._lrelu_slope)])

        for _ in range(n_hidden_layers - 1):
            layers.extend(
                llayer_init(
                    layer=nn.Linear(layer_width, layer_width),
                    init_type=init_type,
                    nonlinearity=nonlinearity,
                    a_leaky_relu=self._lrelu_slope,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm,
                    add_layer_norm=add_layer_norm,
                    add_batch_norm=add_batch_norm,
                    uniform_biases=False, # constant bias init
                    bias_const=0.0
                )
            )
            layers.extend([nn.LeakyReLU(negative_slope=self._lrelu_slope)])

        # Output layer
        layers.extend(
            llayer_init(
                layer=nn.Linear(layer_width, 1),
                init_type="uniform",
                uniform_biases=False, # contact biases
                bias_const=-0.1, # negative to prevent overestimation
                scale_weight=1e-2,
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=False,
                add_layer_norm=False,
                add_batch_norm=False
            )
        )

        # Creating the full sequential network
        self._q_net = nn.Sequential(*layers)
        self._q_net.to(self._torch_device).type(self._torch_dtype)

        print("Critic architecture")
        print(self._q_net)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self._q_net(x)
        
class Actor(nn.Module):
    def __init__(self,
        obs_dim: int, 
        actions_dim: int,
        actions_ub: List[float] = None,
        actions_lb: List[float] = None,
        device: str = "cuda",
        dtype = torch.float32,
        layer_width: int = 256,
        n_hidden_layers: int = 2,
        add_weight_norm: bool = False,
        add_layer_norm: bool = False,
        add_batch_norm: bool = False):
    
        super().__init__()

        self._lrelu_slope=0.01
        
        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        
        self._first_hidden_layer_width=self._obs_dim # fist layer fully connected and of same dim
    
        # Action scale and bias
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
            "action_bias", actions_bias)
        
        # Network configuration
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

        # Input layer followed by hidden layers
        layers=llayer_init(nn.Linear(self._obs_dim, self._first_hidden_layer_width), 
                    init_type="kaiming_uniform",
                    nonlinearity="leaky_relu",
                    a_leaky_relu=self._lrelu_slope,
                    device=self._torch_device, 
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm,
                    add_layer_norm=add_layer_norm,
                    add_batch_norm=add_batch_norm,
                    uniform_biases=False, # constant bias init
                    bias_const=0.0
                    )
        layers.extend([nn.LeakyReLU(negative_slope=self._lrelu_slope)])
    
        # Hidden layers
        layers.extend(
            llayer_init(nn.Linear(self._first_hidden_layer_width, layer_width), 
                init_type="kaiming_uniform",
                nonlinearity="leaky_relu",
                a_leaky_relu=self._lrelu_slope,
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=add_weight_norm,
                add_layer_norm=add_layer_norm,
                add_batch_norm=add_batch_norm,
                uniform_biases=False, # constant bias init
                bias_const=0.0)
        )
        layers.extend([nn.LeakyReLU(negative_slope=self._lrelu_slope)])
        
        for _ in range(n_hidden_layers - 1):
            layers.extend(
                llayer_init(nn.Linear(layer_width, layer_width), 
                    init_type="kaiming_uniform",
                    nonlinearity="leaky_relu",
                    a_leaky_relu=self._lrelu_slope,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm,
                    add_layer_norm=add_layer_norm,
                    add_batch_norm=add_batch_norm,
                    uniform_biases=False, # constant bias init
                    bias_const=0.0)            
            )
            layers.extend([nn.LeakyReLU(negative_slope=self._lrelu_slope)])
        
        # Sequential layers for the feature extractor
        self._fc12 = nn.Sequential(*layers)

        # Mean and log_std layers
        out_fc_mean=llayer_init(nn.Linear(layer_width, self._actions_dim), 
                        init_type="uniform",
                        uniform_biases=False, # constant bias init
                        bias_const=0.0,
                        scale_weight=1e-3, # scaling (output layer)
                        scale_bias=1.0,
                        device=self._torch_device, 
                        dtype=self._torch_dtype,
                        add_weight_norm=False,
                        add_layer_norm=False,
                        add_batch_norm=False
                        )
        self.fc_mean = nn.Sequential(*out_fc_mean)
        out_fc_logstd= llayer_init(nn.Linear(layer_width, self._actions_dim), 
                        init_type="uniform",
                        uniform_biases=False,
                        bias_const=math.log(0.5),
                        scale_weight=1e-3, # scaling (output layer)
                        scale_bias=1.0,
                        device=self._torch_device, 
                        dtype=self._torch_dtype,
                        add_weight_norm=False,
                        add_layer_norm=False,
                        add_batch_norm=False,
                        )
        self.fc_logstd = nn.Sequential(*out_fc_logstd)

        # Move all components to the specified device and dtype
        self._fc12.to(device=self._torch_device, dtype=self._torch_dtype)
        self.fc_mean.to(device=self._torch_device, dtype=self._torch_dtype)
        self.fc_logstd.to(device=self._torch_device, dtype=self._torch_dtype)

        print("Actor architecture")
        print(self._fc12)
        print(self.fc_mean)
        print(self.fc_logstd)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = self._fc12(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick (for SAC we neex action
        # to be differentible since we use Q nets. Using sample() would break the
        # comp. graph and not allow gradients to flow)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob_vec = normal.log_prob(x_t) # per-dimension log prob before tanh
        log_prob_vec = log_prob_vec - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # tanh Jacobian + scaling
        log_prob_sum = log_prob_vec.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, (log_prob_sum, log_prob_vec), mean
    
    def remove_scaling(self, a):
        return (a - self.action_bias)/self.action_scale

if __name__ == "__main__":  
    device = "cpu"  # or "cpu"
    import time
    obs_dim = 273
    agent = SACAgent(
        obs_dim=obs_dim,
        actions_dim=10,
        actions_lb=None,
        actions_ub=None,
        obs_lb=None,
        obs_ub=None,
        rescale_obs=False,
        norm_obs=True,
        use_action_rescale_for_critic=True,
        is_eval=True,
        compression_ratio=0.6,
        layer_width_actor=128,
        layer_width_critic=256,
        n_hidden_layers_actor=3,
        n_hidden_layers_critic=3,
        device=device,
        dtype=torch.float32,
        add_weight_norm=True,
        add_layer_norm=False,
        add_batch_norm=False
    )

    n_samples = 10000
    random_obs = torch.rand((1, obs_dim), dtype=torch.float32, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for i in range(n_samples):
        actions, _, mean = agent.get_action(x=random_obs)
        actions = actions.detach()
        actions[:, :] = mean.detach()

    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    avrg_eval_time = (end - start) / n_samples
    print(f"Average policy evaluation time on {device}: {avrg_eval_time:.6f} s")
