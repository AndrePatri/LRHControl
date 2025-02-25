import torch
import math
from torch.nn.utils import weight_norm, remove_weight_norm    
from torch.nn import BatchNorm1d, LayerNorm, Sequential

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import VLevel

def llayer_init(layer, 
    init_type=None,
    nonlinearity="leaky_relu",
    a_leaky_relu: float=0.01,
    bias_const=0.0,
    device: str = "cuda",
    dtype = torch.float32,
    orth_init_gain: float = 1.0,
    uniform_biases: bool = False,
    add_weight_norm: bool = False,
    add_layer_norm: bool = False,
    add_batch_norm: bool = False):

        if add_layer_norm and add_batch_norm:
            Journal.log(self.__class__.__name__,
                "llayer_init",
                f"Cannot use both layer and batch normalization! Choose one of the two.",
                LogType.EXCEP,
                throw_when_excep = True)

        # Move to device and set dtype
        layer.to(device).type(dtype)

        # Apply weight initialization based on the init_type argument
        if init_type is not None:
            if init_type == "orthogonal":
                torch.nn.init.orthogonal_(layer.weight, gain=orth_init_gain)
                torch.nn.init.constant_(layer.bias, bias_const)
            elif init_type == "uniform":
                k=1/layer.in_features
                bound=math.sqrt(k)
                torch.nn.init.uniform_(layer.weight,a=-bound,b=bound)
                if not uniform_biases:
                    torch.nn.init.constant_(layer.bias, bias_const)
                else:
                    torch.nn.init.uniform_(layer.bias,a=-bound,b=bound)
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity, a=a_leaky_relu,mode='fan_in')
                torch.nn.init.constant_(layer.bias, bias_const)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity, a=a_leaky_relu, mode='fan_in')
                torch.nn.init.constant_(layer.bias, bias_const)
            elif init_type == "default":
                pass
            else:
                raise ValueError(f"Unsupported init_type: {init_type}")

        if add_weight_norm:
            layer = weight_norm(layer)

        # Apply Layer Normalization or Batch Normalization
        processed_layer=[]
        processed_layer.append(layer)
        if add_layer_norm:
            processed_layer.append(LayerNorm(layer.out_features, device=device, dtype=dtype,
                                                elementwise_affine=True, bias=True))
        elif add_batch_norm:
            processed_layer.append(BatchNorm1d(layer.out_features, device=device, dtype=dtype,
                                                    eps=1e-05, 
                                                    affine=True, # y(k) = γ(k)̂ x(k) + β(k). to make sure that the transformation inserted in the network can represent the identity transform
                                                    momentum=0.1, #used if track_running_stats (stats smoothing coeff)
                                                    track_running_stats=True))
            
        return processed_layer

def llayer_reset(layer,
        init_type=None,
        nonlinearity="leaky_relu",
        a_leaky_relu: float = 0.01,
        bias_const=0.0,
        uniform_biases: bool = False,
        add_weight_norm: bool = False,
        orth_init_gain: float = 1.0):
    """
    Resets the parameters of a given layer based on the specified initialization type.

    Args:
        layer (nn.Module): The layer to reset.
        init_type (str): The type of initialization to use.
        nonlinearity (str): The non-linear function used.
        a_leaky_relu (float): The negative slope for leaky ReLU.
        bias_const (float): The constant value to initialize biases.
        uniform_biases (bool): Whether to use uniform initialization for biases.
        add_weight_norm (bool): Whether to add weight normalization.
        orth_init_gain (float): Gain for orthogonal initialization.
    """

    # Apply weight initialization based on the init_type argument
    if init_type is not None:
        if init_type == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=orth_init_gain)
            torch.nn.init.constant_(layer.bias, bias_const)
        elif init_type == "uniform":
            k = 1 / layer.in_features
            bound = math.sqrt(k)
            torch.nn.init.uniform_(layer.weight, a=-bound, b=bound)
            if not uniform_biases:
                torch.nn.init.constant_(layer.bias, bias_const)
            else:
                torch.nn.init.uniform_(layer.bias, a=-bound, b=bound)
        elif init_type == "kaiming_normal":
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity, a=a_leaky_relu, mode='fan_in')
            torch.nn.init.constant_(layer.bias, bias_const)
        elif init_type == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity, a=a_leaky_relu, mode='fan_in')
            torch.nn.init.constant_(layer.bias, bias_const)
        elif init_type == "default":
            pass
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

    if add_weight_norm:
        layer = weight_norm(layer)

    return layer