import torch
import math

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

def get_pitch(quat: torch.Tensor) -> torch.Tensor:

    cos_theta = torch.clamp(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1.0, 1.0)
    
    theta = torch.acos(cos_theta)
    
    return theta.view(-1, 1)

def check_capsize(quat: torch.Tensor, 
        max_angle: float, 
        output_t: torch.Tensor=None):
    
    # Populate the output tensor with True where cos(theta) exceeds cos(max_angle)
    if output_t is not None:
        output_t[:, 0] = torch.clamp(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1.0, 1.0) < math.cos(max_angle)
        return None
    else:
        return torch.clamp(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1.0, 1.0) < math.cos(max_angle)

if __name__ == "__main__":  
    n_envs = 2

    quats= torch.zeros((n_envs,4), dtype=torch.float32, device="cpu")
    is_capsized=torch.zeros((n_envs,1), dtype=torch.bool, device="cpu")
    # 45 degs pich
    quats[:, 0] = 0
    quats[:, 1] = 0.3826834
    quats[:, 2] = 0
    quats[:, 3] = 0.9238795
    theta_check=45*math.pi/180.0 # degs 

    print("###############")
    print(get_pitch(quats)*180/math.pi)
    check_capsize(quat=quats, max_angle=theta_check, output_t=is_capsized)
    print(is_capsized)
    
    