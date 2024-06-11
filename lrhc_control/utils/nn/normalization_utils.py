from typing import Tuple, Union

import torch

class RunningMeanStd(object):
    def __init__(self, tensor_size, torch_device, dtype, epsilon: float = 1e-8):
        """
        Torch version of the same from stable_baselines3 (credits @c-rizz; Carlo Rizzardo)

        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        
        self._epsilon = epsilon
        self.mean = torch.zeros(tensor_size, device=torch_device, dtype=dtype)
        self.var = torch.ones(tensor_size, device=torch_device, dtype=dtype)
        self.count = torch.tensor(epsilon, device=torch_device, dtype=torch.float64)

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(tensor_size=self.mean.size(),
                                    torch_device=self.mean.device,
                                    dtype = self.mean.dtype,
                                    epsilon=self._epsilon)
        # use copy_() to avoid breaking buffer registration
        new_object.mean.copy_(self.mean.detach())
        new_object.var.copy_(self.var.detach())  
        new_object.count.copy_(self.count.detach())
        return new_object
    
    def copy_(self, src) -> None:
        """
        :return: Return a copy of the current object.
        """
        # use copy_() to avoid breaking buffer registration
        self.mean.copy_(src.mean.detach())
        self.var.copy_(src.var.detach())  
        self.count.copy_(src.count.detach())

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine witorch.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, x) -> None:
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_size = x.size()[0]
        self.update_from_moments(batch_mean, batch_var, batch_size)

    def update_from_moments(self, batch_mean, batch_var, batch_size: Union[int, float]) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_size

        new_mean = self.mean + delta * batch_size / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_size / (self.count + batch_size)
        new_var = m_2 / (self.count + batch_size)

        new_count = batch_size + self.count
        
        # skip if there are infs and nans
        if torch.all(torch.isfinite(new_mean)) and torch.all(torch.isfinite(new_var)) and torch.all(torch.isfinite(new_count)):
            # use copy_() to avoid breaking buffer registration
            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.copy_(new_count)
        else:
            print(f"Detected nan/inf in mean/std tracker, skipping")

class RunningNormalizer(torch.nn.Module):
    def __init__(self, shape : Tuple[int,...], dtype, device, epsilon : float = 1e-8, freeze_stats: bool=True):
        super().__init__()
        self._freeze_stats = freeze_stats
        self.register_buffer("_epsilon", torch.tensor(epsilon, device = device))
        self._running_stats = RunningMeanStd(shape, torch_device=device, dtype=dtype)
        self.register_buffer("vec_running_mean",  self._running_stats.mean)
        self.register_buffer("vec_running_var",   self._running_stats.var)
        self.register_buffer("vec_running_count", self._running_stats.count)

    def forward(self, x):
        if not (self.training or self._freeze_stats):
            self._running_stats.update(x)
        return (x - self._running_stats.mean)/(torch.sqrt(self._running_stats.var)+self._epsilon)
