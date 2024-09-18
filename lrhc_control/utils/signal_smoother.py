from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

import torch

from typing import List

class ExponentialSignalSmoother():
    
    # class for performing an exponential signal smoothing on a 
    # vectorized signal s
    # s_tilde(t) = (1-alpha)*s(t)+alpha*s_thilde(t-1)
    # given an horizon of h samples, 
    # s_tilde(t) = (1-alpha)*sum{k=0}^{h}{alpha*s(t-k)}+alpha^{h}*s_tilde(t-h)
    # where usually s_tilde(t-h) = s(t-h)
    # This means that the oldest signal will have influence alpha^{h} on the current smoothed signal.
    # Given a target influence percentage 0<p_t<1, a given horizon length in [s] T and update interval for
    # the signal in [s] dt, the smoothing coefficient alpha can be chosen as 
    # alpha=e^{dt/T*ln(p_t)}
    # For example, if T=1s, dt=0.03 and p_t=0.2 (i.e. 20%) -> alpha=0.953
    def __init__(self,
        signal: torch.Tensor, 
        update_dt: float, # [s]
        smoothing_horizon: float, # [s]
        target_smoothing: float, # percentage 0<pt<1
        name: str = "SignalSmoother",
        debug: bool = False,
        dtype: torch.dtype = torch.float32,
        use_gpu: bool = False):

        self._n_envs=signal.shape[0]
        self._signal_dim=signal.shape[1]

        self._name=name
        self._debug=debug
        self._dtype=dtype
        self._use_gpu=use_gpu

        self._update_dt=update_dt
        self._T=smoothing_horizon
        self._pt=target_smoothing
        self._alpha=0.0# no smoothing by default

    def _init_data(self):
        a=1
    
    def _check_new_data(self,new_data):
        self._check_sizes(new_data=new_data)
        self._check_finite(new_data=new_data)

    def _check_sizes(self,new_data):
        if (not new_data.shape[0] == self._n_envs) or \
            (not new_data.shape[1] == self._signal_dim):
            exception = f"Provided signal tensor shape {new_data.shape[0]}, {new_data.shape[1]}" + \
                f" does not match {self._n_envs}, {self._signal_dim}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
    
    def _check_finite(self,new_data):
        if (not torch.isfinite(new_data).all().item()):
            print(new_data)
            exception = f"Found non finite elements in provided data!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

    def reset_all(self):

    def reset(self,
        to_be_reset: torch.Tensor):

        a=1
    
    def update(self, 
        new_signal: torch.Tensor):
        a=1
    
    def horizon(self):
        return self._T
    
    def dt(self):
        return self._update_dt
    
    def 