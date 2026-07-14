"""Hybrid SAC agent: tanh-squashed Gaussian over the continuous action dims, Bernoulli over the
binary ones.

Motivation
----------
The environment thresholds the contact/swing flag dims (`TwistTrackingEnv._set_rhc_refs`):

    flag_i = 1[a_i > step_thresh],   step_thresh = 0,   a_i in [-1, +1]

so the transition depends on the flags only through `sign(a_i)`. Under the legacy all-Gaussian
actor, `tanh` is strictly monotone with `tanh(0) = 0`, hence

    P(flag_i = 1 | s) = P(x_i > 0) = Phi(mu_i / sigma_i),      x_i ~ N(mu_i, sigma_i)

i.e. the induced policy over MDP actions is *already* Bernoulli. What the legacy formulation gets
wrong is not the policy class but everything around it:

  1. the replay buffer, and therefore the critic's input, stores the raw continuous `a_i` rather
     than the flag. The true Q is piecewise-constant on {a_i < 0} and {a_i > 0}, so dQ/da_i is
     meaningless within each half-interval, yet the actor follows it;
  2. the "discrete entropy" it regularizes is the differential entropy of the squashed Gaussian on
     those dims (a function of sigma_i), not the Bernoulli entropy H_b(Phi(mu_i/sigma_i)) that
     actually governs exploration of the binary decision.

This module fixes both by emitting hard -1/+1 on the binary dims and carrying an analytic Bernoulli
entropy. See `~/Desktop/prompts/ibrido/hybrid_sac.md` for the full derivation and plan.

Old `SACAgent` checkpoints will not load into `HybridSACAgent`: the actor head widths change from
`actions_dim` to `n_cont` / `n_binary`. This is expected and accepted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

from aug_mpc.agents.sactor_critic.sac import SACAgent
from aug_mpc.utils.nn.layer_utils import llayer_init

from typing import Dict, List, Tuple

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

# Gauss-Hermite nodes/weights, for the deterministic entropy quadrature below. Computed once.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(64)

def tanh_gaussian_entropy(std: float):
    """Differential entropy (nats, per dim) of a zero-mean tanh-squashed Gaussian with pre-tanh std.

        a = tanh(x),   x ~ N(0, std)

        H(a) = H(x) + E_x[ log |da/dx| ]
             = 0.5*log(2*pi*e*std^2) + E_x[ log(1 - tanh^2 x) ]

    The expectation is evaluated by 64-point Gauss-Hermite quadrature, so this is deterministic (no
    RNG in the actor's init) and accurate to ~1e-3 against a 4e5-sample Monte Carlo estimate.

    IMPORTANT -- H is NOT monotone in std. It rises, peaks at std = 0.8744 with H = 0.6836 nats
    (= 0.9863 * log 2), and then FALLS: as std grows the tanh saturates and the policy collapses
    onto the two points {-1, +1}, so the differential entropy heads to -inf. Consequently:

      * the uniform bound log 2 is a strict supremum for this family, never attained;
      * inverting H requires picking a branch (see `std_for_entropy_frac`, which takes the rising one).
    """
    x = math.sqrt(2.0)*std*_GH_NODES
    # log(1 - tanh^2 x) = 2*(log 2 - x - softplus(-2x)), in the numerically stable form
    log_sech2 = 2.0*(math.log(2.0) - x - np.logaddexp(0.0, -2.0*x))
    jacobian_corr = float((_GH_WEIGHTS*log_sech2).sum()/math.sqrt(math.pi))
    return 0.5*math.log(2.0*math.pi*math.e*std*std) + jacobian_corr

# std at which tanh_gaussian_entropy peaks, and the peak value, as a fraction of log 2
STD_AT_MAX_ENTROPY = 0.8744
MAX_ENTROPY_FRAC = 0.9863

def std_for_entropy_frac(frac: float, tol: float = 1e-9):
    """Smallest pre-tanh std whose squashed-Gaussian entropy is `frac * log 2` nats per dim.

    This is what lets the actor be initialized *at* its entropy target instead of climbing to it:
    pass the algorithm's `target_H_cont_frac` and the continuous branch starts in equilibrium, so
    alpha_cont has nothing to correct and its initial value stays meaningful.

    Bisects on the rising branch of H (std in (0, STD_AT_MAX_ENTROPY]), which is the sensible one:
    the falling branch would reach the same entropy with a saturated, near-bimodal policy.

    Returns None if `frac` exceeds MAX_ENTROPY_FRAC, i.e. is unattainable by this family.
    """
    target = frac*math.log(2.0)
    if frac > MAX_ENTROPY_FRAC:
        return None
    lo, hi = 1e-4, STD_AT_MAX_ENTROPY
    while hi-lo > tol:
        mid = math.sqrt(lo*hi) # geometric bisection: H is roughly linear in log(std) at small std
        if tanh_gaussian_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo*hi)

def sample_st_bernoulli(logits: torch.Tensor,
        tau: float = 0.7,
        eps: float = 1e-6):
    """Straight-through Gumbel-Sigmoid sampling of independent Bernoulli variables.

    Draw `u ~ U(0,1)` and form the standard logistic noise

        g = log(u) - log(1 - u),        P(g <= z) = sigmoid(z)

    The soft relaxation and its hard threshold are

        y_soft = sigmoid((logits + g) / tau)
        b_hard = 1[y_soft > 0.5] = 1[logits + g > 0]

    the second equality holding because `sigmoid` is strictly increasing with `sigmoid(0) = 0.5`
    and `tau > 0`. Therefore

        P(b_hard = 1) = P(g > -logits) = 1 - sigmoid(-logits) = sigmoid(logits)

    exactly, *for every* tau. `tau` never enters the sampled distribution; it only shapes the
    backward surrogate. The straight-through estimator

        b_st = b_hard + (y_soft - y_soft.detach())

    is numerically `b_hard` in the forward pass and has `d b_st / d logits = sigmoid'((logits+g)/tau) / tau`
    in the backward pass.

    That surrogate is *biased*: the exact gradient of the Q-term would be

        d/d logits_i  E_b[Q(s, b)] = (Q(s, b_i=1) - Q(s, b_i=0)) * sigmoid'(logits_i)

    The two agree in sign (both surrogate factors are strictly positive) but not in magnitude. For
    small `n_binary` the exact expectation can be enumerated instead; see the `exact` grad mode.

    Also returned, both analytic (no sampling variance):

        log pi(b)  = sum_i [ b_i log p_i + (1 - b_i) log(1 - p_i) ]
        H(pi)_i    = -p_i log p_i - (1 - p_i) log(1 - p_i)   in [0, log 2]

    Args:
        logits: [B, n_binary] Bernoulli logits, `p = sigmoid(logits)`.
        tau: Gumbel-Sigmoid temperature. Backward-only; see above.
        eps: clamp on `u` to keep `log(u)` and `log1p(-u)` finite.

    Returns:
        b_st: [B, n_binary] in {0, 1} exactly, differentiable w.r.t. `logits`.
        b_hard: [B, n_binary] in {0, 1}, detached (an exact Bernoulli(sigmoid(logits)) sample).
        logp_sample: [B, 1] log-probability of the drawn sample.
        entropy_vec: [B, n_binary] analytic per-flag Bernoulli entropy, in nats.
    """

    u = torch.rand_like(logits).clamp(eps, 1.0-eps)
    logistic_noise = torch.log(u) - torch.log1p(-u)

    y_soft = torch.sigmoid((logits+logistic_noise)/tau)
    b_hard = (y_soft > 0.5).to(y_soft.dtype)

    # hard forward, soft backward. The parentheses are load-bearing: (y_soft - y_soft.detach()) is
    # bitwise zero, whereas the usual left-to-right (b_hard + y_soft) - y_soft.detach() rounds in
    # fp32 and leaks values like 0.9999998 into the action. torch.nn.functional.gumbel_softmax has
    # the same defect; we cannot afford it, since -1/+1 exactness is the entire point of this actor.
    b_st = b_hard + (y_soft - y_soft.detach())

    bern = torch.distributions.Bernoulli(logits=logits)
    logp_sample = bern.log_prob(b_hard).sum(dim=-1, keepdim=True)
    entropy_vec = bern.entropy() # analytic, [B, n_binary]

    return b_st, b_hard, logp_sample, entropy_vec

class HybridActor(nn.Module):
    """Shared trunk with two heads: a squashed-Gaussian head over the continuous action dims and a
    Bernoulli (straight-through Gumbel-Sigmoid) head over the binary ones.

    The two heads are conditionally independent given the observation, so the joint policy factors:

        pi(a | s) = pi_cont(a_cont | s) * prod_i Bernoulli(b_i ; p_i(s))

    and therefore its entropy is the sum of the two entropies. That factorization is what lets the
    algorithm keep one temperature per branch, and it is what makes the exact marginalization over
    the `2^n_binary` flag combinations tractable (one continuous sample, `2^n_binary` critic evals).

    The flat action tensor handed to the critic and to `env.step()` is unchanged in layout: the
    continuous and binary dims are scattered back into their original positions.
    """

    def __init__(self,
        obs_dim: int,
        actions_dim: int,
        continuous_mask: torch.Tensor, # [actions_dim] bool, from env.is_action_continuous()
        actions_ub: List[float] = None,
        actions_lb: List[float] = None,
        device: str = "cuda",
        dtype = torch.float32,
        layer_width: int = 256,
        n_hidden_layers: int = 2,
        add_weight_norm: bool = False,
        add_layer_norm: bool = False,
        add_batch_norm: bool = False,
        gumbel_tau: float = 0.7,
        st_eps: float = 1e-6,
        init_std: float = None,
        target_H_cont_frac: float = -0.72):

        super().__init__()

        self._lrelu_slope=0.01

        # init_std=None (the default) derives the std from the entropy target, so the continuous
        # branch starts *at* its target rather than climbing to it. Pass a float to pin it instead.
        self._target_H_cont_frac=target_H_cont_frac
        if init_std is None:
            init_std = std_for_entropy_frac(target_H_cont_frac)
            if init_std is None:
                Journal.log(self.__class__.__name__,
                    "__init__",
                    f"target_H_cont_frac={target_H_cont_frac} exceeds the maximum attainable "
                    f"{MAX_ENTROPY_FRAC:.4f} for a tanh-squashed Gaussian (H peaks at std="
                    f"{STD_AT_MAX_ENTROPY}, never reaching the uniform bound log 2). Falling back "
                    f"to the peak-entropy std.",
                    LogType.WARN)
                init_std = STD_AT_MAX_ENTROPY
        self._init_std=init_std

        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim

        self._gumbel_tau = gumbel_tau
        self._st_eps = st_eps

        self._first_hidden_layer_width=self._obs_dim # first layer fully connected and of same dim

        # action masks. continuous_idxs and binary_idxs partition range(actions_dim); they are
        # registered as buffers so they follow .to(device) and land in the state dict.
        if continuous_mask is None:
            continuous_mask = torch.ones((actions_dim, ), dtype=torch.bool)
        continuous_mask = continuous_mask.flatten().to(device=self._torch_device, dtype=torch.bool)
        if continuous_mask.numel() != actions_dim:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Continuous mask length should be equal to {actions_dim}, but got {continuous_mask.numel()}",
                LogType.EXCEP,
                throw_when_excep = True)

        continuous_idxs = torch.where(continuous_mask)[0].to(torch.long)
        binary_idxs = torch.where(~continuous_mask)[0].to(torch.long)
        self.register_buffer("continuous_idxs", continuous_idxs)
        self.register_buffer("binary_idxs", binary_idxs)

        self._n_cont = continuous_idxs.numel()
        self._n_binary = binary_idxs.numel()

        # Action scale and bias: a = tanh(x) * scale + bias maps R -> [lb, ub]
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

        self._validate_binary_bounds()

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

        # gathered views over the continuous dims: the heads are sized n_cont, so the scale/bias
        # they need is the restriction of the full-width buffers to continuous_idxs
        self.register_buffer("cont_scale", action_scale[continuous_idxs].clone())
        self.register_buffer("cont_bias", actions_bias[continuous_idxs].clone())
        self.register_buffer("log_cont_scale", torch.log(action_scale[continuous_idxs].clone()+1e-6))

        # Network configuration
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

        # bias of the log_std head, chosen so the actor actually emits `init_std` at init.
        #
        # forward() squashes the raw head output: log_std = MIN + 0.5*(MAX-MIN)*(tanh(raw)+1).
        # The legacy Actor sets bias = log(0.5), evidently intending std = 0.5, but that value is
        # itself squashed: tanh(-0.693) = -0.6 -> log_std = -3.6 -> std = 0.027, i.e. a near-
        # deterministic policy with H_cont = -2.18 nats/dim against a target of +0.5. Invert the
        # squash instead, so the intent is honoured:
        #
        #   raw = atanh( 2*(log(init_std) - MIN)/(MAX - MIN) - 1 )
        self._logstd_bias = self._raw_logstd_for(self._init_std)

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

        # Shared trunk h(s)
        self._fc12 = nn.Sequential(*layers)

        # Continuous head: mean and log_std over the continuous dims only
        out_fc_mean=llayer_init(nn.Linear(layer_width, self._n_cont),
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
        out_fc_logstd= llayer_init(nn.Linear(layer_width, self._n_cont),
                        init_type="uniform",
                        uniform_biases=False,
                        bias_const=self._logstd_bias, # see _raw_logstd_for()
                        scale_weight=1e-3, # scaling (output layer)
                        scale_bias=1.0,
                        device=self._torch_device,
                        dtype=self._torch_dtype,
                        add_weight_norm=False,
                        add_layer_norm=False,
                        add_batch_norm=False,
                        )
        self.fc_logstd = nn.Sequential(*out_fc_logstd)

        # Binary head: Bernoulli logits over the binary dims only. Zero bias and a small weight
        # scale put p ~ sigmoid(0) = 0.5 at init, i.e. H_disc ~ n_binary * log 2 (maximal entropy):
        # the policy starts out maximally undecided about every contact flag.
        out_fc_logits= llayer_init(nn.Linear(layer_width, self._n_binary),
                        init_type="uniform",
                        uniform_biases=False,
                        bias_const=0.0,
                        scale_weight=1e-3, # scaling (output layer)
                        scale_bias=1.0,
                        device=self._torch_device,
                        dtype=self._torch_dtype,
                        add_weight_norm=False,
                        add_layer_norm=False,
                        add_batch_norm=False,
                        )
        self.fc_logits = nn.Sequential(*out_fc_logits)

        # Move all components to the specified device and dtype
        self._fc12.to(device=self._torch_device, dtype=self._torch_dtype)
        self.fc_mean.to(device=self._torch_device, dtype=self._torch_dtype)
        self.fc_logstd.to(device=self._torch_device, dtype=self._torch_dtype)
        self.fc_logits.to(device=self._torch_device, dtype=self._torch_dtype)

        h_init = tanh_gaussian_entropy(self._init_std)
        Journal.log(self.__class__.__name__,
            "__init__",
            f"Created hybrid actor with {self._n_cont} continuous and {self._n_binary} binary "
            f"action dims (of {self._actions_dim} total).\n"
            f"Binary dims: {self.binary_idxs.tolist()}\n"
            f"Max entropy: disc {self.max_entropy_disc():.4f} nats, cont {self.max_entropy_cont():.4f} nats\n"
            f"Actor init std: {self._init_std:.4f} -> H_cont {h_init*self._n_cont:.4f} nats "
            f"({h_init/math.log(2.0):.3f} of max, target {self._target_H_cont_frac:.3f})\n"
            f"Binary logits init at 0 -> H_disc {self.max_entropy_disc():.4f} nats (1.000 of max)\n"
            f"Gumbel-Sigmoid tau: {self._gumbel_tau}",
            LogType.INFO)

        print("Hybrid actor architecture")
        print(self._fc12)
        print(self.fc_mean)
        print(self.fc_logstd)
        print(self.fc_logits)

    def _validate_binary_bounds(self):
        """Binary dims must be declared as [-1, +1] by the env.

        The -1/+1 encoding `a_i = 2 b_i - 1` presumes `scale = 1`, `bias = 0` on those dims. It also
        makes the env's `step_thresh` irrelevant: any threshold strictly inside (-1, +1) recovers
        `b_i` from `sign(a_i)`. A [0, 1] encoding (`use_prob_based_stepping`) would silently break
        both, so refuse it here rather than emit out-of-range commands.
        """
        if self._n_binary == 0:
            return
        ub = self._actions_ub[self.binary_idxs]
        lb = self._actions_lb[self.binary_idxs]
        ok = torch.allclose(ub, torch.ones_like(ub)) and \
            torch.allclose(lb, -torch.ones_like(lb))
        if not ok:
            Journal.log(self.__class__.__name__,
                "_validate_binary_bounds",
                f"Binary action dims must have bounds [-1, +1] for the -1/+1 encoding, but got "
                f"lb={lb.tolist()}, ub={ub.tolist()} on dims {self.binary_idxs.tolist()}",
                LogType.EXCEP,
                throw_when_excep = True)

    def _raw_logstd_for(self, std: float):
        """Pre-squash head bias that makes `forward()` emit `log(std)`.

        Inverts `log_std = MIN + 0.5*(MAX-MIN)*(tanh(raw)+1)`.
        """
        target_log_std = math.log(std)
        if not (self.LOG_STD_MIN < target_log_std < self.LOG_STD_MAX):
            Journal.log(self.__class__.__name__,
                "_raw_logstd_for",
                f"init_std={std} implies log_std={target_log_std:.3f}, outside the squash range "
                f"({self.LOG_STD_MIN}, {self.LOG_STD_MAX})",
                LogType.EXCEP,
                throw_when_excep = True)
        t = 2.0*(target_log_std-self.LOG_STD_MIN)/(self.LOG_STD_MAX-self.LOG_STD_MIN) - 1.0
        return math.atanh(t)

    def n_cont(self):
        return self._n_cont

    def n_binary(self):
        return self._n_binary

    def init_std(self):
        return self._init_std

    def max_entropy_disc(self):
        """Upper bound of the discrete entropy: `n_binary * log 2` nats, attained at `p_i = 0.5`.

        Each flag is a Bernoulli, whose entropy is maximal (log 2) at p = 0.5.
        """
        return self._n_binary*math.log(2.0)

    def max_entropy_cont(self):
        """Upper bound of the continuous *differential* entropy: `n_cont * log 2` nats.

        The continuous actions are tanh-squashed onto [-1, 1], a support of Lebesgue measure 2 per
        dim, and the maximum-entropy distribution on a bounded support is the uniform one, with
        differential entropy log(2) per dim.

        Note the pleasing coincidence: both branches max out at `N * log 2`, so a single
        "fraction of the attainable maximum" convention covers them both.

        Unlike the discrete entropy, this is only an upper bound: a differential entropy is not
        bounded below and goes to -inf as the policy sharpens. So the measured H_cont may be
        negative (it is, at a near-deterministic init), while the *target* is always positive.
        """
        return self._n_cont*math.log(2.0)

    def gumbel_tau(self):
        return self._gumbel_tau

    def set_gumbel_tau(self, tau: float):
        self._gumbel_tau = tau

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """Returns `(mean, log_std, logits)`; `mean`/`log_std` are [B, n_cont], `logits` [B, n_binary]."""
        x = self._fc12(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        # squash log_std into [LOG_STD_MIN, LOG_STD_MAX] rather than hard-clamping, so the gradient
        # never dies at the bounds (SpinUp / Denis Yarats)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        logits = self.fc_logits(x)
        return mean, log_std, logits

    def _pack(self, a_cont, b_pm1):
        """Scatter the two heads back into the env's flat action layout, [B, actions_dim].

        `continuous_idxs` and `binary_idxs` partition the action space, so every column is written.
        """
        action = torch.zeros((a_cont.shape[0], self._actions_dim),
                device=a_cont.device, dtype=a_cont.dtype)
        action[:, self.continuous_idxs] = a_cont
        action[:, self.binary_idxs] = b_pm1
        return action

    def get_action(self, x) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Sample a hybrid action and return `(action, log_info, deterministic_action)`.

        Continuous branch (standard SAC, reparametrized):

            x_t ~ N(mean, std),   a_cont = tanh(x_t) * scale + bias

        with the change-of-variables correction

            log pi(a_cont) = log N(x_t; mean, std) - sum_i log( scale_i * (1 - tanh^2(x_t_i)) )

        The Jacobian term is evaluated in its numerically stable form. Using
        `1 - tanh^2(x) = sech^2(x) = 4 e^{-2x} / (1 + e^{-2x})^2`,

            log(1 - tanh^2(x)) = log 4 - 2x - 2 log(1 + e^{-2x})
                               = 2 * ( log 2 - x - softplus(-2x) )

        which avoids the catastrophic cancellation of `log(1 - y^2)` as `|y| -> 1`.

        Binary branch:

            b ~ Bernoulli(sigmoid(logits))   via straight-through Gumbel-Sigmoid
            a_bin = 2 * b - 1                in {-1, +1} exactly

        The critic therefore only ever sees action values the environment can distinguish, and
        gradients still reach `logits` through the straight-through path.

        `log_info` keys (all detachable, shapes for a batch of B):
            logp_cont         [B, 1]         sum_i log pi(a_cont_i)
            logp_cont_vec     [B, n_cont]    per-dim log-prob, for group diagnostics
            logp_disc_sample  [B, 1]         log pi(b) of the drawn sample
            entropy_disc      [B, 1]         analytic sum_i H(p_i), in [0, n_binary*log 2]
            entropy_disc_vec  [B, n_binary]  analytic per-flag entropy
            binary_probs      [B, n_binary]  p_i = sigmoid(logits_i), detached
            binary_hard       [B, n_binary]  the drawn sample, in {0, 1}
            binary_logits     [B, n_binary]  needed by the exact-marginalization grad mode
        """

        mean, log_std, logits = self(x)

        # --- continuous dims: squashed Gaussian, reparametrized ---
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick: keeps the comp. graph to the critic
        y_t = torch.tanh(x_t)
        a_cont = y_t * self.cont_scale + self.cont_bias

        # log(1 - tanh^2(x_t)), stable form (see docstring)
        log_det_tanh = 2.0 * (math.log(2.0) - x_t - F.softplus(-2.0*x_t))
        logp_cont_vec = normal.log_prob(x_t) - log_det_tanh - self.log_cont_scale
        logp_cont = logp_cont_vec.sum(dim=-1, keepdim=True)

        # --- binary dims: Bernoulli with straight-through Gumbel-Sigmoid ---
        b_st, b_hard, logp_disc_sample, entropy_disc_vec = sample_st_bernoulli(logits,
                                                            tau=self._gumbel_tau,
                                                            eps=self._st_eps)
        entropy_disc = entropy_disc_vec.sum(dim=-1, keepdim=True)

        action = self._pack(a_cont=a_cont, b_pm1=2.0*b_st-1.0)

        # --- deterministic action (evaluation): tanh of the mean, MAP of the Bernoulli ---
        # no enumeration needed: the flags are independent, so the joint mode is the per-flag mode
        with torch.no_grad():
            binary_probs = torch.sigmoid(logits)
            a_cont_det = torch.tanh(mean) * self.cont_scale + self.cont_bias
            b_det = 2.0*(binary_probs > 0.5).to(self._torch_dtype)-1.0
            action_det = self._pack(a_cont=a_cont_det, b_pm1=b_det)

        log_info = {
            "logp_cont": logp_cont,                 # [B, 1]
            "logp_cont_vec": logp_cont_vec,         # [B, n_cont]
            "logp_disc_sample": logp_disc_sample,   # [B, 1]
            "entropy_disc": entropy_disc,           # [B, 1], analytic
            "entropy_disc_vec": entropy_disc_vec,   # [B, n_binary], analytic
            "binary_probs": binary_probs,           # [B, n_binary]
            "binary_hard": b_hard,                  # [B, n_binary], in {0, 1}
            "binary_logits": logits,                # [B, n_binary], for the exact-marginal path
        }

        return action, log_info, action_det

    def remove_scaling(self, a):
        """Inverse of the affine action scaling: maps [lb, ub] back to [-1, 1]."""
        return (a - self.action_bias)/self.action_scale

class HybridSACAgent(SACAgent):
    """`SACAgent` with a `HybridActor` in place of the all-Gaussian `Actor`.

    Everything else is inherited unchanged: the twin critics and their targets, the running
    observation normalizer, the action rescaling applied before the critic, and the
    `(action, log_info, deterministic_action)` return shape of `get_action()`.

    Only `_create_actor()` is overridden. `SACAgent._build_nets()` calls that hook, so both the
    initial construction and `reset()` pick up the hybrid actor.

    Note: `state_dict()` is not interchangeable with `SACAgent`'s. The actor's `fc_mean` /
    `fc_logstd` are `n_cont` wide instead of `actions_dim`, and `fc_logits` is new. Loading a
    legacy checkpoint here raises, by design.
    """

    def __init__(self,
            obs_dim: int,
            actions_dim: int,
            continuous_mask: torch.Tensor, # [actions_dim] bool, from env.is_action_continuous()
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
            compression_ratio:float=-1.0,
            layer_width_actor:int=256,
            n_hidden_layers_actor:int=2,
            layer_width_critic:int=512,
            n_hidden_layers_critic:int=4,
            torch_compile: bool = False,
            add_weight_norm: bool = False,
            add_layer_norm: bool = False,
            add_batch_norm: bool = False,
            gumbel_tau: float = 0.7,
            init_std: float = None,
            target_H_cont_frac: float = -0.72):

        # these must exist before SACAgent.__init__ runs, since it calls _build_nets() -> _create_actor()
        if continuous_mask is None:
            continuous_mask = torch.ones((actions_dim, ), dtype=torch.bool)
        self._continuous_mask = continuous_mask.flatten().to(torch.bool)
        self._gumbel_tau = gumbel_tau
        self._init_std = init_std # None -> derived from target_H_cont_frac by HybridActor
        self._target_H_cont_frac = target_H_cont_frac
        self._n_cont = int(self._continuous_mask.sum().item())
        self._n_binary = int((~self._continuous_mask).sum().item())

        SACAgent.__init__(self,
            obs_dim=obs_dim,
            actions_dim=actions_dim,
            obs_ub=obs_ub,
            obs_lb=obs_lb,
            actions_ub=actions_ub,
            actions_lb=actions_lb,
            rescale_obs=rescale_obs,
            norm_obs=norm_obs,
            use_action_rescale_for_critic=use_action_rescale_for_critic,
            device=device,
            dtype=dtype,
            is_eval=is_eval,
            load_qf=load_qf,
            epsilon=epsilon,
            debug=debug,
            compression_ratio=compression_ratio,
            layer_width_actor=layer_width_actor,
            n_hidden_layers_actor=n_hidden_layers_actor,
            layer_width_critic=layer_width_critic,
            n_hidden_layers_critic=n_hidden_layers_critic,
            torch_compile=torch_compile,
            add_weight_norm=add_weight_norm,
            add_layer_norm=add_layer_norm,
            add_batch_norm=add_batch_norm)

    def _create_actor(self):
        """Overrides `SACAgent._create_actor()`; called from `_build_nets()` and hence `reset()`."""
        return HybridActor(obs_dim=self._obs_dim,
            actions_dim=self._actions_dim,
            continuous_mask=self._continuous_mask,
            actions_ub=self._actions_ub,
            actions_lb=self._actions_lb,
            device=self._torch_device,
            dtype=self._torch_dtype,
            layer_width=self._layer_width_actor,
            n_hidden_layers=self._n_hidden_layers_actor,
            add_weight_norm=self._add_weight_norm,
            add_layer_norm=self._add_layer_norm,
            add_batch_norm=self._add_batch_norm,
            gumbel_tau=self._gumbel_tau,
            init_std=self._init_std,
            target_H_cont_frac=self._target_H_cont_frac)

    # the counts are cached on the agent rather than delegated to self.actor, which may be wrapped
    # into an OptimizedModule by torch.compile

    def n_cont(self):
        return self._n_cont

    def n_binary(self):
        return self._n_binary

    def max_entropy_disc(self):
        """Upper bound of the discrete entropy: `n_binary * log 2` nats (fair coin on each flag)."""
        return self._n_binary*math.log(2.0)

    def max_entropy_cont(self):
        """Upper bound of the continuous differential entropy: `n_cont * log 2` nats (uniform on
        [-1, 1]^n_cont). NOT a lower bound: the differential entropy is unbounded below."""
        return self._n_cont*math.log(2.0)

    def init_std(self):
        # resolved by HybridActor when it was passed as None
        return self.actor.init_std()

    def continuous_mask(self):
        return self._continuous_mask

    def gumbel_tau(self):
        return self._gumbel_tau

    def set_gumbel_tau(self, tau: float):
        self._gumbel_tau = tau
        self.actor.set_gumbel_tau(tau)
