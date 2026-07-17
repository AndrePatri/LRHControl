"""Hybrid SAC: soft actor-critic with a factored continuous/binary policy.

Subclasses the legacy `SAC` and overrides a small set of hooks. `_update_policy()`,
`_update_validation_losses()` and `_collect_eval_transition()` are inherited unchanged: they route
their entropy arithmetic through `_entropy_penalty()` / `_alpha_losses()` / `_policy_entropies()`,
which is all that differs between the two formulations.

What changes with respect to `SAC`:

  * the agent is a `HybridSACAgent`, whose actor emits hard -1/+1 on the binary dims;
  * the discrete entropy is the analytic Bernoulli entropy, bounded by `n_binary * log 2`, rather
    than the differential entropy of a squashed Gaussian sliced by the discrete mask;
  * entropy targets are expressed as a SIGNED fraction of the attainable maximum (N*log2 nats) for
    both branches, instead of the legacy per-action nat values. frac_disc is in [0,1] (a Bernoulli
    entropy cannot be negative); frac_cont may be negative, and for this task it is: the legacy
    ENTROPY_CONT_START=-0.5 is a target of -0.5 nats/dim, i.e. frac_cont = -0.72;
  * the alpha loss uses `log_alpha` rather than `exp(log_alpha)` as the multiplier;
  * warmstart actions and exploration perturbations keep the binary dims on {-1, +1}.

See `~/Desktop/prompts/ibrido/hybrid_sac.md` for the derivation.
"""

from aug_mpc.training_algs.sac.sac import SAC
from aug_mpc.agents.sactor_critic.hybrid_sac import HybridSACAgent

import torch
import torch.nn.functional as F
import torch.optim as optim

import math
import os

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

class HybridSAC(SAC):

    def __init__(self,
            env,
            debug = False,
            remote_db = False,
            seed: int = 1):

        SAC.__init__(self, env=env,
                    debug=debug,
                    remote_db=remote_db,
                    seed=seed)

        # defaults. _create_agent() runs inside setup() *before* _init_params(), so _gumbel_tau
        # must already exist here; _init_params() re-reads it from custom_args and pushes the final
        # value into the agent.
        self._gumbel_tau = 0.7
        self._disc_grad_mode = "exact"
        self._use_log_alpha_loss = True
        self._lr_alpha = 3e-3
        self._disc_expl_flip_prob = 0.25
        self._init_std = None # None -> derived so the continuous branch starts at its entropy target

        # SIGNED. Negative for the continuous branch: the legacy ENTROPY_CONT_START=-0.5 is a
        # target of -0.5 nats/dim, i.e. frac -0.72. Matching it keeps the continuous branch
        # identical to the baseline, so B-vs-A isolates the discrete parametrization.
        self._target_H_cont_frac = -0.72
        self._target_H_disc_frac = 0.7
        self._target_H_cont = 0.0
        self._target_H_disc = 0.0
        self._max_H_cont = 0.0
        self._max_H_disc = 0.0

        self._this_child_path = os.path.abspath(__file__) # overrides parent

    # ------------------------------------------------------------------ setup

    def _create_agent(self, **kwargs):
        """Overrides `SoftActorCriticBase._create_agent()`; kwargs come resolved from setup().

        setup() builds the agent (line ~335) *before* it calls _init_params() (line ~395), so read
        the construction-time knobs straight from _hyperparameters, which setup() has already merged
        custom_args into (line ~295). `init_std` in particular cannot be applied after the fact: it
        sets the log_std head's bias at layer-init time.

        `actor_init_std` defaults to None, meaning "derive it from target_H_cont_frac" so the
        continuous branch starts *at* its entropy target. Pin it to a float only to deliberately
        start away from the target (e.g. 0.0273 to reproduce the legacy actor's near-deterministic
        init for a strict A/B on the continuous branch).
        """
        self._gumbel_tau = float(self._hyperparameters.get("gumbel_tau", self._gumbel_tau))
        self._target_H_cont_frac = float(self._hyperparameters.get("target_H_cont_frac",
                                                self._target_H_cont_frac))
        init_std = self._hyperparameters.get("actor_init_std", None)
        self._init_std = None if init_std in (None, "", "auto") else float(init_std)
        return HybridSACAgent(continuous_mask=self._env.is_action_continuous(),
                    gumbel_tau=self._gumbel_tau,
                    init_std=self._init_std,
                    target_H_cont_frac=self._target_H_cont_frac,
                    **kwargs)

    def _init_params(self, tot_tsteps: int, custom_args: dict = {}):

        SAC._init_params(self, tot_tsteps=tot_tsteps, custom_args=custom_args)

        # _disc_idxs / _cont_idxs are populated by the parent from the env action masks
        self._n_binary = int(self._disc_idxs.numel())
        self._n_cont = int(self._cont_idxs.numel())

        # gumbel_tau and actor_init_std were already resolved in _create_agent(), which runs first
        self._disc_grad_mode = str(custom_args.get("disc_grad_mode", "exact"))
        # DISCRETE-BRANCH ENTROPY handling. The flag GATES the continuous swing params -- if a flag
        # is not triggered its continuous dims have no effect on the transition -- so the flags are
        # the exploration bottleneck, and coupling their entropy to alpha_disc drove the divergence
        # seen on the hybrid runs (an unreachable Bernoulli target winds alpha_disc up, its entropy
        # bonus inflates soft-Q, Q and alpha co-diverge).
        #   off   : alpha_disc == 0. No discrete entropy in the objective; the discrete target is
        #           effectively deterministic (MAP flags). Exploration on the flags comes ONLY from
        #           the extrinsic flip envs (EXPL_ENVS_PERC / disc_expl_flip_prob), which is decoupled
        #           from the actor's decisiveness. Default.
        #   auto  : autotune alpha_disc toward target_H_disc_frac (max-entropy SAC on the flags).
        #   fixed : hold alpha_disc at alpha_disc_init (constant weight, no autotuning).
        self._disc_entropy_mode = str(custom_args.get("disc_entropy_mode", "off"))
        if self._disc_entropy_mode not in ("off", "auto", "fixed"):
            Journal.log(self.__class__.__name__, "_init_params",
                f"Unknown disc_entropy_mode '{self._disc_entropy_mode}'. Expected 'off', 'auto' or 'fixed'.",
                LogType.EXCEP, throw_when_excep=True)
        self._use_log_alpha_loss = bool(custom_args.get("use_log_alpha_loss", True))
        self._lr_alpha = float(custom_args.get("lr_alpha", 3e-3))
        self._disc_expl_flip_prob = float(custom_args.get("disc_expl_flip_prob", 0.25))
        # alpha bounds (anti-windup), applied by _clamp_log_alphas after each temperature step.
        #
        # FLOOR: on by default (1e-4). Harmless; keeps alpha from collapsing to 0.
        #
        # CEILING: OFF by default (alpha_max stays None unless the config sets a number). It was
        # originally added to contain the alpha divergence caused by the (since-fixed) UNREACHABLE
        # positive continuous entropy target. With the target signs correct the entropy target is
        # reachable, and alpha must be free to grow to the equilibrium that enforces it. Because the
        # actor-loss Q term scales like 1/(1-gamma) (~100 at gamma=0.99) while the entropy term is
        # O(alpha), that equilibrium is well above 1 -- a fixed ceiling of 1.0 silently starves the
        # entropy target (observed: alpha_disc pinned at 1.0, H_disc stuck ~0.30 below target).
        # Re-impose a ceiling only by setting alpha_max explicitly.
        if self._alpha_min is None:
            am = custom_args.get("alpha_min", 1e-4)
            self._alpha_min = 1e-4 if am in (None, "", "none") else float(am)
        self._hyperparameters["alpha_min"] = self._alpha_min
        self._hyperparameters["alpha_max"] = self._alpha_max  # None => no ceiling

        if self._disc_grad_mode not in ("st_gumbel", "exact"):
            Journal.log(self.__class__.__name__,
                "_init_params",
                f"Unknown disc_grad_mode '{self._disc_grad_mode}'. Expected 'st_gumbel' or 'exact'.",
                LogType.EXCEP,
                throw_when_excep = True)

        self._build_binary_combos()
        if self._disc_grad_mode == "exact" and self._n_binary > 0:
            k = 1 << self._n_binary
            Journal.log(self.__class__.__name__,
                "_init_params",
                f"disc_grad_mode='exact': the Q term is marginalized over all {k} = 2^{self._n_binary} "
                f"flag combinations. Unbiased in the logits and free of discrete sampling variance, "
                f"at the cost of {k}x critic rows per update ({k*self._batch_size} rows at "
                f"batch_size={self._batch_size}).",
                LogType.INFO)
            if k > 32:
                Journal.log(self.__class__.__name__,
                    "_init_params",
                    f"2^n_binary = {k} is large: the critic sees {k*self._batch_size} rows per update. "
                    f"Consider disc_grad_mode='st_gumbel' or a smaller batch_size.",
                    LogType.WARN)

        # entropy targets, as a SIGNED fraction of the attainable maximum (N*log2) for both
        # branches; see _refresh_entropy_targets. frac_cont is NEGATIVE by default, matching the
        # baseline's -0.5 nats/dim. Deliberately not named like the legacy
        # _entropy_{disc,cont}_{start,end} so the two conventions cannot be confused.
        self._target_H_cont_frac = float(custom_args.get("target_H_cont_frac", -0.72))
        self._target_H_disc_frac = float(custom_args.get("target_H_disc_frac", 0.7))
        self._target_H_cont_frac_start = float(custom_args.get("target_H_cont_frac_start",
                                                self._target_H_cont_frac))
        self._target_H_cont_frac_end = float(custom_args.get("target_H_cont_frac_end",
                                                self._target_H_cont_frac))
        self._target_H_disc_frac_start = float(custom_args.get("target_H_disc_frac_start",
                                                self._target_H_disc_frac))
        self._target_H_disc_frac_end = float(custom_args.get("target_H_disc_frac_end",
                                                self._target_H_disc_frac))

        self._alpha_cont = float(custom_args.get("alpha_cont_init", 0.2))
        self._alpha_disc = float(custom_args.get("alpha_disc_init", 1.0))
        if self._disc_entropy_mode == "off":
            self._alpha_disc = 0.0   # the discrete entropy term is out of the objective entirely
        self._alpha = 0.5*(self._alpha_disc+self._alpha_cont)

        self._refresh_entropy_targets()

        if self._n_binary == 0:
            Journal.log(self.__class__.__name__,
                "_init_params",
                "The environment declares no discrete action dims: the hybrid actor degenerates to "
                "a plain squashed-Gaussian policy and all discrete quantities will be identically 0.",
                LogType.WARN)

        # tau may have changed with respect to the value the agent was built with
        if hasattr(self._agent, "set_gumbel_tau"):
            self._agent.set_gumbel_tau(self._gumbel_tau)

        self._hyperparameters["gumbel_tau"] = self._gumbel_tau
        self._hyperparameters["disc_grad_mode"] = self._disc_grad_mode
        self._hyperparameters["disc_entropy_mode"] = self._disc_entropy_mode
        self._hyperparameters["use_log_alpha_loss"] = self._use_log_alpha_loss
        self._hyperparameters["lr_alpha"] = self._lr_alpha
        self._hyperparameters["disc_expl_flip_prob"] = self._disc_expl_flip_prob
        self._hyperparameters["actor_init_std"] = self._agent.init_std() # resolved value
        self._hyperparameters["n_binary_actions"] = self._n_binary
        self._hyperparameters["n_continuous_actions"] = self._n_cont
        self._hyperparameters["max_entropy_disc"] = self._max_H_disc
        self._hyperparameters["max_entropy_cont"] = self._max_H_cont
        self._hyperparameters["target_H_disc"] = self._target_H_disc
        self._hyperparameters["target_H_cont"] = self._target_H_cont
        self._hyperparameters["target_H_disc_frac"] = self._target_H_disc_frac
        self._hyperparameters["target_H_cont_frac"] = self._target_H_cont_frac

    def _refresh_entropy_targets(self):
        """Entropy targets, as a SIGNED fraction of the attainable maximum. One convention, both
        branches:

            target_H_disc = frac_disc * n_binary * log 2
            target_H_cont = frac_cont * n_cont   * log 2

        Both branches max out at `N * log 2` nats -- discrete because each flag is a Bernoulli
        (maximal at p = 0.5), continuous because the actions are tanh-squashed onto [-1, 1], a
        support of measure 2 per dim, whose maximum-entropy distribution is the uniform one. So
        `frac` means the same thing in both: how close to the maximally exploratory (fair-coin /
        uniform) policy this branch should sit. frac = 1 is that policy; frac = 0 is "zero nats".

        THE FRACS ARE SIGNED, and the sign matters:

          * `frac_disc` is necessarily in [0, 1]: a Bernoulli entropy is a true (discrete) entropy
            and cannot be negative.
          * `frac_cont` MAY BE NEGATIVE, and for this task it should be. H_cont is a *differential*
            entropy, which is unbounded below; a negative target simply asks for a policy more
            peaked than uniform. The legacy algorithm's `ENTROPY_CONT_START = -0.5` is a target of
            -0.5 nats/dim, i.e. `frac_cont = -0.72` -- NOT +0.72. (Its alpha loss equilibrates at
            `log_pi = -target_entropy`, i.e. `H = target_entropy`, so a negative stored value is a
            negative entropy target.) Setting frac_cont = +0.72 asks for a policy roughly 3x more
            exploratory than the baseline, which on Talos is at the edge of what is reachable while
            still tracking the task -- and an unreachable target makes the Lagrange multiplier
            integrate a permanent deficit and diverge. See _clamp_log_alphas.

        `_target_entropy_*` are the same quantities in the legacy fields (the legacy convention
        stores the entropy target directly, NOT its negation), kept in sync for the hyperparameter
        dump and the console print.
        """
        self._max_H_disc = float(self._n_binary)*math.log(2.0)
        self._max_H_cont = float(self._n_cont)*math.log(2.0)

        self._target_H_disc = self._target_H_disc_frac*self._max_H_disc
        self._target_H_cont = self._target_H_cont_frac*self._max_H_cont

        self._target_entropy_disc = self._target_H_disc
        self._target_entropy_cont = self._target_H_cont
        self._target_entropy = self._target_entropy_disc+self._target_entropy_cont
        self._trgt_avrg_entropy_per_action_disc = self._target_H_disc/float(max(self._n_binary, 1))
        self._trgt_avrg_entropy_per_action_cont = self._target_H_cont/float(max(self._n_cont, 1))
        self._trgt_avrg_entropy_per_action = self._target_entropy/float(max(self._actions_dim, 1))

    # _init_alpha_autotuning is inherited: the base now honours self._lr_alpha directly.

    # ------------------------------------------------- soft policy iteration hooks

    def _entropy_penalty(self, log_info):
        """alpha-weighted entropy term, [B, 1]. Overrides `SAC._entropy_penalty`.

        For the discrete factor, `E_b[-alpha_d * log pi(b|s)] = alpha_d * H_disc(s)` exactly, so we
        substitute the analytic Bernoulli entropy for the sampled log-probability. Same expectation,
        zero sampling variance. Hence

            penalty = alpha_cont * logp_cont - alpha_disc * H_disc

        which the inherited update logic uses in both of the places it needs it:

            critic target:  min_q_next - penalty  =  min_q_next - a_c*logp_c + a_d*H_disc
            actor loss:    (penalty - min_q_pi)   =  a_c*logp_c - a_d*H_disc - min_q_pi

        matching sections 8 and 9 of the plan. In the actor loss `H_disc` is differentiable, and its
        gradient is what pushes the binary logits toward higher entropy; in the target it is
        computed under no_grad.
        """
        logp_cont = log_info["logp_cont"]
        entropy_disc = log_info["entropy_disc"]
        alpha_cont = self._alpha_tensor("cont", logp_cont)
        alpha_disc = self._alpha_disc_value(logp_cont)
        return alpha_cont*logp_cont - alpha_disc*entropy_disc

    def _alpha_disc_value(self, ref_tensor):
        """Effective discrete temperature used in the objective, per disc_entropy_mode.

            off   -> 0            (discrete entropy term drops out of actor loss AND critic target)
            fixed -> alpha_disc   (constant alpha_disc_init)
            auto  -> exp(log_alpha_disc), detached (autotuned; falls back to the scalar if autotune
                     is globally off)

        Returned as a tensor on ref_tensor's device/dtype so it broadcasts against entropy_disc.
        """
        if self._disc_entropy_mode == "off":
            return ref_tensor.new_zeros(())
        if self._disc_entropy_mode == "fixed":
            return ref_tensor.new_tensor(self._alpha_disc)
        # auto
        if self._autotune:
            return self._log_alpha_disc.exp().detach()
        return ref_tensor.new_tensor(self._alpha_disc)

    def _alpha_losses(self, log_info):
        """(alpha_loss_disc, alpha_loss_cont). Overrides `SAC._alpha_losses`.

        Positive-entropy convention:

            L(alpha) = log_alpha * (H_est - target_H)

        so dL/d log_alpha = H_est - target_H: entropy below target drives log_alpha (and alpha) up,
        entropy above target drives it down. Note the multiplier is `log_alpha`, not
        `exp(log_alpha)` as in the legacy loss, which scaled the alpha gradient by alpha itself and
        let it stall near zero. Set `use_log_alpha_loss=False` to recover the legacy multiplier.

        Entropy estimates:
            H_cont = -logp_cont     unbiased estimator of the differential entropy; upper-bounded by
                                    n_cont * log 2 because the actions are tanh-squashed onto [-1,1]
            H_disc                  exact, in [0, n_binary * log 2]
        """
        if not self._autotune:
            return None, None
        h_cont = (-log_info["logp_cont"]).detach()
        mult_cont = self._log_alpha_cont if self._use_log_alpha_loss else self._log_alpha_cont.exp()
        alpha_loss_cont = (mult_cont*(h_cont-self._target_H_cont)).mean()
        # alpha_disc is autotuned only in 'auto' mode; 'off'/'fixed' leave log_alpha_disc frozen
        # (the base update loop skips the disc optimizer step when this is None).
        if self._disc_entropy_mode == "auto":
            h_disc = log_info["entropy_disc"].detach()
            mult_disc = self._log_alpha_disc if self._use_log_alpha_loss else self._log_alpha_disc.exp()
            alpha_loss_disc = (mult_disc*(h_disc-self._target_H_disc)).mean()
        else:
            alpha_loss_disc = None
        return alpha_loss_disc, alpha_loss_cont

    def _sync_alpha_scalars(self):
        """Overrides `SAC._sync_alpha_scalars` to report alpha_disc per disc_entropy_mode:
        0 when 'off', the frozen constant when 'fixed', exp(log_alpha_disc) when 'auto'. alpha_cont
        always tracks its log-temperature. Keeps telemetry consistent with the effective objective.
        """
        self._alpha_cont = self._log_alpha_cont.exp().item()
        if self._disc_entropy_mode == "off":
            self._alpha_disc = 0.0
        elif self._disc_entropy_mode == "auto":
            self._alpha_disc = self._log_alpha_disc.exp().item()
        # 'fixed': leave self._alpha_disc at its init value
        self._alpha = 0.5*(self._alpha_disc + self._alpha_cont)

    def _policy_entropies(self, log_info):
        """(H_total, H_disc, H_cont), each [B, 1]. Overrides `SAC._policy_entropies`.

        Careful when reading the logs: `H_disc` is a true discrete entropy in [0, n_binary*log 2],
        while `H_cont` is a differential entropy (unbounded below, upper-bounded by n_cont*log 2).
        Their sum is logged for continuity with the legacy series but mixes the two units and is not
        comparable across formulations.
        """
        h_cont = -log_info["logp_cont"]
        h_disc = log_info["entropy_disc"]
        return h_cont+h_disc, h_disc, h_cont

    def _policy_entropy_total(self, log_info):
        """Overrides `SoftActorCriticBase._policy_entropy_total` (used for the startup print)."""
        return -log_info["logp_cont"]+log_info["entropy_disc"]

    def _update_target_entropy_from_metric(self, metric: float):
        """Overrides `SAC._update_target_entropy_from_metric` to anneal the *positive* targets.

        With `anneal_entropy=False` (the default) the start/end values coincide and this is a no-op.
        """
        if not self._anneal_entropy:
            return
        metric_clamped = max(min(metric, self._entropy_metric_high), self._entropy_metric_low)
        denom = max(self._entropy_metric_high-self._entropy_metric_low, 1e-6)
        progress = (self._entropy_metric_high-metric_clamped)/denom

        self._target_H_disc_frac = self._target_H_disc_frac_start + \
            progress*(self._target_H_disc_frac_end-self._target_H_disc_frac_start)
        self._target_H_cont_frac = self._target_H_cont_frac_start + \
            progress*(self._target_H_cont_frac_end-self._target_H_cont_frac_start)

        self._refresh_entropy_targets()

    # ------------------------------------------- Q term: exact marginalization over the flags

    def _build_binary_combos(self):
        """All 2^n_binary flag combinations, [K, n_binary] with entries in {0, 1}."""
        if self._n_binary == 0:
            self._combos = None
            return
        k = 1 << self._n_binary
        idx = torch.arange(k, device=self._torch_device)
        bits = torch.arange(self._n_binary, device=self._torch_device)
        # bit i of combination j
        self._combos = ((idx.unsqueeze(1) >> bits.unsqueeze(0)) & 1).to(self._dtype) # [K, n_binary]

    def _expected_min_q(self, obs, action, log_info, use_target: bool):
        """E_b[ min(Q1, Q2)(s, a_cont, b) ] under pi(b|s), computed EXACTLY. Returns [B, 1].

        The straight-through estimator is biased: it backpropagates dQ/db times the Gumbel-Sigmoid
        surrogate's derivative, which has the right sign but the wrong magnitude. Because the two
        heads are conditionally independent given the observation,

            E_{a_cont, b}[Q] = E_{a_cont}[ sum_b pi(b|s) * Q(s, a_cont, b) ]

        the inner sum can be enumerated over the 2^n_binary combinations. With one reparametrized
        continuous sample this is fully differentiable in a_cont AND exactly differentiable in the
        logits (through pi(b|s)) -- zero gradient bias, and zero discrete sampling variance.

        Cost: K = 2^n_binary critic evaluations, batched into a single call of B*K rows. Talos has
        n_binary = 2 -> K = 4, so it is nearly free; a quadruped has n_binary = 4 -> K = 16.

        pi(b|s) is assembled in log space:

            log pi(b_k|s) = sum_i [ b_ki*log p_i + (1 - b_ki)*log(1 - p_i) ]
                          = logsigmoid(logits) . b_k  +  logsigmoid(-logits) . (1 - b_k)
        """
        B = obs.shape[0]
        K = self._combos.shape[0]

        cont_idxs = self._cont_idxs.to(action.device)
        binary_idxs = self._disc_idxs.to(action.device)
        a_cont = action.index_select(1, cont_idxs) # [B, n_cont]; keeps the reparametrized gradient

        combos = self._combos.to(device=action.device, dtype=action.dtype) # [K, n_binary]
        b_pm1 = 2.0*combos - 1.0 # the env's -1/+1 encoding

        # [B, K, actions_dim] -> [B*K, actions_dim]
        act = torch.zeros((B, K, self._actions_dim), device=action.device, dtype=action.dtype)
        act[:, :, cont_idxs] = a_cont.unsqueeze(1).expand(B, K, a_cont.shape[1])
        act[:, :, binary_idxs] = b_pm1.unsqueeze(0).expand(B, K, self._n_binary)
        act = act.reshape(B*K, self._actions_dim)

        obs_rep = obs.unsqueeze(1).expand(B, K, obs.shape[1]).reshape(B*K, obs.shape[1])

        if use_target:
            q1 = self._agent.get_qf1t_val(obs_rep, act)
            q2 = self._agent.get_qf2t_val(obs_rep, act)
        else:
            q1 = self._agent.get_qf1_val(obs_rep, act)
            q2 = self._agent.get_qf2_val(obs_rep, act)
        min_q = torch.min(q1, q2).view(B, K) # [B, K]

        logits = log_info["binary_logits"] # [B, n_binary]; carries the gradient into the head
        log_p = F.logsigmoid(logits)
        log_1mp = F.logsigmoid(-logits)
        log_pi = log_p @ combos.t() + log_1mp @ (1.0-combos).t() # [B, K]
        pi_b = log_pi.exp() # rows sum to 1 by construction

        return (pi_b*min_q).sum(dim=1, keepdim=True)

    def _min_q_actor(self, obs, action, log_info):
        if self._disc_grad_mode != "exact" or self._n_binary == 0:
            return SAC._min_q_actor(self, obs, action, log_info)
        return self._expected_min_q(obs, action, log_info, use_target=False)

    def _min_q_target(self, next_obs, next_action, next_log_info):
        if self._disc_grad_mode != "exact" or self._n_binary == 0:
            return SAC._min_q_target(self, next_obs, next_action, next_log_info)
        # already under no_grad at the call site; marginalizing here is not about bias (the target is
        # not differentiated) but about variance: it replaces one b' sample with the exact expectation
        return self._expected_min_q(next_obs, next_action, next_log_info, use_target=True)

    # ------------------------------------------------------------------ telemetry

    def _init_custom_dbdata(self):
        """Allocate the `hybrid/*` series. Same [_db_data_size, 1] layout as the base series, so
        they dump to hdf5 and plot exactly like the rest."""
        def _series():
            return torch.full((self._db_data_size, 1), dtype=torch.float32,
                fill_value=torch.nan, device="cpu")

        self._binary_prob_mean = _series()
        self._binary_prob_min = _series()
        self._binary_prob_max = _series()
        self._binary_rate_mean = _series()   # fraction of flags commanded "on" by the sampled policy
        self._binary_flip_rate = _series()   # flips per flag per step in the sampled batch: chattering
        self._H_disc_total = _series()
        self._H_disc_per_flag = _series()
        self._H_disc_frac = _series()        # H_disc / (n_binary * log 2), in [0, 1]
        self._H_cont_total = _series()
        self._H_cont_per_dim = _series()
        self._H_cont_frac = _series()        # H_cont / (n_cont * log 2), <= 1 but may be negative
        self._logp_cont_mean = _series()
        self._actor_std_mean = _series()     # mean std of the continuous head
        self._logits_grad_norm = _series()   # gradient reaching the binary head: the ST path, live
        self._replay_binary_ok = _series()   # 1.0 iff every binary column in the batch is +-1

        # flip-rate tracking across collection steps
        self._prev_binary = None
        self._flip_accum = 0.0
        self._flip_count = 0

    def _update_custom_dbdata(self, log_info, obs=None):
        i = self._log_it_counter
        with torch.no_grad():
            h_disc = log_info["entropy_disc"]
            h_cont = -log_info["logp_cont"]
            self._H_disc_total[i, 0] = h_disc.mean().item()
            self._H_cont_total[i, 0] = h_cont.mean().item()
            self._H_disc_per_flag[i, 0] = h_disc.mean().item()/float(max(self._n_binary, 1))
            self._H_cont_per_dim[i, 0] = h_cont.mean().item()/float(max(self._n_cont, 1))
            self._H_disc_frac[i, 0] = h_disc.mean().item()/self._max_H_disc \
                if self._max_H_disc > 0.0 else float("nan")
            self._H_cont_frac[i, 0] = h_cont.mean().item()/self._max_H_cont \
                if self._max_H_cont > 0.0 else float("nan")
            self._logp_cont_mean[i, 0] = log_info["logp_cont"].mean().item()

            if self._n_binary > 0:
                p = log_info["binary_probs"]
                self._binary_prob_mean[i, 0] = p.mean().item()
                self._binary_prob_min[i, 0] = p.min().item()
                self._binary_prob_max[i, 0] = p.max().item()
                self._binary_rate_mean[i, 0] = log_info["binary_hard"].mean().item()

            # gradient norm on the binary head, taken after the actor step: if the straight-through
            # path ever silently detaches, this goes to zero while everything else still looks fine
            logits_layer = self._agent.actor.fc_logits[0]
            if logits_layer.weight.grad is not None:
                self._logits_grad_norm[i, 0] = logits_layer.weight.grad.norm().item()

            # flip rate accumulated over the collection steps since the last log iteration
            if self._flip_count > 0:
                self._binary_flip_rate[i, 0] = self._flip_accum/float(self._flip_count)
            self._flip_accum = 0.0
            self._flip_count = 0

    # _collect_transition and _track_binary_flips are inherited. The parent's sign-based flip
    # tracker is correct here too: the hybrid flags are exactly +-1, so a sign change IS a flip,
    # and both algorithms then report `binary_flip_rate` in the same units.

    def _log_actor_std(self, obs):
        """Mean std of the continuous head over a batch, for the entropy diagnostics."""
        with torch.no_grad():
            _, log_std, _ = self._agent.actor(self._agent._preprocess_obs(obs))
            return log_std.exp().mean().item()

    def _check_replay_binary_support(self, actions):
        """The central invariant: every binary column of a replay batch must be exactly +-1.

        If warmstart or the exploration noise ever writes an intermediate flag value, the critic is
        trained off the two-point support the actor lives on -- which is the very defect this
        refactor removes, and it would be invisible in the returns. Cheap, so it runs every debug
        policy update rather than once.
        """
        if self._n_binary == 0:
            return True
        binary = actions.index_select(1, self._disc_idxs.to(actions.device))
        ok = bool(torch.all((binary == 1.0) | (binary == -1.0)).item())
        if not ok:
            bad = binary[(binary != 1.0) & (binary != -1.0)]
            Journal.log(self.__class__.__name__,
                "_check_replay_binary_support",
                f"Replay buffer contains {bad.numel()} non-binary flag values "
                f"(e.g. {bad.flatten()[:4].tolist()}). The critic is being trained off-support.",
                LogType.EXCEP,
                throw_when_excep = True)
        return ok

    def _sample(self, size: int = None):
        """Overrides the parent sampler purely to assert the replay invariant while debugging."""
        obs, actions, next_obs, rewards, next_terminal = SAC._sample(self, size=size)
        if self._debug:
            self._replay_binary_ok[self._log_it_counter, 0] = \
                1.0 if self._check_replay_binary_support(actions) else 0.0
            self._actor_std_mean[self._log_it_counter, 0] = self._log_actor_std(obs)
        return obs, actions, next_obs, rewards, next_terminal

    def _dump_custom_dbdata(self, hf, _ds):
        _ds('hybrid_binary_prob_mean', self._binary_prob_mean)
        _ds('hybrid_binary_prob_min', self._binary_prob_min)
        _ds('hybrid_binary_prob_max', self._binary_prob_max)
        _ds('hybrid_binary_rate_mean', self._binary_rate_mean)
        _ds('hybrid_binary_flip_rate', self._binary_flip_rate)
        _ds('hybrid_H_disc_total', self._H_disc_total)
        _ds('hybrid_H_disc_per_flag', self._H_disc_per_flag)
        _ds('hybrid_H_disc_frac', self._H_disc_frac)
        _ds('hybrid_H_cont_total', self._H_cont_total)
        _ds('hybrid_H_cont_per_dim', self._H_cont_per_dim)
        _ds('hybrid_H_cont_frac', self._H_cont_frac)
        _ds('hybrid_logp_cont_mean', self._logp_cont_mean)
        _ds('hybrid_actor_std_mean', self._actor_std_mean)
        _ds('hybrid_logits_grad_norm', self._logits_grad_norm)
        _ds('hybrid_replay_binary_ok', self._replay_binary_ok)

    def _custom_db_info_str(self):
        i = self._log_it_counter
        return (
            f"[hybrid] H_disc {float(self._H_disc_total[i, 0]):.4f}/{self._target_H_disc:.4f} nats "
            f"(frac {float(self._H_disc_frac[i, 0]):.3f}/{self._target_H_disc_frac:.3f}, "
            f"max {self._max_H_disc:.4f})\n"
            f"[hybrid] H_cont {float(self._H_cont_total[i, 0]):.4f}/{self._target_H_cont:.4f} nats "
            f"(frac {float(self._H_cont_frac[i, 0]):.3f}/{self._target_H_cont_frac:.3f}, "
            f"max {self._max_H_cont:.4f})\n"
            f"[hybrid] alpha_disc {self._alpha_disc:.5f}, alpha_cont {self._alpha_cont:.5f}, "
            f"tau {self._gumbel_tau:.2f}\n"
            f"[hybrid] binary p (mean/min/max) {float(self._binary_prob_mean[i, 0]):.4f} / "
            f"{float(self._binary_prob_min[i, 0]):.4f} / {float(self._binary_prob_max[i, 0]):.4f}, "
            f"on-rate {float(self._binary_rate_mean[i, 0]):.4f}\n"
            f"[hybrid] actor std {float(self._actor_std_mean[i, 0]):.4f}, "
            f"logits grad norm {float(self._logits_grad_norm[i, 0]):.3e}, "
            f"replay binary ok {float(self._replay_binary_ok[i, 0]):.0f}\n"
        )

    # ------------------------------------------------------- exploration on binary dims

    def _sample_random_actions(self):
        """Warmstart actions. Overrides the parent's uniform draw over every dim.

        The binary dims must be drawn from Bernoulli(0.5) and mapped to -1/+1, otherwise the replay
        buffer -- and hence the critic -- would be populated with flag values the hybrid actor can
        never emit. That off-support extrapolation is precisely what this refactor removes.

        `sign(U(-1, 1))` is already a fair coin, so no extra RNG draw is needed.
        """
        self._random_uniform.uniform_(-1, 1)
        if self._n_binary > 0:
            disc = self._disc_idxs.to(self._random_uniform.device)
            binary = self._random_uniform.index_select(1, disc)
            self._random_uniform[:, disc] = torch.where(binary >= 0.0,
                                                torch.ones_like(binary),
                                                -torch.ones_like(binary))
        return self._random_uniform

    def _perturb_some_actions(self, actions: torch.Tensor):
        """Exploration noise. Continuous dims get the parent's Gaussian perturbation; binary dims are
        randomly flipped instead of having uniform noise added to them.

        The parent perturbs the discrete dims with `uniform(-1,1) * 1.2` followed by a clamp, which
        would turn a hard flag into an arbitrary value in [-1, 1].
        """
        if self._is_continuous_actions_bool.any():
            self._perturb_actions(actions,
                action_idxs=self._is_continuous_actions,
                env_idxs=self._expl_env_selector.to(actions.device),
                normal=True, # use normal for continuous
                scaling=self._continuous_act_expl_noise_std)
        if self._is_discrete_actions_bool.any():
            self._flip_binary_actions(actions,
                action_idxs=self._is_discrete_actions,
                env_idxs=self._expl_env_selector.to(actions.device),
                flip_prob=self._disc_expl_flip_prob)
        self._pert_counter+=1
        if self._pert_counter >= self._noise_duration_vec:
            self._pert_counter=0

    def _flip_binary_actions(self,
        actions: torch.Tensor,
        action_idxs: torch.Tensor,
        env_idxs: torch.Tensor,
        flip_prob: float):
        """Flip each selected binary action with probability `flip_prob`, in place.

        `a -> -a` maps +1 <-> -1 and so keeps the action on the two-point support.
        """
        env_indices = env_idxs.reshape(-1, 1)
        action_indices = action_idxs.reshape(1, -1)
        selected = actions[env_indices, action_indices]
        flip = torch.rand_like(selected) < flip_prob
        actions[env_indices, action_indices] = torch.where(flip, -selected, selected)
