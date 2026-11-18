"""QARL (Quantal Adversarial RL, Reddi et al. 2023, "Robust Adversarial
Reinforcement Learning via Bounded Rationality Curricula") — plain-PyTorch
port for the unified MuJoCo baseline package.

Upstream: /root/mjbexp/upstreams/quantal-adversarial-rl (mushroom-rl 1.9.1 +
dm_control), ported to the shared gymnasium-v4 stack. Hyperparameters come
from the frozen experiment config `mujoco_baselines.config.QARL` (RMBPO paper
appendix D, MuJoCo adaptation); everything the config does not pin is taken
from the upstream code.

Interface (called by the runner):
    train(job_dir, task, seed, cfg)      -- trains to cfg["steps"] real env
                                             transitions, resumable/idempotent
    load_eval_adapter(job_dir, task, seed, cfg) -- frozen-eval adapter
    (deterministic protagonist SAC mean action, NO adversary, temperature
    observation input = 0.0, the QARL official no-adversary evaluation
    default)

========================================================================== PORT FIDELITY ==========================================================================

PRESERVED (semantics ported from upstream, verified against source):
* Training-loop structure (src/experiments/mujoco_experiments/qarl_experiment.py
  train_protagonist): per iteration the teacher samples
  n_episodes_per_iteration=30 temperatures (ONE per episode,
  sample_n_temperatures); the ADVERSARY learn block runs first (30 episodes
  in which only the adversary fits and only the adversary's replay memory
  grows), then the PROTAGONIST learn block runs 30 episodes with the SAME 30
  temperatures (StateVariableConcatenation episode indexing resets per
  core.learn call). Both players act in EVERY simulation step (one counted
  transition per step; MultiAgentCore._step draws both actions, the
  non-learning agent's samples are discarded, replicating
  get_n_steps_per_fit_per_agent which gives the learning agent
  n_steps_per_fit and the other agent an interval larger than the whole
  block so it never fits).
* Temperature input: exactly ONE scalar per episode — the teacher's
  temperature for that episode, which IS the adversary's SAC entropy
  temperature — is concatenated to the observation seen by BOTH agents
  (src/utils/state_preprocessor.py + SelfPacedRLCore.set_variables_for_
  training). The protagonist thus observes the adversary's temperature
  (task book: "双方读入对手温度 α"); the adversary additionally USES it as
  its alpha (see next point).
* The two SAC variants
  (src/mushroom_rl_extensions/algorithms/actor_critic/{sac,adaptive_temp_sac}.py):
  the PROTAGONIST is a plain SAC whose entropy temperature alpha is a
  learned scalar (Adam, lr 3e-4, alpha_loss = -(log_alpha*(log_prob +
  target_entropy)).mean(), log_alpha initialised to -5.30 => alpha=0.005).
  The ADVERSARY ("AdaptiveTempSAC") does NOT learn alpha: its
  _update_alpha(state[:, -1]) OVERWRITES log_alpha with log(temperature)
  read from the temperature-augmented states of the current batch — a
  per-sample VECTOR (no gradient step; its alpha optimizer is never
  stepped). This is the "different entropy temperature update" of the two
  players. The adversary's alpha vector (built from the batch's CURRENT
  states) is also the one multiplied into the NEXT-state log-probs inside
  its critic target — an upstream quirk (alpha of s, applied to s') that is
  preserved here.
* SAC update semantics per fit() call (mushroom-rl 1.9.1 verified from
  source): add the new transitions to the replay memory; if the replay is
  initialised (size > 3000) draw ONE uniform-with-replacement batch of 256
  and do, in this order: (1) [adversary only] set alpha from batch temps;
  (2) if replay size > 5000 (warmup): one actor gradient step on
  (alpha*log_prob - min(q_0,q_1)).mean() over mu+sigma networks (Adam), and
  for the protagonist one alpha gradient step; (3) critic target =
  r + gamma * (min over twin TARGET critics(s',a'(s')) - alpha*log_prob(a'))
  with the terminal (terminated, not truncated) entries zeroed, then ONE
  gradient step per twin critic (separate Adams) on MSE; (4) soft target
  update tau=0.005. No gradient clipping, no random-action warmup (actions
  are always sampled from the policy; upstream has no random prefill, only
  the replay-size gates — the pre-warmup steps are ordinary counted
  transitions).
* Networks (setup_sac_agent.py): SEPARATE mu and sigma actors (3 hidden
  layers of 256, xavier_uniform init with relu gain, linear gain on the
  output; default PyTorch bias init), twin critics with the same trunk on
  concat(state, action); log_std clamped to [-20, 2]; tanh-squashed
  reparameterised policy with log-prob correction log(1 - a^2 + 1e-6);
  target critic initialised as a copy of the online critic. Replay memory:
  ring buffer of 1e6, uniform sampling with replacement via the global
  numpy RNG (mushroom ReplayMemory). lr: critic 3e-4, alpha 3e-4 (upstream
  values); actor lr from cfg (see AMBIGUITIES).
* Teacher (teachers/self_paced_gamma_teacher_fixed_beta.py): ported
  essentially verbatim — torch Gamma(conc, rate) distributions with fixed
  rate (1000), scipy trust-constr minimisation of KL(dist||target) over the
  concentration subject to the nonlinear constraints KL(new||old) <= max_kl
  (0.5) and importance-weighted expected performance >= lower bound
  (hopper 1.0 / halfcheetah 10 / walker2d 10), conc lower bound 1e-8 and
  target-KL 0.01 threshold deciding the conc bounds branch, the
  "frozen until performance has ever been above the bound" latch, the
  keep-old-distribution fallback when the optimiser fails, and
  G_TOL/X_TOL = 1e-4/1e-6.
* Temp/performance buffer (utils/buffer.py): Buffer(2, 100, FIFO),
  (temperature, discounted return with gamma=0.99) pairs for EVERY episode
  of BOTH blocks (60 per iteration), teacher update once the buffer holds
  >= n_samples_for_dist_update=30 samples, buffer emptied on read (so the
  update fires every iteration with the freshest ~60 pairs).
* In-training progress evaluation replicates evaluate_vs_adversary /
  evaluate_without_adversary: stochastic draw_action for both players
  (mushroom evaluates with sampled actions), evaluation temperature =
  teacher.temp_dist.mean(), zero-force constant adversary for the
  no-adversary arm, undiscounted episode return (compute_J default
  gamma=1.0).

ADAPTED (documented deviations forced by the shared protocol):
* dm_control two-player envs -> gymnasium v4 PerturbEnv wrapped in
  TwoPlayerForceEnv: the force mechanics are identical to the upstream
  two-player physics (xfrc_applied written on the listed bodies, world x/z
  components, +-1 N cap, cleared after each step; bodies/axes/force from
  cfg/constants). Observation/reward/termination are the vanilla v4 ones
  (upstream used dm_control hopper/walker/cheetah rewards); episode cap
  1000 via TimeLimit (upstream main.py default horizon 500, dm_control
  hopper 1000 — the shared EPISODE_CAP=1000 governs here).
* mushroom-rl -> plain torch: TorchApproximator.fit with batch_size=0 and
  n_epochs=1 means exactly one gradient step per fit on the full 256-sample
  batch (reproduced); ReplayMemory.get uses np.random.randint (reproduced);
  states stored float32 (upstream float64 — identical after the networks'
  .float() cast).
* Budget: upstream stops after n_total_iterations; here training runs until
  cfg["steps"] real transitions, cutting the final episode mid-way if
  needed (the partial episode still contributes a (temp, J) pair, matching
  upstream compute_J's tail handling). temperature_samples counts teacher
  draws (30 per started iteration).
* Fit cadence: upstream adjust_steps_per_fit gives n_steps_per_fit=3 for
  SAC (one gradient step per 3 env steps of the learning agent, inside the
  alternating blocks). The frozen config specifies utd=1, so the learning
  agent takes cfg["utd"] gradient steps per env step here; the block
  structure, per-agent replay growth and adversary-first order are
  preserved. (The task instruction's "each fit once per step" is realised
  as: every counted transition triggers exactly one fit — of whichever
  agent owns the current block — with utd gradient rounds.)
* In-training evaluation cadence: upstream evaluates 1 episode per arm per
  iteration; here every >= 50k steps (cfg override "eval_interval") plus
  the final iteration, 3 episodes per arm (cfg override
  "eval_episodes"), never counted in the budget, never used for
  checkpoint selection (final checkpoint only — upstream additionally
  logged "best" agents).
* Deterministic evaluation: the frozen evaluator requires the SAC MEAN
  action; the adapter returns tanh(mu) rescaled to the action box
  (upstream only ever samples actions; the mean of the squashed Gaussian is
  approximated the standard way, as SB3 does).
* Determinism: torch.manual_seed / np.random.seed / first env reset
  seeded with `seed`; RNG states are saved in the periodic checkpoint so a
  resumed run continues the same streams.
* gamma = 0.99 (upstream main.py default; not in cfg).

AMBIGUITIES RESOLVED:
1. Actor lr: upstream SetupSACAgent.LR_ACTOR = 1e-4, but the frozen config
   (RMBPO paper appendix D) pins actor_lr = 3e-4. The config is the
   experiment's single source of truth -> cfg["actor_lr"] is used for BOTH
   players' actors; critic/alpha lrs keep the upstream 3e-4.
2. utd: upstream 1/3 (fit every 3 steps) vs cfg utd=1 -> cfg wins, block
   structure preserved (see ADAPTED).
3. Temperature at final evaluation: upstream's standalone evaluate() uses
   evaluation_temp=0.0 for the official no-adversary evaluation (main.py),
   while its in-training loop uses teacher.temp_dist.mean(); we follow both
   respectively (adapter appends 0.0, progress evals use the teacher mean).
4. The adversary's alpha in its critic target is taken from the batch's
   current-state temperatures while multiplying next-state log-probs
   (upstream quirk) — preserved.
5. Upstream's teacher fallback branch ("res not successful but improved")
   compares res.fun against objective(...) which returns a (value, grad)
   tuple — a latent upstream TypeError that only triggers on optimiser
   failure; we compare against the value only (intent preserved).
6. dtype inside the teacher is normalised to float64 (upstream mixes
   float32/float64 relying on torch promotion).

File outputs (job_dir = runs/shared/qarl/<task>/seed<k>/):
    ckpt/final.pt   protagonist (+ adversary, teacher) final state
    ckpt/state.pt   periodic full resume state (every iteration)
    counters.json   {"real_transitions", "protagonist_steps",
                     "adversary_steps", "temperature_samples"}
    progress.jsonl  one line per iteration:
                     {"step", "mean_return_vs_adv", "mean_return_no_adv",
                      "mean_temp", "wall_s"} (returns null unless an
                     evaluation ran that iteration)
Counters semantics: both players act in every counted transition, so
protagonist_steps == adversary_steps == real_transitions; each agent's
gradient updates happen only inside its own block (upstream cadence).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from ..constants import ADVERSARY_BODIES, ADVERSARY_FORCE
from ..envs.force import TwoPlayerForceEnv
from ..envs.perturbed import make_train_env

# ---------------------------------------------------------------------------
# Constants not pinned by the frozen cfg (upstream SetupSACAgent defaults).
# `cfg` may override "gamma", "initial_replay_size", "warmup_transitions",
# "eval_interval", "eval_episodes" and "n_iterations_per_agent" (used by the
# smoke test only; the production config leaves them at these values).
# ---------------------------------------------------------------------------
_GAMMA = 0.99                       # upstream main.py default
_INITIAL_REPLAY_SIZE = 3000         # replay "initialized" when size > this
_MAX_REPLAY_SIZE = 1_000_000
_BATCH_SIZE = 256
_N_FEATURES = 256
_WARMUP_TRANSITIONS = 5000          # actor/alpha updates when size > this
_TAU = 0.005
_LR_ALPHA = 3e-4
_LR_CRITIC = 3e-4
_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0
_LOG_ALPHA_INIT = -5.30             # alpha = 0.005 (upstream hard-coded)
_EPS_LOG_PROB = 1e-6
_EVAL_INTERVAL = 50_000             # progress-eval cadence (steps)
_EVAL_EPISODES = 3                  # episodes per arm per progress eval
_G_TOL = 1e-4                       # teacher trust-constr tolerances
_X_TOL = 1e-6
_N_ITERATIONS_PER_AGENT = 1         # upstream default


# ============================================================ networks ===
class ActorNetwork(nn.Module):
    """Ported from setup_sac_agent.ActorNetwork: 3 hidden layers, xavier."""

    def __init__(self, input_dim: int, output_dim: int,
                 n_features: int = _N_FEATURES):
        super().__init__()
        self._in = nn.Linear(input_dim, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_dim)
        nn.init.xavier_uniform_(self._in.weight,
                                gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        x = state.float() if state.dtype != torch.float32 else state
        x = F.relu(self._in(x))
        x = F.relu(self._h1(x))
        x = F.relu(self._h2(x))
        return self._out(x)


class CriticNetwork(nn.Module):
    """Ported from setup_sac_agent.CriticNetwork (state-action input)."""

    def __init__(self, input_dim: int, n_features: int = _N_FEATURES):
        super().__init__()
        self._in = nn.Linear(input_dim, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self._in.weight,
                                gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        x = torch.cat((state.float(), action.float()), dim=1)
        x = F.relu(self._in(x))
        x = F.relu(self._h1(x))
        x = F.relu(self._h2(x))
        return self._out(x).squeeze(-1)


# ======================================================= replay memory ===
class ReplayMemory:
    """mushroom_rl ReplayMemory semantics on preallocated numpy ring arrays.

    ``initialized`` is size > initial_size (strict, as upstream); ``get``
    samples uniformly WITH replacement through the global numpy RNG.
    """

    def __init__(self, initial_size: int, max_size: int,
                 obs_dim: int, act_dim: int):
        self._initial_size = initial_size
        self._max_size = max_size
        self._idx = 0
        self._full = False
        self._s = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._a = np.zeros((max_size, act_dim), dtype=np.float32)
        self._r = np.zeros((max_size,), dtype=np.float32)
        self._ns = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._ab = np.zeros((max_size,), dtype=bool)
        self._last = np.zeros((max_size,), dtype=bool)

    def add(self, dataset):
        for s, a, r, ns, ab, last in dataset:
            i = self._idx
            self._s[i] = s
            self._a[i] = a
            self._r[i] = r
            self._ns[i] = ns
            self._ab[i] = ab
            self._last[i] = last
            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples: int):
        idx = np.random.randint(self.size, size=n_samples)
        return (self._s[idx], self._a[idx], self._r[idx],
                self._ns[idx], self._ab[idx], self._last[idx])

    @property
    def initialized(self) -> bool:
        return self.size > self._initial_size

    @property
    def size(self) -> int:
        return self._idx if not self._full else self._max_size

    def state(self) -> dict:
        n = self.size
        return {"idx": self._idx, "full": self._full,
                "s": self._s[:n].copy(), "a": self._a[:n].copy(),
                "r": self._r[:n].copy(), "ns": self._ns[:n].copy(),
                "ab": self._ab[:n].copy(), "last": self._last[:n].copy()}

    def load(self, d: dict):
        self._idx = int(d["idx"])
        self._full = bool(d["full"])
        n = int(d["s"].shape[0])
        self._s[:n] = d["s"]
        self._a[:n] = d["a"]
        self._r[:n] = d["r"]
        self._ns[:n] = d["ns"]
        self._ab[:n] = d["ab"]
        self._last[:n] = d["last"]


# ============================================================= policy ====
class SACPolicy:
    """Ported from sac.SACPolicy (tanh-squashed Gaussian, mu/sigma nets)."""

    def __init__(self, mu_net: ActorNetwork, sigma_net: ActorNetwork,
                 low: np.ndarray, high: np.ndarray, device: torch.device):
        self._mu = mu_net
        self._sigma = sigma_net
        self._device = device
        self._delta = torch.tensor(0.5 * (np.asarray(high) - np.asarray(low)),
                                   dtype=torch.float32, device=device)
        self._central = torch.tensor(0.5 * (np.asarray(high) + np.asarray(low)),
                                      dtype=torch.float32, device=device)

    def distribution(self, state_t):
        mu = self._mu(state_t)
        log_sigma = torch.clamp(self._sigma(state_t), _LOG_STD_MIN, _LOG_STD_MAX)
        return torch.distributions.Normal(mu, log_sigma.exp())

    def compute_action_and_log_prob_t(self, state_t):
        dist = self.distribution(state_t)
        a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        a_true = a * self._delta + self._central
        log_prob = dist.log_prob(a_raw).sum(dim=1)
        log_prob -= torch.log(1.0 - a.pow(2) + _EPS_LOG_PROB).sum(dim=1)
        return a_true, log_prob

    @torch.no_grad()
    def draw_action(self, state_np) -> np.ndarray:
        """Stochastic action (upstream draw_action: rsample + tanh)."""
        s = np.asarray(state_np, dtype=np.float32).reshape(1, -1)
        s_t = torch.as_tensor(s, device=self._device)
        a, _ = self.compute_action_and_log_prob_t(s_t)
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def deterministic_action_t(self, state_t) -> torch.Tensor:
        """MEAN action tanh(mu) rescaled to the action box (eval protocol)."""
        return torch.tanh(self._mu(state_t)) * self._delta + self._central


# ============================================================== agents ===
class SACAgent:
    """Protagonist: plain SAC (sac.py). Subclass AdaptiveTempSACAgent is the
    adversary. One ``fit`` call = replay add + (per update round) the exact
    upstream order: alpha pre-set (adversary) -> actor step -> alpha step
    (protagonist) -> critic target -> critic steps -> target soft update.
    """

    adaptive_temp = False  # class flag: adversary overrides

    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high,
                 device: torch.device, actor_lr: float,
                 gamma: float = _GAMMA,
                 batch_size: int = _BATCH_SIZE,
                 initial_replay_size: int = _INITIAL_REPLAY_SIZE,
                 max_replay_size: int = _MAX_REPLAY_SIZE,
                 warmup_transitions: int = _WARMUP_TRANSITIONS):
        self._device = device
        self._gamma = gamma
        self._batch_size = batch_size
        self._warmup = warmup_transitions
        self._obs_dim = obs_dim
        self._act_dim = act_dim

        self.mu = ActorNetwork(obs_dim, act_dim).to(device)
        self.sigma = ActorNetwork(obs_dim, act_dim).to(device)
        self.policy = SACPolicy(self.mu, self.sigma, act_low, act_high, device)

        self.actor_optim = optim.Adam(
            list(self.mu.parameters()) + list(self.sigma.parameters()),
            lr=actor_lr)

        critic_in = obs_dim + act_dim
        self.critics = [CriticNetwork(critic_in).to(device) for _ in range(2)]
        self.target_critics = [CriticNetwork(critic_in).to(device)
                               for _ in range(2)]
        self.critic_optims = [optim.Adam(c.parameters(), lr=_LR_CRITIC)
                              for c in self.critics]
        for c, t in zip(self.critics, self.target_critics):
            t.load_state_dict(c.state_dict())  # _init_target

        # entropy temperature: protagonist learns a scalar (log_alpha init
        # -5.30 -> alpha 0.005); adversary overwrites it per batch
        self._log_alpha = torch.tensor(_LOG_ALPHA_INIT, dtype=torch.float32,
                                       device=device)
        if not self.adaptive_temp:
            self._log_alpha.requires_grad_(True)
            self.alpha_optim = optim.Adam([self._log_alpha], lr=_LR_ALPHA)
        self._target_entropy = -float(np.prod(
            np.asarray(act_dim).reshape(-1)))  # -dim(A), upstream default

        self._replay = ReplayMemory(initial_replay_size, max_replay_size,
                                    obs_dim, act_dim)

        # update bookkeeping (smoke-test / verification)
        self.n_actor_updates = 0
        self.n_critic_updates = 0
        self.n_alpha_updates = 0
        self.n_alpha_from_temp = 0

    # -- acting ----------------------------------------------------------
    def draw_action(self, state_with_temp: np.ndarray) -> np.ndarray:
        return self.policy.draw_action(state_with_temp)

    # -- learning --------------------------------------------------------
    def _alpha_scalar_t(self) -> torch.Tensor:
        return self._log_alpha.exp()  # requires grad for the protagonist

    def _pre_update_alpha(self, s_t: torch.Tensor) -> None:
        """Adversary hook: alpha := temperature read from the batch states."""
        pass

    def _alpha_value_t(self) -> torch.Tensor:
        """No-grad alpha used in the critic target (scalar or vector)."""
        return self._log_alpha.exp().detach()

    def fit(self, dataset, n_updates: int = 1) -> None:
        self._replay.add(dataset)
        if not self._replay.initialized:
            return
        for _ in range(n_updates):
            self._fit_round()

    def _fit_round(self) -> None:
        s, a, r, ns, ab, _last = self._replay.get(self._batch_size)
        s_t = torch.as_tensor(s, device=self._device)
        a_t = torch.as_tensor(a, device=self._device)
        r_t = torch.as_tensor(r, device=self._device)
        ns_t = torch.as_tensor(ns, device=self._device)
        ab_t = torch.as_tensor(ab.astype(np.float32), device=self._device)

        self._pre_update_alpha(s_t)

        # Actor update (and protagonist alpha update) past the warmup
        if self._replay.size > self._warmup:
            a_new, log_prob = self.policy.compute_action_and_log_prob_t(s_t)
            q = torch.min(self.critics[0](s_t, a_new),
                          self.critics[1](s_t, a_new))
            loss = (self._alpha_scalar_t() * log_prob - q).mean()
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.n_actor_updates += 1
            if not self.adaptive_temp:
                alpha_loss = -(self._log_alpha *
                               (log_prob.detach() + self._target_entropy)).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.n_alpha_updates += 1

        # Critic update
        with torch.no_grad():
            a_next, log_prob_next = self.policy.compute_action_and_log_prob_t(ns_t)
            q_next = torch.min(self.target_critics[0](ns_t, a_next),
                               self.target_critics[1](ns_t, a_next))
            q_next = q_next - self._alpha_value_t() * log_prob_next
            q_next = q_next * (1.0 - ab_t)          # zero on absorbing
            target = r_t + self._gamma * q_next
        for i in range(2):
            pred = self.critics[i](s_t, a_t)
            closs = F.mse_loss(pred, target)
            self.critic_optims[i].zero_grad()
            closs.backward()
            self.critic_optims[i].step()
            self.n_critic_updates += 1

        # Soft target update
        with torch.no_grad():
            for c, t in zip(self.critics, self.target_critics):
                for pc, pt in zip(c.parameters(), t.parameters()):
                    pt.mul_(1.0 - _TAU).add_(_TAU * pc)

    # -- (de)serialisation for the periodic resume checkpoint ------------
    def state(self) -> dict:
        return {
            "mu": self.mu.state_dict(),
            "sigma": self.sigma.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "target_critics": [t.state_dict() for t in self.target_critics],
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optims": [o.state_dict() for o in self.critic_optims],
            "log_alpha": self._log_alpha.detach().cpu().clone(),
            "alpha_optim": (None if self.adaptive_temp
                            else self.alpha_optim.state_dict()),
            "replay": self._replay.state(),
            "update_counters": {"actor": self.n_actor_updates,
                                "critic": self.n_critic_updates,
                                "alpha": self.n_alpha_updates,
                                "alpha_from_temp": self.n_alpha_from_temp},
            "meta": {"obs_dim": self._obs_dim, "act_dim": self._act_dim,
                     "adaptive_temp": self.adaptive_temp},
        }

    def load(self, d: dict) -> None:
        self.mu.load_state_dict(d["mu"])
        self.sigma.load_state_dict(d["sigma"])
        for c, sd in zip(self.critics, d["critics"]):
            c.load_state_dict(sd)
        for t, sd in zip(self.target_critics, d["target_critics"]):
            t.load_state_dict(sd)
        self.actor_optim.load_state_dict(d["actor_optim"])
        for o, sd in zip(self.critic_optims, d["critic_optims"]):
            o.load_state_dict(sd)
        self._log_alpha = d["log_alpha"].to(self._device)
        if not self.adaptive_temp:
            self._log_alpha.requires_grad_(True)
            self.alpha_optim = optim.Adam([self._log_alpha], lr=_LR_ALPHA)
            if d.get("alpha_optim") is not None:
                self.alpha_optim.load_state_dict(d["alpha_optim"])
        self._replay.load(d["replay"])
        uc = d["update_counters"]
        self.n_actor_updates = int(uc["actor"])
        self.n_critic_updates = int(uc["critic"])
        self.n_alpha_updates = int(uc["alpha"])
        self.n_alpha_from_temp = int(uc["alpha_from_temp"])


class AdaptiveTempSACAgent(SACAgent):
    """Adversary: AdaptiveTempSAC — alpha is dictated per batch by the
    temperature stored in the states (log(state[:, -1])), a per-sample
    vector, never gradient-updated."""

    adaptive_temp = True

    def _pre_update_alpha(self, s_t: torch.Tensor) -> None:
        temps = s_t[:, -1]
        self._log_alpha = torch.log(temps)
        self.n_alpha_from_temp += 1

    def _alpha_scalar_t(self) -> torch.Tensor:
        # per-sample vector, no grad (broadcast against the batch)
        return self._log_alpha.exp()


# ============================================================= teacher ===
class GammaTorchDistribution:
    """Ported from torch_distributions.GammaTorchDistribution (shapes
    normalised to (1,) parameter vectors; dtype float64)."""

    def __init__(self, conc, rate, dtype=torch.float64):
        conc = np.atleast_1d(np.asarray(conc, dtype=np.float64))
        rate = np.atleast_1d(np.asarray(rate, dtype=np.float64))
        self._conc = nn.Parameter(torch.as_tensor(conc, dtype=dtype),
                                  requires_grad=True)
        self._rate = nn.Parameter(torch.as_tensor(rate, dtype=dtype),
                                  requires_grad=True)
        self.distribution_t = torch.distributions.Gamma(self._conc, self._rate)

    @staticmethod
    def from_weights(weights, dtype=torch.float64):
        return GammaTorchDistribution(weights[0], weights[1], dtype=dtype)

    def set_weights(self, weights):
        conc_t = torch.tensor([float(np.asarray(weights[0]).reshape(-1)[0])],
                               dtype=self._conc.dtype)
        rate_t = torch.tensor([float(np.asarray(weights[1]).reshape(-1)[0])],
                              dtype=self._conc.dtype)
        with torch.no_grad():
            self._conc.data = conc_t
            self._rate.data = rate_t
        self.distribution_t = torch.distributions.Gamma(self._conc, self._rate)

    def get_weights(self):
        return np.concatenate([self._conc.data.detach().cpu().numpy(),
                               self._rate.data.detach().cpu().numpy()])

    def log_pdf_t(self, x):
        return self.distribution_t.log_prob(x)

    def mean(self):
        return self.distribution_t.mean.detach().cpu().numpy()

    def sample(self):
        return self.distribution_t.rsample()

    def parameters(self):
        return [self._conc, self._rate]


class SelfPacedGammaTemperatureTeacherFixedBeta:
    """Ported (near-verbatim) from teachers/self_paced_gamma_teacher_fixed_beta.py."""

    def __init__(self, initial_conc, target_conc, rate, max_kl,
                 performance_lower_bound, conc_lower_bound,
                 target_kl_threshold, temperature_bounds, error_dir="."):
        self.temp_dist = GammaTorchDistribution(initial_conc, rate,
                                                dtype=torch.float64)
        self.target_dist = GammaTorchDistribution(target_conc, rate,
                                                  dtype=torch.float64)
        self.max_kl = max_kl
        self.perf_lb = performance_lower_bound
        self.above_perf_lb = False
        self.conc_lb = conc_lower_bound
        self.target_kl_threshold = target_kl_threshold
        self.temp_bounds = temperature_bounds
        self.rate = np.atleast_1d(np.asarray(rate, dtype=np.float64))
        self.iteration = 0
        self._error_dir = error_dir
        self.conc_temperature_data = [self._param(self.temp_dist, 0)]
        self.rate_temperature_data = [self._param(self.temp_dist, 1)]

    @staticmethod
    def _param(dist, idx):
        return float(dist.parameters()[idx].data.detach().cpu().numpy().reshape(-1)[0])

    def compute_temp_kl(self, old_temp_dist):
        return torch.distributions.kl.kl_divergence(
            self.temp_dist.distribution_t, old_temp_dist.distribution_t)

    def compute_target_temp_kl(self):
        kl_div = torch.distributions.kl.kl_divergence(
            self.temp_dist.distribution_t, self.target_dist.distribution_t)
        return kl_div.detach().cpu().numpy()

    def compute_expected_performance(self, dist, temp_t, old_temp_log_prob_t,
                                     temp_value_t):
        temp_ratio_t = torch.exp(dist.log_pdf_t(temp_t) - old_temp_log_prob_t)
        return torch.mean(temp_ratio_t * temp_value_t)

    def update_temperature_distribution(self, temperatures, values,
                                        training_iteration):
        # self.iteration and training_iteration are different (upstream note)
        self.iteration += 1

        old_temp_dist = GammaTorchDistribution.from_weights(
            self.temp_dist.get_weights(), dtype=torch.float64)

        temps_t = torch.as_tensor(np.asarray(temperatures, dtype=np.float64))
        old_temp_log_prob_t = old_temp_dist.log_pdf_t(temps_t).detach()
        temp_value_t = torch.as_tensor(np.asarray(values, dtype=np.float64))

        # KL constraint
        def kl_constraint_func(x_conc):
            dist = GammaTorchDistribution.from_weights(
                [x_conc, self.rate], dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(
                dist.distribution_t, old_temp_dist.distribution_t)
            return kl_div.detach().cpu().numpy()

        def kl_constraint_grad_func(x_conc):
            dist = GammaTorchDistribution.from_weights(
                [x_conc, self.rate], dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(
                dist.distribution_t, old_temp_dist.distribution_t)
            conc_grad = torch.autograd.grad(kl_div, dist.parameters()[:1])
            return np.array([conc_grad[0].detach().cpu().numpy()])

        kl_constraint = NonlinearConstraint(
            kl_constraint_func, -np.inf, self.max_kl,
            jac=kl_constraint_grad_func, keep_feasible=True)

        # performance constraint
        def perf_constraint_func(x_conc):
            dist = GammaTorchDistribution.from_weights(
                [x_conc, self.rate], dtype=torch.float64)
            perf = self.compute_expected_performance(
                dist, temps_t, old_temp_log_prob_t, temp_value_t)
            return perf.detach().cpu().numpy()

        def perf_constraint_grad_func(x_conc):
            dist = GammaTorchDistribution.from_weights(
                [x_conc, self.rate], dtype=torch.float64)
            perf = self.compute_expected_performance(
                dist, temps_t, old_temp_log_prob_t, temp_value_t)
            conc_grad = torch.autograd.grad(perf, dist.parameters()[:1])
            return np.array([conc_grad[0].detach().cpu().numpy()])

        perf_constraint = NonlinearConstraint(
            perf_constraint_func, self.perf_lb, np.inf,
            jac=perf_constraint_grad_func, keep_feasible=True)

        if self.compute_target_temp_kl() > self.target_kl_threshold:
            cones = np.ones_like(self.temp_dist.get_weights()[:1])
            lb = self.conc_lb
            ub = np.inf * cones.copy()
            bounds = Bounds(lb, ub, keep_feasible=True)
            x0_conc = np.clip(self.temp_dist.get_weights()[:1].copy(), lb, ub)
        else:
            x0_conc = self.temp_dist.get_weights()[:1].copy()
            bounds = None

        try:
            if kl_constraint_func(x0_conc) >= self.max_kl:
                print("Warning! KL-Bound of x0 violates constraint already",
                      flush=True)

            if perf_constraint_func(x0_conc) >= self.perf_lb:
                print("Optimizing KL divergence", flush=True)
                self.above_perf_lb = True
                constraints = [kl_constraint, perf_constraint]

                def objective(x_conc):
                    dist = GammaTorchDistribution.from_weights(
                        [x_conc, self.rate], dtype=torch.float64)
                    kl_div = torch.distributions.kl.kl_divergence(
                        dist.distribution_t, self.target_dist.distribution_t)
                    conc_grad = torch.autograd.grad(kl_div,
                                                    dist.parameters()[:1])
                    return kl_div.detach().cpu().numpy(), np.array(
                        conc_grad[0].detach().cpu().numpy()).astype(np.float64)

                res = minimize(
                    objective, x0_conc, method="trust-constr", jac=True,
                    bounds=bounds, constraints=constraints,
                    options={"gtol": _G_TOL, "xtol": _X_TOL})

            # only move the distribution if the performance threshold has
            # been exceeded at least once (upstream comment)
            elif not self.above_perf_lb:
                print("Distribution frozen", flush=True)
                res = None
            else:
                res = None
        except Exception:  # noqa: BLE001 - upstream swallows optimiser errors
            import pickle
            os.makedirs(self._error_dir, exist_ok=True)
            with open(os.path.join(self._error_dir,
                                   "error_" + str(time.time())), "wb") as f:
                pickle.dump((self.temp_dist.get_weights(), temperatures,
                             values), f)
            print("Exception occurred during optimization! Storing state and "
                  "keeping old values!", flush=True)
            res = None

        if res is not None and res.success:
            self.temp_dist.set_weights([res.x[0], self.rate[0]])
        elif res is not None:
            # not a success, but if the objective improved and the
            # constraints still hold, use the result anyway (upstream intent;
            # see port note 5 — upstream compares against the objective tuple)
            old_f = objective(self.temp_dist.get_weights()[:1])[0]
            cons_ok = True
            for constraint in constraints:
                cons_ok = (cons_ok and constraint.lb <= constraint.fun(res.x)
                           <= constraint.ub)
            rate_ok = bounds is None or (np.all(bounds.lb <= res.x)
                                         and np.all(res.x <= bounds.ub))
            if cons_ok and rate_ok and res.fun < old_f:
                self.temp_dist.set_weights([res.x[0], self.rate[0]])
            else:
                print("Warning! Context optimization unsuccessful - will "
                      "keep old values. Message: %s" % res.message, flush=True)

        # data collection (upstream padding logic)
        self.conc_temperature_data.extend(
            [self.conc_temperature_data[-1]]
            * (training_iteration - len(self.conc_temperature_data)))
        self.conc_temperature_data.append(self._param(self.temp_dist, 0))
        self.rate_temperature_data.extend(
            [self.rate_temperature_data[-1]]
            * (training_iteration - len(self.rate_temperature_data)))
        self.rate_temperature_data.append(self._param(self.temp_dist, 1))

    def sample(self):
        sample = self.temp_dist.sample().detach().cpu().numpy()
        sample = np.clip(sample, self.temp_bounds[0], self.temp_bounds[1])
        return float(np.asarray(sample).reshape(-1)[0])

    def state(self) -> dict:
        return {
            "weights": [float(x) for x in self.temp_dist.get_weights()],
            "target_weights": [float(x) for x in self.target_dist.get_weights()],
            "above_perf_lb": bool(self.above_perf_lb),
            "iteration": int(self.iteration),
            "conc_data": list(self.conc_temperature_data),
            "rate_data": list(self.rate_temperature_data),
        }

    def load(self, d: dict) -> None:
        self.temp_dist.set_weights(d["weights"])
        self.target_dist.set_weights(d["target_weights"])
        self.above_perf_lb = bool(d["above_perf_lb"])
        self.iteration = int(d["iteration"])
        self.conc_temperature_data = list(d["conc_data"])
        self.rate_temperature_data = list(d["rate_data"])


class Buffer:
    """Ported from utils/buffer.py (n parallel FIFO lists, max size)."""

    def __init__(self, n_buffers: int, max_buffer_size: int,
                 reset_on_query: bool):
        self.reset_on_query = reset_on_query
        self.max_buffer_size = max_buffer_size
        self.buffers = [[] for _ in range(n_buffers)]

    def update_buffer(self, datas):
        if isinstance(datas[0], list):
            for buffer, data in zip(self.buffers, datas):
                buffer.extend(data)
        else:
            for buffer, data in zip(self.buffers, datas):
                buffer.append(data)
        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def read_buffer(self, reset=None):
        if reset is None:
            reset = self.reset_on_query
        res = tuple(self.buffers)
        if reset:
            self.buffers = [[] for _ in range(len(self.buffers))]
        return res

    def __len__(self):
        return len(self.buffers[0])


def _sample_n_temperatures(teacher, n) -> np.ndarray:
    """Ported from QARLExperiment.sample_n_temperatures."""
    sampled = np.zeros(n)
    for i in range(n):
        sampled[i] = teacher.sample()
    return sampled


# ==================================================== eval adapter =======
class QARLEvalAdapter:
    """Frozen-evaluation adapter: deterministic protagonist SAC MEAN action
    with the temperature observation input fixed at 0.0 (the QARL official
    no-adversary evaluation default) and NO adversary.

    Stateless: ``reset_episodes`` is a no-op; actions are computed for every
    row (the evaluator only accumulates rows whose ``active`` flag is set).
    """

    def __init__(self, mu_net: ActorNetwork, obs_dim_with_temp: int,
                 act_dim: int, act_low: np.ndarray, act_high: np.ndarray):
        self._mu = mu_net.cpu().eval()
        self._obs_dim = obs_dim_with_temp
        self._act_dim = act_dim
        self._delta = torch.tensor(
            0.5 * (np.asarray(act_high) - np.asarray(act_low)),
            dtype=torch.float32)
        self._central = torch.tensor(
            0.5 * (np.asarray(act_high) + np.asarray(act_low)),
            dtype=torch.float32)

    def act(self, obs: np.ndarray, active: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        n = obs.shape[0]
        # temperature input = 0 (evaluation default, experiment protocol)
        temp = np.zeros((n, 1), dtype=np.float32)
        x = torch.from_numpy(np.concatenate([obs, temp], axis=1))
        with torch.no_grad():
            a = torch.tanh(self._mu(x)) * self._delta + self._central
        return a.numpy().astype(np.float32)

    def reset_episodes(self, mask) -> None:  # protocol no-op
        pass


def load_eval_adapter(job_dir: str, task: str, seed: int, cfg: dict):
    """Build the frozen-eval adapter from job_dir/ckpt/final.pt."""
    path = os.path.join(job_dir, "ckpt", "final.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"QARL checkpoint not found: {path} (run train() first)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    if meta["task"] != task:
        print(f"[qarl] warning: checkpoint task {meta['task']} != {task}")
    prot = ckpt["prot"]
    mu = ActorNetwork(int(prot["obs_dim_with_temp"]), int(prot["act_dim"]))
    mu.load_state_dict(prot["mu"])
    return QARLEvalAdapter(mu, int(prot["obs_dim_with_temp"]),
                           int(prot["act_dim"]), prot["act_low"],
                           prot["act_high"])


# ======================================================== training =======
def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def train(job_dir: str, task: str, seed: int, cfg: dict) -> None:
    """Train QARL to cfg["steps"] real env transitions (see module docstring
    for the ported semantics and the counters/progress conventions)."""
    t0 = time.time()
    os.makedirs(os.path.join(job_dir, "ckpt"), exist_ok=True)
    final_path = os.path.join(job_dir, "ckpt", "final.pt")
    state_path = os.path.join(job_dir, "ckpt", "state.pt")
    counters_path = os.path.join(job_dir, "counters.json")
    progress_path = os.path.join(job_dir, "progress.jsonl")

    steps_total = int(cfg["steps"])
    gamma = float(cfg.get("gamma", _GAMMA))
    utd = int(cfg.get("utd", 1))
    n_epi = int(cfg.get("n_episodes_per_iteration", 30))
    n_iter_per_agent = int(cfg.get("n_iterations_per_agent",
                                   _N_ITERATIONS_PER_AGENT))
    n_samples_for_update = int(cfg.get("n_samples_for_dist_update", 30))
    eval_interval = int(cfg.get("eval_interval", _EVAL_INTERVAL))
    eval_episodes = int(cfg.get("eval_episodes", _EVAL_EPISODES))
    initial_replay_size = int(cfg.get("initial_replay_size",
                                      _INITIAL_REPLAY_SIZE))
    warmup_transitions = int(cfg.get("warmup_transitions",
                                     _WARMUP_TRANSITIONS))
    plb = cfg["performance_lower_bound"]
    if isinstance(plb, dict):
        plb = float(plb[task])

    # idempotent completion: budget already reached -> return immediately
    if os.path.exists(final_path) and os.path.exists(counters_path):
        try:
            with open(counters_path) as f:
                prev = json.load(f)
            if int(prev.get("real_transitions", 0)) >= steps_total:
                print(f"[qarl] {task} seed{seed}: budget already reached "
                      f"({prev['real_transitions']} >= {steps_total}), "
                      "returning", flush=True)
                return
        except (ValueError, OSError):
            pass

    device = torch.device(cfg.get("device") or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[qarl] task={task} seed={seed} steps={steps_total} "
          f"device={device}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- environment (both players act in the same transition) --------
    bodies = tuple(cfg.get("adversary_bodies", ADVERSARY_BODIES)[task])
    axes = tuple(cfg.get("adversary_axes", ("x", "z")))
    force = float(cfg.get("adversary_force", ADVERSARY_FORCE))
    # env_mode "perturbed" (phase B, shared sampler) | "nominal" (phase A,
    # RMBPO-paper protocol: default v4 physics, adversary only)
    if cfg.get("env_mode", "perturbed") == "nominal":
        import gymnasium as _gym
        base_env = _gym.make({"hopper": "Hopper-v4", "halfcheetah": "HalfCheetah-v4",
                              "walker2d": "Walker2d-v4"}[task],
                             disable_env_checker=True)
    else:
        base_env = make_train_env(task, seed)
    env = TwoPlayerForceEnv(base_env, bodies, axes, force)

    base_obs_dim = int(env.observation_space.shape[0])
    obs_dim = base_obs_dim + 1                       # + temperature input
    p_low = np.asarray(env.action_space.low, dtype=np.float64)
    p_high = np.asarray(env.action_space.high, dtype=np.float64)
    adv_space = env.adversary_action_space
    a_low = np.asarray(adv_space.low, dtype=np.float64)
    a_high = np.asarray(adv_space.high, dtype=np.float64)

    # ---- agents --------------------------------------------------------
    prot = SACAgent(obs_dim, int(env.action_space.shape[0]), p_low, p_high,
                    device, actor_lr=float(cfg["actor_lr"]), gamma=gamma,
                    initial_replay_size=initial_replay_size,
                    warmup_transitions=warmup_transitions)
    adv = AdaptiveTempSACAgent(obs_dim, int(adv_space.shape[0]), a_low,
                               a_high, device,
                               actor_lr=float(cfg["actor_lr"]), gamma=gamma,
                               initial_replay_size=initial_replay_size,
                               warmup_transitions=warmup_transitions)

    # ---- teacher + temp/performance buffer ----------------------------
    teacher = SelfPacedGammaTemperatureTeacherFixedBeta(
        cfg["initial_conc"], cfg["target_conc"], cfg["fixed_rate"],
        cfg["max_kl"], plb, cfg["conc_lower_bound"],
        cfg["target_kl_threshold"],
        cfg.get("temperature_bounds", [-np.inf, np.inf]),
        error_dir=os.path.join(job_dir, "optimization_errors"))
    temp_perf_buffer = Buffer(2, 100, reset_on_query=True)

    # ---- loop state ----------------------------------------------------
    counters = {"real_transitions": 0, "protagonist_steps": 0,
                "adversary_steps": 0, "temperature_samples": 0}
    iteration = 0
    last_eval_step = 0
    n_evals = 0
    sampled_temp_means: list = []
    seeded_env = False

    # ---- resume from the periodic iteration-boundary checkpoint -------
    if os.path.exists(state_path):
        try:
            st = torch.load(state_path, map_location=device,
                            weights_only=False)
            prot.load(st["prot"])
            adv.load(st["adv"])
            teacher.load(st["teacher"])
            counters = {k: int(v) for k, v in st["counters"].items()}
            iteration = int(st["iteration"])
            last_eval_step = int(st["last_eval_step"])
            n_evals = int(st["n_evals"])
            sampled_temp_means = list(st["sampled_temp_means"])
            tb, vb = st["temp_perf_buffer"]
            temp_perf_buffer.buffers = [list(tb), list(vb)]
            torch.set_rng_state(st["rng"]["torch"].cpu())
            np.random.set_state(st["rng"]["np"])
            try:
                env.unwrapped._np_random.bit_generator.state = \
                    st["rng"]["env"]
                env.env._sampler.rng.bit_generator.state = \
                    st["rng"]["sampler"]
            except Exception:  # noqa: BLE001 - best-effort RNG restore
                pass
            seeded_env = True
            print(f"[qarl] resumed from {state_path} at "
                  f"{counters['real_transitions']} transitions "
                  f"(iteration {iteration})", flush=True)
        except Exception as e:  # noqa: BLE001 - corrupt state -> fresh
            print(f"[qarl] could not resume ({e}); starting fresh",
                  flush=True)
            counters = {"real_transitions": 0, "protagonist_steps": 0,
                        "adversary_steps": 0, "temperature_samples": 0}
            iteration = 0
            torch.manual_seed(seed)
            np.random.seed(seed)

    def _append_progress(step, r_adv, r_no, mean_temp):
        line = {"step": int(step),
                "mean_return_vs_adv": (None if r_adv is None
                                       else float(r_adv)),
                "mean_return_no_adv": (None if r_no is None
                                       else float(r_no)),
                "mean_temp": float(mean_temp),
                "wall_s": time.time() - t0}
        with open(progress_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    def _progress_eval(with_adversary: bool, evaluation_temp: float,
                       n_episodes: int) -> float:
        """evaluate_vs_adversary / evaluate_without_adversary replica:
        stochastic draw_action for both players, evaluation temperature =
        teacher mean, zero-force adversary for the no-adversary arm,
        undiscounted return. Fresh deterministic eval env per call."""
        eseed = seed + 100_000 + n_evals_
        if cfg.get("env_mode", "perturbed") == "nominal":
            import gymnasium as _gym
            e_base = _gym.make({"hopper": "Hopper-v4", "halfcheetah": "HalfCheetah-v4",
                                "walker2d": "Walker2d-v4"}[task],
                               disable_env_checker=True)
        else:
            e_base = make_train_env(task, eseed)
        eenv = TwoPlayerForceEnv(e_base, bodies, axes, force)
        returns = []
        for ep in range(n_episodes):
            obs, _ = eenv.reset(seed=eseed + ep)
            done = False
            steps = 0
            ret = 0.0
            while not done and steps < 1000:
                s = np.append(np.asarray(obs, dtype=np.float64),
                              evaluation_temp)
                a_p = prot.draw_action(s)
                if with_adversary:
                    a_a = adv.draw_action(s)
                else:
                    a_a = np.zeros(int(adv_space.shape[0]),
                                   dtype=np.float32)
                obs, r, term, trunc, _ = eenv.step(a_p, a_a)
                ret += float(r)
                steps += 1
                done = bool(term or trunc)
            returns.append(ret)
        eenv.close()
        return float(np.mean(returns))

    # mutable eval counter captured by _progress_eval
    n_evals_ = n_evals

    def _run_block(learner: SACAgent, other: SACAgent,
                   temps: np.ndarray, iter_temps: list,
                   iter_returns: list) -> None:
        """One core.learn call: n_episodes episodes in which only `learner`
        fits (and only its replay grows) while `other` keeps acting. Episode
        k of the block uses temps[k] (upstream preprocessor episode index).
        The budget may cut the block mid-episode; the partial episode still
        contributes a (temp, J) pair, as upstream's dataset-tail handling."""
        nonlocal seeded_env
        n_eps = len(temps)
        ep = 0
        obs = None
        temp = float(temps[0])
        ep_j = 0.0
        ep_k = 0
        while ep < n_eps and counters["real_transitions"] < steps_total:
            if obs is None:
                kw = {} if seeded_env else {"seed": seed}
                obs, _ = env.reset(**kw)
                seeded_env = True
                temp = float(temps[ep])
            s = np.append(np.asarray(obs, dtype=np.float64), temp)
            a_p = prot.draw_action(s)
            a_a = adv.draw_action(s)
            nobs, r, term, trunc, _ = env.step(a_p, a_a)
            counters["real_transitions"] += 1
            counters["protagonist_steps"] += 1
            counters["adversary_steps"] += 1
            last = bool(term or trunc)
            r = float(r)
            ns = np.append(np.asarray(nobs, dtype=np.float64), temp)
            if learner is adv:
                own = (s, a_a, -r, ns, bool(term), last)
            else:
                own = (s, a_p, r, ns, bool(term), last)
            learner.fit([own], n_updates=utd)
            ep_j += gamma ** ep_k * r
            ep_k += 1
            obs = nobs
            if last:
                iter_temps.append(temp)
                iter_returns.append(ep_j)
                ep += 1
                ep_j = 0.0
                ep_k = 0
                obs = None
        if ep_k > 0:  # budget cut mid-episode -> partial episode pair
            iter_temps.append(temp)
            iter_returns.append(ep_j)

    def _save_state() -> None:
        torch.save({
            "prot": prot.state(),
            "adv": adv.state(),
            "teacher": teacher.state(),
            "counters": counters,
            "iteration": iteration,
            "last_eval_step": last_eval_step,
            "n_evals": n_evals_,
            "sampled_temp_means": sampled_temp_means,
            "temp_perf_buffer": [list(temp_perf_buffer.buffers[0]),
                                 list(temp_perf_buffer.buffers[1])],
            "rng": {
                "torch": torch.get_rng_state(),
                "np": np.random.get_state(),
                "env": env.unwrapped._np_random.bit_generator.state,
                "sampler": (env.env._sampler.rng.bit_generator.state
                             if hasattr(getattr(env.env, "_sampler", None), "rng")
                             else None),
            },
        }, state_path)

    def _save_final() -> None:
        def _agent_final(agent):
            return {
                "mu": agent.mu.state_dict(),
                "sigma": agent.sigma.state_dict(),
                "critics": [c.state_dict() for c in agent.critics],
                "target_critics": [t.state_dict()
                                   for t in agent.target_critics],
                "log_alpha": float(agent._log_alpha.reshape(-1)[0].item())
                if agent._log_alpha.numel() == 1
                else None,
                "update_counters": {"actor": agent.n_actor_updates,
                                    "critic": agent.n_critic_updates,
                                    "alpha": agent.n_alpha_updates,
                                    "alpha_from_temp":
                                        agent.n_alpha_from_temp},
                "act_dim": int(agent._act_dim),
            }
        torch.save({
            "format": "mujoco_baselines.algos.qarl.final.v1",
            "meta": {"task": task, "seed": int(seed),
                     "obs_dim_base": base_obs_dim,
                     "obs_dim_with_temp": obs_dim,
                     "iteration": iteration,
                     "counters": dict(counters)},
            "prot": {**_agent_final(prot), "obs_dim_with_temp": obs_dim,
                     "act_low": p_low, "act_high": p_high},
            "adv": {**_agent_final(adv), "obs_dim_with_temp": obs_dim,
                    "act_low": a_low, "act_high": a_high},
            "teacher": teacher.state(),
            "sampled_temp_means": sampled_temp_means,
            "cfg_echo": {"actor_lr": cfg["actor_lr"], "utd": utd,
                         "gamma": gamma,
                         "n_episodes_per_iteration": n_epi,
                         "steps": steps_total},
        }, final_path)

    # ==================================================================
    # main loop (upstream: per iteration — temperature sampling, adversary
    # block, protagonist block, evaluation, teacher update)
    # ==================================================================
    while counters["real_transitions"] < steps_total:
        iteration += 1
        sampled = _sample_n_temperatures(teacher, n_epi)
        counters["temperature_samples"] += len(sampled)
        iter_temps: list = []
        iter_returns: list = []

        # Optimization of adversary (upstream order), then protagonist
        for _ in range(n_iter_per_agent):
            _run_block(adv, prot, sampled, iter_temps, iter_returns)
            if counters["real_transitions"] >= steps_total:
                break
        for _ in range(n_iter_per_agent):
            if counters["real_transitions"] >= steps_total:
                break
            _run_block(prot, adv, sampled, iter_temps, iter_returns)

        # Update of the temperature distribution from the temp/perf buffer
        if iter_temps:
            temp_perf_buffer.update_buffer([iter_temps, iter_returns])
        if len(temp_perf_buffer) >= n_samples_for_update:
            buf_t, buf_v = temp_perf_buffer.read_buffer(reset=True)
            teacher.update_temperature_distribution(
                np.asarray(buf_t, dtype=np.float64),
                np.asarray(buf_v, dtype=np.float64), iteration - 1)

        sampled_temp_means.append(float(np.mean(sampled)))

        # evaluation of the iteration (progress logging only)
        budget_done = counters["real_transitions"] >= steps_total
        do_eval = budget_done or (
            counters["real_transitions"] - last_eval_step >= eval_interval)
        r_adv = r_no = None
        if do_eval:
            evaluation_temp = float(np.asarray(teacher.temp_dist.mean())
                                    .reshape(-1)[0])
            n_evals_ += 1
            r_adv = _progress_eval(True, evaluation_temp, eval_episodes)
            n_evals_ += 1
            r_no = _progress_eval(False, evaluation_temp, eval_episodes)
            last_eval_step = counters["real_transitions"]
            print(f"[qarl] it={iteration} step={counters['real_transitions']} "
                  f"temp_mean={sampled_temp_means[-1]:.5f} "
                  f"R_vs_adv={r_adv:.1f} R_no_adv={r_no:.1f} "
                  f"conc={teacher._param(teacher.temp_dist, 0):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        _append_progress(counters["real_transitions"], r_adv, r_no,
                         sampled_temp_means[-1])

        _write_json(counters_path, counters)
        _save_state()

    # final checkpoint + counters (budget exactly reached)
    _save_final()
    _write_json(counters_path, counters)
    print(f"[qarl] done: task={task} seed={seed} "
          f"real_transitions={counters['real_transitions']} "
          f"iterations={iteration} "
          f"teacher_conc={teacher._param(teacher.temp_dist, 0):.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    env.close()
