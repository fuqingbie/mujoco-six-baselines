"""RMA (Rapid Motor Adaptation) two-phase baseline — port of SPARC's rma_phase1/rma_phase2.

Port provenance: upstreams/sparc @ 53f1e2e9eb2d88de84bce3a82453b2608ed2a0bb
(github bramgrooten/sparc, arxiv 2511.09737).  Phase 1 trains a privileged
"expert" policy with QR-SAC on context=(torso-mass multiplier, friction
multiplier); phase 2 freezes the expert and trains a history adapter that
regresses the expert's context-encoder latent from a rolling
history_length-step (obs, action) history by representation MSE.

Ported vs reconstructed:
  * utils/modules.py   -> ExpertPolicy / AdapterPolicy / CriticContext /
    HistoryAdapter / ContextEncoder: ported essentially verbatim (same layer
    names, sizes, activation order, tanh-squashed Gaussian action heads with
    the 1e-6 log-prob correction, action_scale/action_bias buffers); only the
    constructors take explicit dims/bounds instead of a gym.vector env.
  * utils/replay_buffer_dict.py -> ReplayBufferExpert / ReplayBufferDict:
    ported verbatim (numpy circular buffers, np.random.randint sampling).
  * algorithms/rma_phase1.py / rma_phase2.py -> the two training loops below:
    ported structure (update cadence, entropy autotune, policy_frequency
    double-update, target-network frequency 1, critic grad clip 10,
    learning_starts random actions in phase 1, adapter-policy rollouts from
    step 0 in phase 2, MSE regression gated at the same learning_starts).
  * algorithms/qr_sac.py is MISSING from the pinned upstream tree (see
    upstreams.lock.json) — `_quantile_huber_loss` and `min_critic_quantiles`
    are reconstructed here per the paper §4.1 and the call sites in
    rma_phase1.py / sparc.py (QR critics with N=num_quantiles heads,
    tau_hat=(i+0.5)/N, kappa=1.0, elementwise min over the two critics,
    policy loss on the min-critics' mean quantile value; see the docstrings
    for the exact formulation and the one upstream shape bug fixed).

Environment differences vs SPARC (per task §2):
  * MuJoCo v4 PerturbedEnv (torso-mass + friction multipliers from the shared
    PhysSampler U[0.9,1.1]) instead of the v5 wind envs; the context is 2-dim
    either way.
  * The rolling history is maintained by the training loop (the wind env did
    it inside step()); the update convention is mirrored EXACTLY:
        reset:      history = zeros
        after step: cur_history[:-1] = cur_history[1:]
                    cur_history[-1] = concat(next_obs, action)   # POST-step obs
    so the row appended at step t is (obs_{t+1}, a_t) and the policy at step
    t conditions on the history BEFORE its own action is appended (its last
    row is (obs_t, a_{t-1})).
  * Single non-vectorized env (num_envs=1 like SPARC's default); no
    auto-reset, so episode boundaries are handled explicitly; on
    truncation the terminal observation is stored with done=terminations
    (same as SPARC's final_observation handling).

Budget accounting: phase1_steps + phase2_steps real env steps total
(learning_starts random-action prefill counts inside phase 1).
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..envs.perturbed import make_train_env

# sparc utils/constants.py
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# sparc utils/args.py defaults not carried in config.RMA
LOG_FREQ = 1000            # progress.jsonl cadence (steps)
SAVE_INTERVAL = 200_000    # mid-phase resume checkpoint cadence (steps)


# --------------------------------------------------------------------------- #
# QR-SAC reconstruction (upstream algorithms/qr_sac.py is missing from the    #
# pinned tree; signatures/semantics pinned by the call sites in rma_phase1.py #
# and sparc.py, formulation per paper §4.1 / Dabney et al. 2017).             #
# --------------------------------------------------------------------------- #
def _quantile_huber_loss(q_pred, target_q, tau_hat, kappa=1.0):
    """Quantile-regression (pinball) Huber loss.

    q_pred:   (B, N) predicted quantile values, head i estimates quantile
              level tau_hat[i].
    target_q: (B, N) sample-based target quantiles (one sample per column;
              the target distribution is represented by its N samples).
    tau_hat:  (N,) quantile mid-points (i+0.5)/N.

    Pairwise differences u[b, i, j] = target_q[b, j] - q_pred[b, i];
    rho^kappa_{tau_i}(u) = |tau_i - 1{u<0}| * Huber_kappa(u);
    loss = E_b[ (1/N) sum_j sum_i rho ]  (mean over the N target samples,
    summed over the N predicted quantile heads, averaged over the batch) —
    the standard sample-based quantile regression objective.

    NOTE: upstream rma_phase1.py calls this with `batch_rewards.unsqueeze(1)`
    where its replay buffer already returns (B, 1) rewards, which would
    broadcast the QR target to (B, B, N); the intended math (and the plain
    'sac' branch of the same file) is unambiguous, so here the target is
    always shaped (B, N) by construction.
    """
    # (B, N_pred, 1) - (B, 1, N_target) -> (B, N_pred, N_target)
    u = target_q.detach().unsqueeze(1) - q_pred.unsqueeze(2)
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa,
                        0.5 * u.pow(2),
                        kappa * (abs_u - 0.5 * kappa))
    # |tau_hat[i] - 1{u[b,i,j] < 0}| broadcast over (B, N_pred, N_target)
    weight = (tau_hat.view(1, -1, 1) - (u < 0).float()).abs()
    # sum over predicted heads, mean over target samples and batch
    return (huber * weight).sum(dim=1).mean()


def min_critic_quantiles(q1, q2, top_quantiles_to_drop=0):
    """Elementwise min over the two QR critics' quantile outputs.

    q1, q2: (B, N). With top_quantiles_to_drop=0 (config default, and the
    only case exercised here) this is torch.min(q1, q2): per quantile head,
    the smaller of the two critics — "min over critics of quantiles".
    For top_quantiles_to_drop=k > 0 each critic's k highest quantile VALUES
    are dropped first (sort descending, keep the N-k lowest), mirroring the
    TQC-style risk-averse drop the upstream arg documents.
    """
    if top_quantiles_to_drop and top_quantiles_to_drop > 0:
        k = int(top_quantiles_to_drop)
        n = q1.shape[-1]
        assert 0 < k < n, "top_quantiles_to_drop must be in (0, N)"
        q1 = torch.sort(q1, dim=-1, descending=True).values[..., n - k:]
        q2 = torch.sort(q2, dim=-1, descending=True).values[..., n - k:]
    return torch.min(q1, q2)


# --------------------------------------------------------------------------- #
# Networks (ported verbatim from sparc utils/modules.py; constructors take    #
# explicit dims/bounds instead of a gym.vector env — layer names and state-   #
# dict keys are identical to upstream so expert.pt/phase-2 copies interop).  #
# --------------------------------------------------------------------------- #
class ContextEncoder(nn.Module):
    """Expert context encoder (sparc utils/modules.py, verbatim)."""

    def __init__(self, context_dim: int):
        super().__init__()
        self.enc1 = nn.Linear(context_dim, 32)
        self.enc2 = nn.Linear(32, 32)

    def forward(self, context):
        x = F.relu(self.enc1(context))
        z = F.relu(self.enc2(x))
        return z


class HistoryAdapter(nn.Module):
    """RMA history adapter (sparc utils/modules.py, verbatim).

    Input (B, history_length, obs_dim + action_dim) -> (B, 32) latent that
    regresses the expert ContextEncoder output.
    """

    def __init__(self, history_length: int, obs_dim: int, action_dim: int):
        super().__init__()
        self.history_length = history_length

        self.fc1 = nn.Linear(obs_dim + action_dim, 32)
        big_stride = 4 if history_length >= 10 else 1
        big_pad = 3 if history_length >= 10 else 1
        self.conv1 = nn.Conv1d(32, 32, kernel_size=8, stride=big_stride,
                               padding=big_pad, padding_mode="replicate")
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2,
                               padding_mode="replicate")
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2,
                               padding_mode="replicate")
        flat_dim = 32 * math.ceil(history_length / big_stride)
        self.flat = nn.Flatten()
        self.fc2 = nn.Linear(flat_dim, 32)

    def forward(self, history):
        x = F.relu(self.fc1(history))      # (B, H, 32)
        x = x.permute(0, 2, 1)             # (B, 32, H)
        x = F.relu(self.conv1(x))          # (B, 32, ceil(H/4))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = F.relu(self.fc2(x))            # (B, 32)
        return x


def _action_scale_bias(action_low, action_high):
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    scale = torch.tensor((high - low) / 2.0, dtype=torch.float32)
    bias = torch.tensor((high + low) / 2.0, dtype=torch.float32)
    return scale, bias


class ExpertPolicy(nn.Module):
    """Privileged policy: obs encoder + context encoder -> decision (verbatim).

    forward(obs, context) -> (mean, log_std); get_action returns the
    tanh-squashed reparameterized sample, its log-prob, and the deterministic
    mean action, exactly as sparc utils/modules.py.
    """

    def __init__(self, obs_dim: int, context_dim: int, action_dim: int,
                 action_low, action_high):
        super().__init__()
        # observation encoder
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # context encoder
        self.encoder = ContextEncoder(context_dim)
        # decision layers
        self.fc3 = nn.Linear(256 + 32, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        scale, bias = _action_scale_bias(action_low, action_high)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(self, obs, context):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        z = self.encoder(context)
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs, context):
        mean, log_std = self(obs, context)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class AdapterPolicy(nn.Module):
    """RMA adapter policy: expert's obs encoder + HistoryAdapter (verbatim)."""

    def __init__(self, obs_dim: int, action_dim: int, history_length: int,
                 action_low, action_high):
        super().__init__()
        # observation encoder (copied from the frozen expert in phase 2)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # history adapter
        self.adapter = HistoryAdapter(history_length, obs_dim, action_dim)
        # decision layers (copied from the frozen expert in phase 2)
        self.fc3 = nn.Linear(256 + 32, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        scale, bias = _action_scale_bias(action_low, action_high)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(self, obs, history):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        z = self.adapter(history)
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs, history):
        mean, log_std = self(obs, history)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class CriticContext(nn.Module):
    """QR critic on (obs, action, context) (sparc utils/modules.py, verbatim).

    Outputs num_quantiles values (N=1 degenerates to plain SAC's scalar).
    """

    def __init__(self, obs_dim: int, action_dim: int, context_dim: int,
                 num_quantiles: int = 1):
        super().__init__()
        # observation-action encoder
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # context encoder
        self.enc1 = nn.Linear(context_dim, 32)
        self.enc2 = nn.Linear(32, 32)
        # decision layers
        self.fc3 = nn.Linear(256 + 32, 256)
        self.fc4 = nn.Linear(256, num_quantiles)
        self.num_quantiles = num_quantiles

    def forward(self, obs, act, context):
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = F.relu(self.enc1(context))
        z = F.relu(self.enc2(z))
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc3(x))
        q = self.fc4(x)
        return q


# --------------------------------------------------------------------------- #
# Replay buffers (ported verbatim from sparc utils/replay_buffer_dict.py)     #
# --------------------------------------------------------------------------- #
class ReplayBufferExpert:
    """Phase-1 buffer: obs/context/next_obs/next_context/action/reward/done."""

    def __init__(self, capacity, obs_shape, context_shape, action_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.context_buf = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_context_buf = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.actions_buf = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards_buf = np.zeros((capacity,), dtype=np.float32)
        self.dones_buf = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, context, next_obs, next_context, action, reward, done):
        idx = self.ptr % self.capacity
        self.obs_buf[idx] = obs
        self.context_buf[idx] = context
        self.next_obs_buf[idx] = next_obs
        self.next_context_buf[idx] = next_context
        self.actions_buf[idx] = action
        self.rewards_buf[idx] = reward
        self.dones_buf[idx] = float(done)
        self.ptr += 1
        if self.size < self.capacity:
            self.size += 1

    def arrays(self):
        return dict(obs=self.obs_buf, context=self.context_buf,
                    next_obs=self.next_obs_buf, next_context=self.next_context_buf,
                    actions=self.actions_buf, rewards=self.rewards_buf,
                    dones=self.dones_buf, ptr=self.ptr, size=self.size)

    def load_arrays(self, d):
        self.obs_buf = d["obs"]; self.context_buf = d["context"]
        self.next_obs_buf = d["next_obs"]; self.next_context_buf = d["next_context"]
        self.actions_buf = d["actions"]; self.rewards_buf = d["rewards"]
        self.dones_buf = d["dones"]; self.ptr = int(d["ptr"]); self.size = int(d["size"])

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs_buf[idx], dtype=torch.float32, device=self.device),
            "context": torch.tensor(self.context_buf[idx], dtype=torch.float32, device=self.device),
            "next_obs": torch.tensor(self.next_obs_buf[idx], dtype=torch.float32, device=self.device),
            "next_context": torch.tensor(self.next_context_buf[idx], dtype=torch.float32, device=self.device),
            "actions": torch.tensor(self.actions_buf[idx], dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(self.rewards_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
            "dones": torch.tensor(self.dones_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
        }


class ReplayBufferDict:
    """Phase-2 buffer: adds per-transition rolling history (pre-action)."""

    def __init__(self, capacity, obs_shape, context_shape, history_shape,
                 action_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.context_buf = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.history_buf = np.zeros((capacity, *history_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_context_buf = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.actions_buf = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards_buf = np.zeros((capacity,), dtype=np.float32)
        self.dones_buf = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, context, history, next_obs, next_context, action, reward, done):
        idx = self.ptr % self.capacity
        self.obs_buf[idx] = obs
        self.context_buf[idx] = context
        self.history_buf[idx] = history
        self.next_obs_buf[idx] = next_obs
        self.next_context_buf[idx] = next_context
        self.actions_buf[idx] = action
        self.rewards_buf[idx] = reward
        self.dones_buf[idx] = float(done)
        self.ptr += 1
        if self.size < self.capacity:
            self.size += 1

    def arrays(self):
        return dict(obs=self.obs_buf, context=self.context_buf,
                    history=self.history_buf, next_obs=self.next_obs_buf,
                    next_context=self.next_context_buf, actions=self.actions_buf,
                    rewards=self.rewards_buf, dones=self.dones_buf,
                    ptr=self.ptr, size=self.size)

    def load_arrays(self, d):
        self.obs_buf = d["obs"]; self.context_buf = d["context"]
        self.history_buf = d["history"]; self.next_obs_buf = d["next_obs"]
        self.next_context_buf = d["next_context"]; self.actions_buf = d["actions"]
        self.rewards_buf = d["rewards"]; self.dones_buf = d["dones"]
        self.ptr = int(d["ptr"]); self.size = int(d["size"])

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs_buf[idx], dtype=torch.float32, device=self.device),
            "context": torch.tensor(self.context_buf[idx], dtype=torch.float32, device=self.device),
            "history": torch.tensor(self.history_buf[idx], dtype=torch.float32, device=self.device),
            "next_obs": torch.tensor(self.next_obs_buf[idx], dtype=torch.float32, device=self.device),
            "next_context": torch.tensor(self.next_context_buf[idx], dtype=torch.float32, device=self.device),
            "actions": torch.tensor(self.actions_buf[idx], dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(self.rewards_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
            "dones": torch.tensor(self.dones_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
        }


# --------------------------------------------------------------------------- #
# misc                                                                        #
# --------------------------------------------------------------------------- #
def _copy_modules(src: nn.Module, dst: nn.Module, names):
    """sparc utils/misc_utils.py copy_modules, verbatim."""
    for n in names:
        getattr(dst, n).load_state_dict(getattr(src, n).state_dict())


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # sparc train.py: torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _current_context(env) -> np.ndarray:
    """Per-reset privileged context (mass_mult, friction_mult), float32 (2,).

    PerturbEnv.reset() stores it on itself (context_out=True); the raw
    unwrapped env does not carry it, so try the wrapper first.
    """
    for obj in (env, getattr(env, "unwrapped", None)):
        c = getattr(obj, "current_context", None) if obj is not None else None
        if c is not None:
            return np.asarray(c, dtype=np.float32)
    raise RuntimeError("PerturbEnv.current_context is not set; "
                       "make_train_env(..., context_out=True) required")


def _rng_states():
    st = {
        "torch": torch.get_rng_state(),
        "np": np.random.get_state(),
        "py": random.getstate(),
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_states(st):
    torch.set_rng_state(st["torch"].cpu() if torch.is_tensor(st["torch"]) else st["torch"])
    np.random.set_state(st["np"])
    random.setstate(st["py"])
    if torch.cuda.is_available() and st.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all([t.cpu() for t in st["torch_cuda"]])
        except Exception:
            pass


def _env_rng_state(env):
    """Capture the env's sampling streams: raw-env np_random + PhysSampler."""
    raw = env.unwrapped
    out = {}
    gen = getattr(raw, "_np_random", None)
    if gen is not None:
        out["np_random"] = gen.bit_generator.state
    sampler = getattr(env, "_sampler", None)
    if sampler is not None:
        out["sampler"] = sampler.rng.bit_generator.state
    return out


def _restore_env_rng(env, st):
    """Restore np_random/sampler streams, then reset to start a fresh episode.

    Resume granularity is the episode boundary: the in-progress episode at
    crash time is abandoned (<= 1000 real steps, re-collected after resume as
    part of the remaining budget); all network/optimizer/buffer/RNG state
    continues exactly from the checkpoint.
    """
    if st is None:
        return
    raw = env.unwrapped
    if st.get("np_random") is not None:
        # gymnasium 0.29.2-fork Env.np_random has no setter -> _np_random
        bg = np.random.PCG64()
        bg.state = st["np_random"]
        raw._np_random = np.random.Generator(bg)
    sampler = getattr(env, "_sampler", None)
    if sampler is not None and st.get("sampler") is not None:
        bg = np.random.PCG64()
        bg.state = st["sampler"]
        sampler.rng = np.random.Generator(bg)


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def _paths(job_dir: str):
    ckpt = os.path.join(job_dir, "ckpt")
    return {
        "ckpt_dir": ckpt,
        "expert": os.path.join(ckpt, "expert.pt"),
        "final": os.path.join(ckpt, "final.pt"),
        "p1_state": os.path.join(ckpt, "phase1_state.pt"),
        "p2_state": os.path.join(ckpt, "phase2_state.pt"),
        "counters": os.path.join(job_dir, "counters.json"),
        "progress": os.path.join(job_dir, "progress.jsonl"),
    }


def _load_counters(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"phase1_steps": 0, "phase2_steps": 0, "real_transitions": 0}


def _env_dims(env):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    return obs_dim, act_dim, env.action_space.low, env.action_space.high


def _torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Phase 1: QR-SAC expert with privileged context (sparc rma_phase1.py)        #
# --------------------------------------------------------------------------- #
def _run_phase1(job_dir, task, seed, cfg, counters, P):
    device = _get_device()
    p1_steps = int(cfg["phase1_steps"])
    learning_starts = int(cfg["learning_starts"])
    batch_size = int(cfg["batch_size"])
    gamma = float(cfg["gamma"])
    tau = float(cfg["tau"])
    policy_lr = float(cfg["policy_lr"])
    q_lr = float(cfg["q_lr"])
    policy_frequency = int(cfg["policy_frequency"])
    autotune = bool(cfg["autotune"])
    grad_clip_critic = float(cfg["grad_clip_critic"])
    num_quantiles = int(cfg["num_quantiles"])
    quantile_kappa = float(cfg["quantile_kappa"])
    top_quantiles_to_drop = int(cfg["top_quantiles_to_drop"])
    save_interval = int(cfg.get("save_interval", SAVE_INTERVAL))
    context_dim = int(cfg.get("context_dim", 2))

    env = make_train_env(task, seed, context_out=True)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    obs_dim, act_dim, a_low, a_high = _env_dims(env)
    context = _current_context(env)
    obs_shape, context_shape, action_shape = (obs_dim,), (context_dim,), (act_dim,)

    # networks (sparc rma_phase1.py lines 66-76)
    expert_policy = ExpertPolicy(obs_dim, context_dim, act_dim, a_low, a_high).to(device)
    q_kwargs = dict(obs_dim=obs_dim, context_dim=context_dim, action_dim=act_dim,
                    num_quantiles=num_quantiles)
    q1 = CriticContext(**q_kwargs).to(device)
    q2 = CriticContext(**q_kwargs).to(device)
    tq1 = CriticContext(**q_kwargs).to(device)
    tq2 = CriticContext(**q_kwargs).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=q_lr)
    expert_optimizer = optim.Adam(list(expert_policy.parameters()), lr=policy_lr)

    # automatic entropy tuning (sparc rma_phase1.py lines 79-85)
    if autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=q_lr)
    else:
        target_entropy = -float(act_dim)
        log_alpha = None
        alpha = float(cfg.get("alpha", 0.01))
        a_optimizer = None

    # replay buffer sized to the phase budget (sparc default buffer_size)
    rb = ReplayBufferExpert(p1_steps, obs_shape, context_shape, action_shape, device)

    # QR-SAC quantile mid-points
    tau_hat = (torch.arange(num_quantiles, dtype=torch.float32, device=device) + 0.5) \
        / num_quantiles

    # ---- optional resume from mid-phase checkpoint ----
    start_step = 0
    ep_ret = ep_len = 0.0
    episodes = 0
    last_ep_return = None
    t_start = time.time()
    sig = dict(task=task, seed=seed, phase1_steps=p1_steps,
               learning_starts=learning_starts)
    if os.path.exists(P["p1_state"]):
        try:
            st = torch.load(P["p1_state"], map_location="cpu", weights_only=False)
            if st.get("sig") == sig and 0 < st["step"] <= p1_steps:
                expert_policy.load_state_dict(st["expert"])
                q1.load_state_dict(st["q1"]); q2.load_state_dict(st["q2"])
                tq1.load_state_dict(st["tq1"]); tq2.load_state_dict(st["tq2"])
                q_optimizer.load_state_dict(st["q_optimizer"])
                expert_optimizer.load_state_dict(st["expert_optimizer"])
                if autotune and st.get("log_alpha") is not None:
                    log_alpha.data.copy_(st["log_alpha"].to(device))
                    alpha = log_alpha.exp().item()
                    a_optimizer.load_state_dict(st["a_optimizer"])
                rb.load_arrays(st["buffer"])
                _set_rng_states(st["rng"])
                start_step = int(st["step"])
                ep_ret = float(st["ep_ret"]); ep_len = float(st["ep_len"])
                episodes = int(st["episodes"])
                last_ep_return = st.get("last_ep_return")
                t_start = time.time() - float(st.get("elapsed", 0.0))
                # fresh episode from the restored RNG streams; the abandoned
                # in-progress episode's partial return is discarded
                env.reset(seed=seed)
                _restore_env_rng(env, st["env_rng"])
                obs, _ = env.reset()
                context = _current_context(env)
                ep_ret = ep_len = 0.0
                print(f"[rma:{task}:s{seed}] phase1 resume from step {start_step}",
                      flush=True)
        except Exception as e:  # corrupt state -> fresh start
            print(f"[rma:{task}:s{seed}] phase1 state ignored ({e})", flush=True)
            env.reset(seed=seed)
            obs, _ = env.reset(seed=seed)
            context = _current_context(env)

    log_freq = int(cfg.get("log_freq", LOG_FREQ))
    actor_loss = q1_loss = q2_loss = alpha_loss = None

    for global_step in range(start_step, p1_steps):
        # sample actions (random prefill inside the phase-1 budget)
        if global_step < learning_starts:
            action = env.action_space.sample()
            action = np.asarray(action, dtype=np.float32)
        else:
            with torch.no_grad():
                a, _, _ = expert_policy.get_action(
                    torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1),
                    torch.tensor(context, dtype=torch.float32, device=device).view(1, -1))
            action = a.detach().cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        reward = float(reward)
        ep_ret += reward
        ep_len += 1

        # context is constant within the episode (PerturbEnv resamples only on
        # reset); at an episode end the stored next_context is the terminal
        # step's episode context, matching sparc's final_observation handling.
        next_context = _current_context(env)

        # store (done = terminated only: timeouts bootstrap from the final obs)
        rb.add(obs, context, next_obs, next_context, action, reward, terminated)

        if terminated or truncated:
            episodes += 1
            last_ep_return = ep_ret
            _append_jsonl(P["progress"], {
                "phase": 1, "step": global_step, "episode": episodes,
                "episode_return": round(ep_ret, 4),
                "episode_length": int(ep_len),
            })
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            context = _current_context(env)
            ep_ret = ep_len = 0.0
        else:
            obs = next_obs
            context = next_context

        # gradient updates: 1 update per env step after learning_starts
        if global_step > learning_starts:
            data = rb.sample(batch_size)
            batch_obs = data["obs"]
            batch_ctx = data["context"]
            batch_next_obs = data["next_obs"]
            batch_next_ctx = data["next_context"]
            batch_actions = data["actions"]
            batch_rewards = data["rewards"]      # (B, 1)
            batch_dones = data["dones"]          # (B, 1)

            with torch.no_grad():
                next_actions, next_log_pi, _ = expert_policy.get_action(
                    batch_next_obs, batch_next_ctx)
                q1_next = tq1(batch_next_obs, next_actions, batch_next_ctx)
                q2_next = tq2(batch_next_obs, next_actions, batch_next_ctx)
                # QR-SAC target: (B, N); entropy term broadcast over quantiles
                min_next = min_critic_quantiles(q1_next, q2_next,
                                                 top_quantiles_to_drop) \
                    - alpha * next_log_pi
                target_q = batch_rewards + (1.0 - batch_dones) * gamma * min_next

            q1_pred = q1(batch_obs, batch_actions, batch_ctx)
            q2_pred = q2(batch_obs, batch_actions, batch_ctx)

            q1_loss = _quantile_huber_loss(q1_pred, target_q, tau_hat, quantile_kappa)
            q2_loss = _quantile_huber_loss(q2_pred, target_q, tau_hat, quantile_kappa)
            q_loss = q1_loss + q2_loss
            q_optimizer.zero_grad()
            q_loss.backward()
            if grad_clip_critic > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(q1.parameters()) + list(q2.parameters()),
                    max_norm=grad_clip_critic)
            q_optimizer.step()

            # delayed policy update (policy_frequency updates per trigger)
            if global_step % policy_frequency == 0:
                for _ in range(policy_frequency):
                    pi, log_pi, _ = expert_policy.get_action(batch_obs, batch_ctx)
                    q1_pi = q1(batch_obs, pi, batch_ctx)
                    q2_pi = q2(batch_obs, pi, batch_ctx)
                    # policy loss on the min-critics' MEAN quantile value
                    min_q_pi = min_critic_quantiles(q1_pi, q2_pi,
                                                    top_quantiles_to_drop) \
                        .mean(dim=1, keepdim=True)
                    actor_loss = ((alpha * log_pi) - min_q_pi).mean()
                    expert_optimizer.zero_grad()
                    actor_loss.backward()
                    expert_optimizer.step()

                    if autotune:
                        with torch.no_grad():
                            _, new_log_pi, _ = expert_policy.get_action(
                                batch_obs, batch_ctx)
                        alpha_loss = (-log_alpha.exp()
                                      * (new_log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # target networks (frequency 1 -> every step)
            if global_step % int(cfg.get("target_network_frequency", 1)) == 0:
                for param, target_param in zip(q1.parameters(), tq1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), tq2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # periodic progress log
            if global_step % log_freq == 0:
                el = time.time() - t_start
                entry = {
                    "phase": 1, "step": global_step,
                    "episode_return": (round(last_ep_return, 4)
                                       if last_ep_return is not None else None),
                    "loss": round(float(q_loss.item()) / 2.0, 6),
                    "q1_loss": round(float(q1_loss.item()), 6),
                    "q2_loss": round(float(q2_loss.item()), 6),
                    "actor_loss": (round(float(actor_loss.item()), 6)
                                   if actor_loss is not None else None),
                    "alpha": round(alpha, 6),
                    "episodes": episodes,
                    "elapsed": round(el, 1),
                    "sps": round((global_step + 1 - start_step) / max(el, 1e-9), 1),
                }
                if autotune and alpha_loss is not None:
                    entry["alpha_loss"] = round(float(alpha_loss.item()), 6)
                _append_jsonl(P["progress"], entry)

        # periodic resumable checkpoint
        if (global_step + 1) % save_interval == 0 or (global_step + 1) == p1_steps:
            state = {
                "sig": sig, "step": global_step + 1,
                "expert": expert_policy.state_dict(),
                "q1": q1.state_dict(), "q2": q2.state_dict(),
                "tq1": tq1.state_dict(), "tq2": tq2.state_dict(),
                "q_optimizer": q_optimizer.state_dict(),
                "expert_optimizer": expert_optimizer.state_dict(),
                "log_alpha": (log_alpha.detach().cpu() if autotune else None),
                "a_optimizer": (a_optimizer.state_dict() if autotune else None),
                "buffer": rb.arrays(),
                "rng": _rng_states(),
                "env_rng": _env_rng_state(env),
                "ep_ret": ep_ret, "ep_len": ep_len, "episodes": episodes,
                "last_ep_return": last_ep_return,
                "elapsed": time.time() - t_start,
                "obs_dim": obs_dim, "act_dim": act_dim, "context_dim": context_dim,
            }
            _torch_save(state, P["p1_state"])
            counters["phase1_steps"] = global_step + 1
            counters["real_transitions"] = counters["phase1_steps"] + counters["phase2_steps"]
            _atomic_json(P["counters"], counters)

    # save the frozen expert for phase 2 (sparc rma_phase1.py lines 232-236)
    _torch_save({
        "state_dict": expert_policy.state_dict(),
        "obs_dim": obs_dim, "act_dim": act_dim, "context_dim": context_dim,
        "action_low": np.asarray(a_low, dtype=np.float32),
        "action_high": np.asarray(a_high, dtype=np.float32),
        "task": task, "seed": seed, "num_quantiles": num_quantiles,
    }, P["expert"])
    counters["phase1_steps"] = p1_steps
    counters["real_transitions"] = p1_steps + counters["phase2_steps"]
    _atomic_json(P["counters"], counters)
    if os.path.exists(P["p1_state"]):
        os.remove(P["p1_state"])  # phase complete; free the (large) buffer state
    env.close()
    el = time.time() - t_start
    print(f"[rma:{task}:s{seed}] phase1 done: {p1_steps} steps, "
          f"{el:.0f}s ({p1_steps / max(el, 1e-9):.1f} sps), saved {P['expert']}",
          flush=True)


# --------------------------------------------------------------------------- #
# Phase 2: frozen expert + history adapter regression (sparc rma_phase2.py)   #
# --------------------------------------------------------------------------- #
def _run_phase2(job_dir, task, seed, cfg, counters, P):
    device = _get_device()
    p2_steps = int(cfg["phase2_steps"])
    learning_starts = int(cfg["learning_starts"])
    batch_size = int(cfg["batch_size"])
    adapter_lr = float(cfg["adapter_lr"])
    history_length = int(cfg["history_length"])
    context_dim = int(cfg.get("context_dim", 2))
    save_interval = int(cfg.get("save_interval", SAVE_INTERVAL))

    if not os.path.exists(P["expert"]):
        raise FileNotFoundError(f"expert checkpoint missing: {P['expert']}")
    expert_ckpt = torch.load(P["expert"], map_location="cpu", weights_only=False)
    obs_dim = int(expert_ckpt["obs_dim"])
    act_dim = int(expert_ckpt["act_dim"])
    a_low = np.asarray(expert_ckpt["action_low"], dtype=np.float32)
    a_high = np.asarray(expert_ckpt["action_high"], dtype=np.float32)
    assert int(expert_ckpt["context_dim"]) == context_dim

    # fresh RNG streams for phase 2 (upstream runs it as a separate process)
    _seed_everything(seed)

    env = make_train_env(task, seed, context_out=True)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    context = _current_context(env)

    # frozen expert
    expert_policy = ExpertPolicy(obs_dim, context_dim, act_dim, a_low, a_high).to(device)
    expert_policy.load_state_dict(expert_ckpt["state_dict"])
    expert_policy.eval()
    for param in expert_policy.parameters():
        param.requires_grad = False

    # adapter policy: expert's encoders/decision layers + fresh HistoryAdapter
    adapter_policy = AdapterPolicy(obs_dim, act_dim, history_length,
                                    a_low, a_high).to(device)
    _copy_modules(expert_policy, adapter_policy,
                  ["fc1", "fc2", "fc3", "fc_mean", "fc_logstd"])
    for param in adapter_policy.parameters():
        param.requires_grad = False
    for param in adapter_policy.adapter.parameters():
        param.requires_grad = True
    adapter_optimizer = optim.Adam(adapter_policy.adapter.parameters(), lr=adapter_lr)

    obs_shape, context_shape, action_shape = (obs_dim,), (context_dim,), (act_dim,)
    history_shape = (history_length, obs_dim + act_dim)
    rb = ReplayBufferDict(p2_steps, obs_shape=obs_shape,
                          context_shape=context_shape, history_shape=history_shape,
                          action_shape=action_shape, device=device)

    # rolling history, mirrored from the wind env:
    # zeros at reset; after env.step(a) -> append (next_obs, a)
    history = np.zeros(history_shape, dtype=np.float32)

    # ---- optional resume ----
    start_step = 0
    ep_ret = ep_len = 0.0
    episodes = 0
    last_ep_return = None
    adapter_loss = None
    t_start = time.time()
    sig = dict(task=task, seed=seed, phase2_steps=p2_steps,
               learning_starts=learning_starts, history_length=history_length)
    if os.path.exists(P["p2_state"]):
        try:
            st = torch.load(P["p2_state"], map_location="cpu", weights_only=False)
            if st.get("sig") == sig and 0 < st["step"] <= p2_steps:
                adapter_policy.load_state_dict(st["adapter"])
                adapter_optimizer.load_state_dict(st["adapter_optimizer"])
                rb.load_arrays(st["buffer"])
                _set_rng_states(st["rng"])
                start_step = int(st["step"])
                ep_ret = float(st["ep_ret"]); ep_len = float(st["ep_len"])
                episodes = int(st["episodes"])
                last_ep_return = st.get("last_ep_return")
                t_start = time.time() - float(st.get("elapsed", 0.0))
                env.reset(seed=seed)
                _restore_env_rng(env, st["env_rng"])
                obs, _ = env.reset()
                obs = np.asarray(obs, dtype=np.float32)
                context = _current_context(env)
                history = np.zeros(history_shape, dtype=np.float32)
                ep_ret = ep_len = 0.0  # abandoned in-progress episode
                print(f"[rma:{task}:s{seed}] phase2 resume from step {start_step}",
                      flush=True)
        except Exception as e:
            print(f"[rma:{task}:s{seed}] phase2 state ignored ({e})", flush=True)
            env.reset(seed=seed)
            obs, _ = env.reset(seed=seed)
            obs = np.asarray(obs, dtype=np.float32)
            context = _current_context(env)

    log_freq = int(cfg.get("log_freq", LOG_FREQ))

    for global_step in range(start_step, p2_steps):
        # actions from the ADAPTER policy (from step 0; no random prefill)
        hist_t = history.copy()
        with torch.no_grad():
            a, _, _ = adapter_policy.get_action(
                torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1),
                torch.tensor(hist_t, dtype=torch.float32, device=device).view(1, *history_shape))
        action = a.detach().cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        reward = float(reward)
        ep_ret += reward
        ep_len += 1

        # history update AFTER acting (wind env step() convention):
        # (post-step obs, action) enters the last row
        history[:-1] = history[1:]
        history[-1] = np.concatenate([next_obs, action]).astype(np.float32)

        next_context = _current_context(env)
        rb.add(obs, context, hist_t, next_obs, next_context, action, reward, terminated)

        if terminated or truncated:
            episodes += 1
            last_ep_return = ep_ret
            _append_jsonl(P["progress"], {
                "phase": 2, "step": global_step, "episode": episodes,
                "episode_return": round(ep_ret, 4), "episode_length": int(ep_len),
            })
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            context = _current_context(env)
            history[:] = 0.0
            ep_ret = ep_len = 0.0
        else:
            obs = next_obs
            context = next_context

        # supervised regression on the expert's context latent
        if global_step > learning_starts:
            data = rb.sample(batch_size)
            batch_ctx = data["context"]
            batch_hist = data["history"]
            with torch.no_grad():
                z = expert_policy.encoder(batch_ctx)
            z_hat = adapter_policy.adapter(batch_hist)
            adapter_loss = F.mse_loss(z_hat, z.detach())
            adapter_optimizer.zero_grad()
            adapter_loss.backward()
            adapter_optimizer.step()

            if global_step % log_freq == 0:
                el = time.time() - t_start
                _append_jsonl(P["progress"], {
                    "phase": 2, "step": global_step,
                    "episode_return": (round(last_ep_return, 4)
                                       if last_ep_return is not None else None),
                    "loss": round(float(adapter_loss.item()), 8),
                    "adapter_loss": round(float(adapter_loss.item()), 8),
                    "episodes": episodes,
                    "elapsed": round(el, 1),
                    "sps": round((global_step + 1 - start_step) / max(el, 1e-9), 1),
                })

        # periodic resumable checkpoint
        if (global_step + 1) % save_interval == 0 or (global_step + 1) == p2_steps:
            state = {
                "sig": sig, "step": global_step + 1,
                "adapter": adapter_policy.state_dict(),
                "adapter_optimizer": adapter_optimizer.state_dict(),
                "buffer": rb.arrays(),
                "rng": _rng_states(),
                "env_rng": _env_rng_state(env),
                "ep_ret": ep_ret, "ep_len": ep_len, "episodes": episodes,
                "last_ep_return": last_ep_return,
                "elapsed": time.time() - t_start,
                "obs_dim": obs_dim, "act_dim": act_dim,
                "history_length": history_length,
            }
            _torch_save(state, P["p2_state"])
            counters["phase2_steps"] = global_step + 1
            counters["real_transitions"] = counters["phase1_steps"] + counters["phase2_steps"]
            _atomic_json(P["counters"], counters)

    # final adapter policy (the deliverable checkpoint)
    _torch_save({
        "state_dict": adapter_policy.state_dict(),
        "obs_dim": obs_dim, "act_dim": act_dim,
        "history_length": history_length,
        "action_low": a_low, "action_high": a_high,
        "task": task, "seed": seed,
    }, P["final"])
    counters["phase2_steps"] = p2_steps
    counters["real_transitions"] = counters["phase1_steps"] + p2_steps
    _atomic_json(P["counters"], counters)
    if os.path.exists(P["p2_state"]):
        os.remove(P["p2_state"])
    env.close()
    el = time.time() - t_start
    print(f"[rma:{task}:s{seed}] phase2 done: {p2_steps} steps, "
          f"{el:.0f}s ({p2_steps / max(el, 1e-9):.1f} sps), saved {P['final']}",
          flush=True)


# --------------------------------------------------------------------------- #
# Public entry points (runner interface)                                      #
# --------------------------------------------------------------------------- #
def train(job_dir: str, task: str, seed: int, cfg: dict) -> None:
    """Two-phase RMA training; see module docstring and the task interface."""
    os.makedirs(job_dir, exist_ok=True)
    P = _paths(job_dir)
    os.makedirs(P["ckpt_dir"], exist_ok=True)
    counters = _load_counters(P["counters"])
    p1 = int(cfg["phase1_steps"])
    p2 = int(cfg["phase2_steps"])

    # idempotent: fully complete job -> return immediately
    if (os.path.exists(P["final"]) and counters["phase1_steps"] >= p1
            and counters["phase2_steps"] >= p2):
        print(f"[rma:{task}:s{seed}] already complete "
              f"({counters['real_transitions']} transitions)", flush=True)
        return

    _seed_everything(seed)

    if counters["phase1_steps"] < p1 or not os.path.exists(P["expert"]):
        if counters["phase1_steps"] > 0 and not os.path.exists(P["expert"]):
            # expert.pt is only written at phase-1 completion; without it any
            # partial phase-1 progress cannot serve phase 2 -> restart phase 1.
            counters["phase1_steps"] = 0
            counters["real_transitions"] = counters["phase2_steps"]
        _run_phase1(job_dir, task, seed, cfg, counters, P)
    else:
        print(f"[rma:{task}:s{seed}] phase1 already complete, skipping", flush=True)
        counters["phase1_steps"] = p1

    if counters["phase2_steps"] < p2 or not os.path.exists(P["final"]):
        _run_phase2(job_dir, task, seed, cfg, counters, P)
    else:
        counters["phase2_steps"] = p2
        counters["real_transitions"] = counters["phase1_steps"] + p2
        _atomic_json(P["counters"], counters)
    print(f"[rma:{task}:s{seed}] train complete: "
          f"{counters['real_transitions']} real transitions", flush=True)


# --------------------------------------------------------------------------- #
# Frozen evaluation adapter (evaluator protocol: act / reset_episodes)        #
# --------------------------------------------------------------------------- #
class RMAEvalAdapter:
    """Frozen AdapterPolicy with per-env rolling history.

    The privileged context is NEVER used here — actions come from
    (base obs, history) only.

    act(obs, active): deterministic action = tanh(actor mean) rescaled to the
    action bounds.  History rows advance only for envs that are active: the
    (obs, action) pair of a step is appended at the NEXT act() call, when the
    post-step observation is known — exactly the training convention
    (row appended after env.step is (next_obs, a)).  Rows of finished envs
    are frozen; reset_episodes(mask) zeroes them (and clears any pending
    pair) so re-evaluating on the same adapter reproduces results bit-exactly.
    """

    def __init__(self, policy: AdapterPolicy, history_length: int,
                 obs_dim: int, act_dim: int, device):
        self.policy = policy.to(device).eval()
        self.device = device
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._pair_dim = obs_dim + act_dim
        self.history = np.zeros((0, history_length, self._pair_dim), dtype=np.float32)
        self._pending_actions = None   # actions issued at the previous act()
        self._pending_mask = None      # rows that were active then

    def _resize(self, n: int):
        if self.history.shape[0] != n:
            self.history = np.zeros((n, self.history_length, self._pair_dim),
                                    dtype=np.float32)
            self._pending_actions = None
            self._pending_mask = None

    def act(self, obs, active):
        obs = np.asarray(obs, dtype=np.float32)
        n = obs.shape[0]
        if obs.ndim == 1:
            obs = obs[None, :]
            n = 1
        self._resize(n)
        active = np.asarray(active, dtype=bool)
        if active.shape != (n,):
            raise ValueError(f"active mask shape {active.shape} != ({n},)")

        # advance history for envs that took a step since the last act() and
        # are still active: append (current obs, previous action)
        if self._pending_actions is not None:
            rows = np.where(self._pending_mask & active)[0]
            if len(rows):
                pair = np.concatenate(
                    [obs[rows], self._pending_actions[rows]], axis=1
                ).astype(np.float32)
                rolled = np.concatenate(
                    [self.history[rows, 1:], pair[:, None, :]], axis=1)
                self.history[rows] = rolled

        actions = np.zeros((n, self.act_dim), dtype=np.float32)
        if active.any():
            with torch.no_grad():
                mean, _ = self.policy(
                    torch.tensor(obs[active], dtype=torch.float32,
                                 device=self.device),
                    torch.tensor(self.history[active], dtype=torch.float32,
                                 device=self.device))
                a = torch.tanh(mean) * self.policy.action_scale \
                    + self.policy.action_bias
            actions[active] = a.detach().cpu().numpy().astype(np.float32)

        self._pending_actions = actions.copy()
        self._pending_mask = active.copy()
        return actions

    def reset_episodes(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape and self.history.shape[0] and mask.shape[0] == self.history.shape[0]:
            self.history[mask] = 0.0
            if self._pending_mask is not None:
                self._pending_mask = self._pending_mask & ~mask
                if not self._pending_mask.any():
                    self._pending_actions = None
                    self._pending_mask = None
        return None


def load_eval_adapter(job_dir: str, task: str, seed: int, cfg: dict):
    """Load the frozen adapter (ckpt/final.pt) for the unified evaluator."""
    P = _paths(job_dir)
    if not os.path.exists(P["final"]):
        raise FileNotFoundError(
            f"adapter checkpoint missing: {P['final']} (train first)")
    ckpt = torch.load(P["final"], map_location="cpu", weights_only=False)
    history_length = int(ckpt.get("history_length", cfg.get("history_length", 50)))
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    a_low = np.asarray(ckpt.get("action_low", -np.ones(act_dim)), dtype=np.float32)
    a_high = np.asarray(ckpt.get("action_high", np.ones(act_dim)), dtype=np.float32)
    policy = AdapterPolicy(obs_dim, act_dim, history_length, a_low, a_high)
    policy.load_state_dict(ckpt["state_dict"])
    device = _get_device()
    return RMAEvalAdapter(policy, history_length, obs_dim, act_dim, device)
