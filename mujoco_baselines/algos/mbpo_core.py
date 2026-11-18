"""Shared MBPO / RMBPO training skeleton — plain-PyTorch port of Unstable-Baselines MBPO.

This module is the single code path used by both ``algos/mbpo.py`` and
``algos/rmbpo.py`` (phase B trains both from scratch on the shared v4 stack; the
ONLY allowed differences are the auxiliary-model round and the source of model
rollouts, both injected by :class:`RmbpoTrainer` in ``algos/rmbpo.py``).

Port provenance (SHAs in ``upstreams.lock.json``):

* ``upstreams/unstable_baselines`` == github.com/x35f/model_based_rl @ 2c2c4ac
  (2023-05-10): ``mbpo/{agent,transition_model,trainer,main}.py`` and
  ``common/models.py`` (EnsembleModel) — vendored verbatim in this tree; the
  classes below are line-by-line torch re-implementations of those files.
* github.com/x35f/unstable_Baselines @ 5556a32 ("UPDATE: support mbpo",
  2023-05-10, single-commit repo): supplies the ``unstable_baselines.common``
  modules that the files above import (``networks.py`` GaussianPolicyNetwork /
  SequentialNetwork, ``buffer.py`` ReplayBuffer, ``normalizer.py``
  StandardNormalizer, ``functional.py`` {soft_update_network,
  dict_batch_generator, minibatch_inference, merge_data_batch}, ``scheduler.py``
  Scheduler, ``util.set_global_seed``).  Those files are NOT vendored here; their
  math is re-implemented below and was cross-checked against that commit
  (cloned read-only; not part of this tree).

Deliberate changes vs. upstream (everything else is a faithful port):

1. No runtime dependency on ``unstable_baselines`` (plain torch / numpy only).
2. ``_termination_fn`` name lists extended from {Hopper-v2, Hopper-v3, ...} to
   also accept the -v4 ids (upstream would raise NotImplementedError for v4; the
   RMBPO authors trained on v4, and v4 termination semantics are identical).
   The (no-op) ``np.abs(next_obs[:, 1:] < 100)`` Hopper quirk is kept verbatim.
3. ``ReplayBuffer.add_traj`` is vectorized (upstream loops per sample); the ring
   semantics (final content = the last ``size`` samples, ``curr`` position) are
   reproduced exactly, only faster.  ``sample``/``resize`` keep upstream logic.
4. The trainer's per-step/round structure is upstream's, plus phase-B plumbing:
   budget bookkeeping, counters.json / progress.jsonl, resumable checkpoints,
   and hooks (``_after_model_training``) for the RMBPO auxiliary model.
5. In-training evaluation is dropped (the frozen evaluator runs separately;
   its interactions are budget-exempt per the task book).

Upstream facts worth recording (verified against the sources above):

* rollout / env actions both use ``agent.select_action(obs)`` with the
  ``deterministic=False`` default, i.e. STOCHASTIC tanh-Gaussian samples (the
  task brief guessed "deterministic mean"; ``trainer.rollout_model`` actually
  calls ``select_action`` without the flag — replicated as-is).
* The Gaussian policy predicts its log-std from the MLP (predicted_std=True,
  output 2*act_dim; confirmed by the released RMBPO ``.pt`` weights whose
  policy head is Linear(256 -> 2*act_dim)); ``stablize_log_prob`` tanh
  correction is kept verbatim.
* The env/model buffer split ratio is governed by the *trainer's*
  ``hold_out_ratio`` (0.1 in the upstream configs) while the transition-model
  config's ``holdout_ratio`` (0.2) is dead config.  This experiment freezes a
  single number (``cfg['model']['holdout_ratio'] = 0.2``), which is what the
  split uses here.
* ``model_buffer.resize`` (called once, right after warmup, because the rollout
  horizon scheduler is constant 1) sizes the model replay to
  ``int(rollout_batch * 1000 / train_model_interval)`` = one epoch worth of
  model transitions — with the budget-scaled intervals this keeps the
  "model data freshness window = last 1000 real steps" property of the paper
  regime at every point of the run.
"""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import random
import time
import warnings
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from ..constants import MPS_PIPE_DIR
from ..envs.perturbed import PERTURBED_IDS, make_train_env

# Keep every CUDA process off the kev MPS pipe (see constants.MPS_PIPE_DIR).
if "CUDA_MPS_PIPE_DIRECTORY" not in os.environ:
    try:
        os.makedirs(MPS_PIPE_DIR, exist_ok=True)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE_DIR
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# seeds / device
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int):
    """Port of unstable_baselines.common.util.set_global_seed."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict) -> torch.device:
    dev = os.environ.get("MJB_ALGO_DEVICE") or cfg.get("device")
    if dev:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# networks (ports of unstable_baselines.common.networks)
# --------------------------------------------------------------------------- #
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


_ACT_FNS = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "identity": nn.Identity,
    "swish": Swish,
}


def _act_cls(name: str):
    return _ACT_FNS[name.lower()]


class MLP(nn.Module):
    """Port of common.networks.SequentialNetwork (mlp-only, list-input concat)."""

    def __init__(self, in_dim: int, out_dim: int, hidden, act: str = "relu",
                 out_act: str = "identity"):
        super().__init__()
        hidden = list(hidden)
        layers = []
        curr = in_dim
        for h in hidden:
            layers += [nn.Linear(curr, int(h)), _act_cls(act)()]
            curr = int(h)
        layers += [nn.Linear(curr, out_dim), _act_cls(out_act)()]
        self.networks = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.cat(list(inputs), dim=1)
        return self.networks(inputs)

    @property
    def weights(self):
        """Linear layer weights in layer order (used by the L2 decay loss)."""
        return [m.weight for m in self.networks if isinstance(m, nn.Linear)]


class GaussianPolicyNetwork(nn.Module):
    """Port of common.networks.GaussianPolicyNetwork.

    Configured exactly as the unstable_baselines / rmbpo-tmlr ``agent``
    config: network_params [(mlp,256),(mlp,256)], relu, identity out,
    re_parameterize=True, stablize_log_prob=True, predicted_std=True (default —
    the MLP outputs mean and log_std, 2*act_dim).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256),
                 act: str = "relu", out_act: str = "identity",
                 re_parameterize: bool = True, stablize_log_prob: bool = True,
                 log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.action_dim = act_dim
        self.deterministic = False
        self.policy_type = "Gaussian"
        self.predicted_std = True
        self.re_parameterize = re_parameterize
        self.stablize_log_prob = stablize_log_prob
        self.networks = MLP(obs_dim, act_dim * 2, hidden, act, out_act)
        self.register_buffer("action_scale", torch.ones(act_dim))
        self.register_buffer("action_bias", torch.zeros(act_dim))
        self.register_buffer("log_std_min", torch.tensor(float(log_std_min)))
        self.register_buffer("log_std_max", torch.tensor(float(log_std_max)))

    def set_action_range(self, low, high):
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        with torch.no_grad():
            self.action_scale.copy_(
                torch.from_numpy((high - low) / 2.0))
            self.action_bias.copy_(
                torch.from_numpy((high + low) / 2.0))

    def forward(self, obs):
        out = self.networks(obs)
        action_mean = out[:, : self.action_dim]
        action_log_std = out[:, self.action_dim:]
        return action_mean, action_log_std

    def sample(self, obs, deterministic: bool = False):
        mean, log_std = self.forward(obs)
        if deterministic:
            action_mean_raw = mean.detach()
            action = torch.tanh(action_mean_raw) * self.action_scale + self.action_bias
            return {
                "action_mean_raw": mean,
                "action": action,
                "log_prob": torch.ones_like(action_mean_raw),
                "log_std": torch.zeros_like(action_mean_raw),
            }
        if self.stablize_log_prob:
            # sac-like actor: re-parameterized tanh-Gaussian with the upstream
            # log-prob stabilization (kept verbatim).
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mean)
            dist = Normal(mean, log_std.exp())
            if self.re_parameterize:
                action_prev_tanh = dist.rsample()
            else:
                action_prev_tanh = dist.sample()
            action_raw = torch.tanh(action_prev_tanh)
            action = action_raw * self.action_scale + self.action_bias
            log_prob_prev_tanh = dist.log_prob(action_prev_tanh)
            log_prob = log_prob_prev_tanh - (
                2.0 * (math.log(2) - action_prev_tanh - F.softplus(-2.0 * action_prev_tanh)))
            log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
            return {
                "action_mean_raw": mean,
                "action": action,
                "log_prob": log_prob,
                "log_std": log_std,
            }
        else:
            action_mean = torch.tanh(mean)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mean)
            dist = Normal(action_mean, log_std.exp())
            action = dist.sample()
            action = torch.clip(action, min=-1.0, max=1.0)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            action = action * self.action_scale + self.action_bias
            return {
                "action_mean_raw": action_mean * self.action_scale,
                "action": action,
                "log_prob": log_prob,
                "log_std": log_std,
            }


# --------------------------------------------------------------------------- #
# functional / normalizer / scheduler (ports of common.{functional,normalizer,scheduler})
# --------------------------------------------------------------------------- #
def soft_update_network(source_network, target_network, tau):
    for target_param, local_param in zip(target_network.parameters(),
                                         source_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


def dict_batch_generator(data, batch_size, keys=None):
    if keys is None:
        keys = list(data.keys())
    num_data = len(data[keys[0]])
    num_batches = int(np.ceil(num_data / batch_size))
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    for batch_id in range(num_batches):
        batch_start = batch_id * batch_size
        batch_end = min(num_data, (batch_id + 1) * batch_size)
        batch_data = {}
        for key in keys:
            batch_data[key] = data[key][indices[batch_start:batch_end]]
        yield batch_data


def minibatch_inference(args, rollout_fn, batch_size=10000, cat_dim=1):
    data_size = len(args[0])
    num_batches = int(np.ceil(data_size / batch_size))
    inference_results = None
    multi_op = False
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(data_size, (i + 1) * batch_size)
        input_batch = [ip[batch_start:batch_end] for ip in args]
        outputs = rollout_fn(*input_batch)
        if i == 0:
            if isinstance(outputs, tuple):
                multi_op = True
            else:
                multi_op = False
            inference_results = outputs
        else:
            if multi_op:
                inference_results = tuple(
                    torch.cat([prev_re, op], dim=cat_dim)
                    for prev_re, op in zip(inference_results, outputs))
            else:
                inference_results = torch.cat([inference_results, outputs])
    return inference_results


def merge_data_batch(data1_dict, data2_dict):
    for key in data1_dict:
        if isinstance(data1_dict[key], np.ndarray):
            data1_dict[key] = np.concatenate([data1_dict[key], data2_dict[key]], axis=0)
        elif isinstance(data1_dict[key], torch.Tensor):
            data1_dict[key] = torch.cat([data1_dict[key], data2_dict[key]], dim=0)
    return data1_dict


def materialize(d: dict) -> dict:
    """.item() any 0-dim tensors in a log dict (single host sync per dict)."""
    return {k: (v.item() if torch.is_tensor(v) else v) for k, v in d.items()}


class StandardNormalizer:
    """Port of common.normalizer.StandardNormalizer (torch branch only — the
    upstream object only ever sees tensors in this code path; they live on the
    trainer's device)."""

    def __init__(self, device=None):
        self.device = device
        self.reset()

    def reset(self):
        self.mean = None
        self.var = None
        self.tot_count = 0

    def update(self, samples):
        sample_count = len(samples)
        if self.tot_count == 0:
            dim = samples.shape[1]
            self.mean = torch.zeros(dim, dtype=torch.float32, device=samples.device)
            self.var = torch.ones(dim, dtype=torch.float32, device=samples.device)
        sample_mean = torch.mean(samples, dim=0, keepdims=True)
        sample_var = torch.var(samples, dim=0, keepdims=True)
        delta_mean = sample_mean - self.mean
        new_mean = self.mean + delta_mean * sample_count / (self.tot_count + sample_count)
        prev_var = self.var * self.tot_count
        sample_var = sample_var * sample_count
        new_var = prev_var + sample_var + delta_mean * delta_mean * self.tot_count * sample_count / (
            self.tot_count + sample_count)
        new_var = new_var / (self.tot_count + sample_count)
        self.mean = new_mean
        self.var = new_var
        self.var[self.var < 1e-12] = 1.0
        self.tot_count += sample_count

    def transform(self, data):
        return (data - self.mean) / torch.sqrt(self.var)


class Scheduler:
    """Port of common.scheduler.Scheduler (linear schedule)."""

    def __init__(self, initial_val, start_timestep=None, end_timestep=None,
                 target_val=None, schedule_type="linear"):
        assert schedule_type in ["linear", "identical"]
        if schedule_type == "linear":
            assert target_val is not None and start_timestep is not None and end_timestep is not None
        self.initial_val = initial_val
        self.target_val = target_val
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.curr_timestep = -1
        self.schedule_type = schedule_type

    def next(self):
        self.curr_timestep += 1
        if self.schedule_type == "linear":
            if self.curr_timestep >= self.end_timestep:
                return self.target_val
            elif self.curr_timestep <= self.start_timestep:
                return self.initial_val
            return self.initial_val + (self.target_val - self.initial_val) * (
                (self.curr_timestep - self.start_timestep) * 1.0
                / (self.end_timestep - self.start_timestep))
        elif self.schedule_type == "identical":
            return self.initial_val
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# SAC agent (port of mbpo/agent.py MBPOAgent)
# --------------------------------------------------------------------------- #
class SACAgent(nn.Module):
    """MBPOAgent port.  ``update`` keeps the upstream op order exactly
    (including q-targets under no_grad, the joint (q1+q2).backward before both
    optimizer steps, alpha loss detached inside, soft target update at tau)."""

    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high,
                 sac_cfg: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        hidden = sac_cfg.get("hidden", (256, 256))
        lr = float(sac_cfg.get("lr", 3e-4))
        self.gamma = float(sac_cfg.get("gamma", 0.99))
        self.target_smoothing_tau = float(sac_cfg.get("tau", 0.005))
        # python float until the first entropy-tuning step swaps it for
        # log_alpha.exp() (upstream behaviour)
        self.alpha = float(sac_cfg.get("alpha", 0.2))
        self.reward_scale = float(sac_cfg.get("reward_scale", 1.0))

        self.q1_network = MLP(obs_dim + act_dim, 1, hidden, "relu", "identity")
        self.q2_network = MLP(obs_dim + act_dim, 1, hidden, "relu", "identity")
        self.target_q1_network = MLP(obs_dim + act_dim, 1, hidden, "relu", "identity")
        self.target_q2_network = MLP(obs_dim + act_dim, 1, hidden, "relu", "identity")
        self.policy_network = GaussianPolicyNetwork(obs_dim, act_dim, hidden, "relu", "identity")
        self.policy_network.set_action_range(act_low, act_high)

        soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        self.to(device)

        self.q1_optimizer = torch.optim.Adam(self.q1_network.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2_network.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.automatic_entropy_tuning = bool(sac_cfg.get("automatic_entropy_tuning", True))
        target_entropy = sac_cfg.get("target_entropy", "auto")
        if target_entropy == "auto" or target_entropy is None:
            self.target_entropy = -float(act_dim)
        else:
            self.target_entropy = float(target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
        self.tot_update_count = 0

    # -- upstream API ------------------------------------------------------ #
    def update(self, data_batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)

        reward_batch = reward_batch * self.reward_scale

        # critic loss
        curr_q1 = self.q1_network([obs_batch, action_batch])
        curr_q2 = self.q2_network([obs_batch, action_batch])
        with torch.no_grad():
            next_sample = self.policy_network.sample(next_obs_batch)
            next_obs_action = next_sample["action"]
            next_obs_log_prob = next_sample["log_prob"]
            target_q1_value = self.target_q1_network([next_obs_batch, next_obs_action])
            target_q2_value = self.target_q2_network([next_obs_batch, next_obs_action])
            next_obs_min_q = torch.min(target_q1_value, target_q2_value)
            target_q = (next_obs_min_q - self.alpha * next_obs_log_prob)
            target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # policy loss
        new_sample = self.policy_network.sample(obs_batch)
        new_curr_obs_action = new_sample["action"]
        new_curr_obs_log_prob = new_sample["log_prob"]
        new_curr_obs_q1_value = self.q1_network([obs_batch, new_curr_obs_action])
        new_curr_obs_q2_value = self.q2_network([obs_batch, new_curr_obs_action])
        new_min_curr_obs_q_value = torch.min(new_curr_obs_q1_value, new_curr_obs_q2_value)
        policy_loss = ((self.alpha * new_curr_obs_log_prob) - new_min_curr_obs_q_value).mean()
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_obs_log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
        else:
            alpha_loss = 0.

        self.policy_optimizer.zero_grad()
        (policy_loss + alpha_loss).backward()
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning:
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.tot_update_count += 1
        self.update_target_network()

        # NOTE (performance-only deviation): upstream materialises every loss
        # with .item() here; each .item() is a host sync, which dominates the
        # update time when the GPU is co-tenanted.  We return detached 0-dim
        # tensors and the trainer materialises them once per epoch — the logged
        # values are identical.
        return {
            "loss/q1": q1_loss.detach(),
            "loss/q2": q2_loss.detach(),
            "loss/policy": policy_loss.detach(),
            "loss/entropy": alpha_loss.detach() if torch.is_tensor(alpha_loss)
            else alpha_loss,
            "misc/entropy_alpha": self.alpha.detach() if torch.is_tensor(self.alpha)
            else float(self.alpha),
            "misc/train_reward_mean": torch.mean(reward_batch).detach(),
            "misc/train_reward_var": torch.var(reward_batch).detach(),
        }

    def update_target_network(self):
        soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
        soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None,]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
        out = self.policy_network.sample(obs, deterministic)
        return {
            "action": out["action"].detach().cpu().numpy(),
            "log_prob": out["log_prob"].detach().cpu().numpy()[0],
        }

    @torch.no_grad()
    def deterministic_action(self, obs_np) -> np.ndarray:
        obs_t = torch.FloatTensor(np.asarray(obs_np, dtype=np.float32)).to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t[None,]
        return self.policy_network.sample(obs_t, deterministic=True)["action"].detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# ensemble dynamics model (ports of common/models.py + mbpo/transition_model.py)
# --------------------------------------------------------------------------- #
class EnsembleModel(nn.Module):
    """Port of common/models.py EnsembleModel (7 nets, 5 elites, 4x200 swish,
    max/min logvar bounds, per-layer L2 decay)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(200, 200, 200, 200),
                 ensemble_size: int = 7, num_elite: int = 5,
                 decay_weights=(2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4),
                 act: str = "swish", device=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = act_dim
        self.reward_dim = 1
        self.out_dim = obs_dim + self.reward_dim
        self.ensemble_size = int(ensemble_size)
        self.num_elite = int(num_elite)
        self.decay_weights = list(decay_weights)
        self.ensemble_models = nn.ModuleList([
            MLP(obs_dim + act_dim, self.out_dim * 2, hidden, act, "identity")
            for _ in range(self.ensemble_size)])
        assert len(self.decay_weights) == len(self.ensemble_models[0].weights), \
            "decay_weights must match the number of Linear layers"
        self.elite_model_idxes = np.arange(self.num_elite)  # upstream: [0..num_elite)
        self.max_logvar = nn.Parameter(torch.ones((1, self.out_dim)) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones((1, self.out_dim)) * 10.0)
        if device is not None:
            self.to(device)

    def predict(self, input):
        if not torch.is_tensor(input):
            input = torch.FloatTensor(np.asarray(input)).to(
                next(self.parameters()).device)
        if len(input.shape) == 3:
            model_outputs = [net(ip) for ip, net in zip(torch.unbind(input), self.ensemble_models)]
        elif len(input.shape) == 2:
            model_outputs = [net(input) for net in self.ensemble_models]
        else:
            raise ValueError(f"bad model input shape {tuple(input.shape)}")
        predictions = torch.stack(model_outputs)  # [E, B, 2*out_dim]

        mean = predictions[:, :, : self.out_dim]
        logvar = predictions[:, :, self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_decay_loss(self):
        decay_losses = []
        for model_net in self.ensemble_models:
            curr_net_decay_losses = [
                decay_weight * torch.sum(torch.square(weight))
                for decay_weight, weight in zip(self.decay_weights, model_net.weights)]
            decay_losses.append(torch.sum(torch.stack(curr_net_decay_losses)))
        return torch.sum(torch.stack(decay_losses))

    def load_state_dicts(self, state_dicts):
        for i in range(self.ensemble_size):
            self.ensemble_models[i].load_state_dict(state_dicts[i])


def make_termination_fn(env_name: str):
    """Vendored from mbpo/transition_model.py ``_termination_fn``.

    The Hopper / Walker2d / HalfCheetah branches are replicated verbatim
    (including the no-op ``np.abs(next_obs[:, 1:] < 100)`` Hopper term); the
    only change is that the v2/v3 name lists also accept the -v4 ids (upstream
    raises NotImplementedError for v4, but v4 termination semantics are
    identical and the RMBPO paper trains on v4).
    """

    def _termination_fn(env_name, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        if env_name in ["Hopper-v2", "Hopper-v3", "Hopper-v4"]:
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)
            done = ~not_done
        elif env_name in ["Walker2d-v2", "Walker2d-v3", "Walker2d-v4"]:
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            return done
        elif "Swimmer" in env_name or "HalfCheetah" in env_name:  # No done for these
            return np.array([False for _ in obs])
        else:
            raise NotImplementedError(f"termination_fn not implemented for {env_name}")
        return done

    return lambda env_name, obs, act, next_obs: _termination_fn(env_name, obs, act, next_obs)


def make_termination_fn_torch(env_name: str):
    """Torch twin of ``make_termination_fn`` (identical checks, device-side).

    Used by the RMBPO auxiliary update: the numpy termination function forces a
    host sync per batch (pred_state -> cpu), which dominates the aux round time
    when the GPU is co-tenanted.  The Hopper quirk term
    ``np.abs(next_obs[:, 1:] < 100).all(-1)`` is a no-op (always True) and is
    therefore omitted here; everything else matches ``_termination_fn``.
    """

    def _fn(next_obs: torch.Tensor) -> torch.Tensor:
        if env_name in ["Hopper-v2", "Hopper-v3", "Hopper-v4"]:
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = torch.isfinite(next_obs).all(dim=-1) \
                       & (height > .7) \
                       & (torch.abs(angle) < .2)
            return ~not_done
        elif env_name in ["Walker2d-v2", "Walker2d-v3", "Walker2d-v4"]:
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) & (height < 2.0) & (angle > -1.0) & (angle < 1.0)
            return ~not_done
        elif "Swimmer" in env_name or "HalfCheetah" in env_name:
            return torch.zeros(next_obs.shape[0], dtype=torch.bool,
                               device=next_obs.device)
        raise NotImplementedError(f"termination_fn not implemented for {env_name}")

    return _fn


class TransitionModel:
    """Port of mbpo/transition_model.py TransitionModel (the nominal model ptheta)."""

    def __init__(self, obs_dim: int, act_dim: int, env_name: str,
                 model_cfg: dict, device: torch.device):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = act_dim
        self.env_name = env_name
        self.model = EnsembleModel(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden=model_cfg.get("hidden", (200, 200, 200, 200)),
            ensemble_size=model_cfg.get("ensemble", 7),
            num_elite=model_cfg.get("elites", 5),
            decay_weights=model_cfg.get(
                "decay_weights", (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4)),
            act=model_cfg.get("act_fn", "swish"), device=device)
        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(model_cfg.get("lr", 1e-3)))
        self.model_batch_size = int(model_cfg.get("batch", 256))
        self.holdout_ratio = float(model_cfg.get("holdout_ratio", 0.2))
        self.inc_var_loss = bool(model_cfg.get("inc_var_loss", True))
        self.use_weight_decay = True
        self.obs_normalizer = StandardNormalizer(device)
        self.act_normalizer = StandardNormalizer(device)
        self.model_train_timesteps = 0
        self._termination_fn = make_termination_fn(env_name)
        self.model_best_snapshots = None
        self.best_snapshot_losses = None

    # -- normalizers -------------------------------------------------------- #
    def reset_normalizers(self):
        self.obs_normalizer.reset()
        self.act_normalizer.reset()

    def update_normalizer(self, obs, action):
        self.obs_normalizer.update(obs)
        self.act_normalizer.update(action)

    def transform_obs_action(self, obs, action):
        obs = self.obs_normalizer.transform(obs)
        action = self.act_normalizer.transform(action)
        return obs, action

    # -- losses / training -------------------------------------------------- #
    def model_loss(self, predictions, groundtruths, mse_only=False):
        pred_means, pred_logvars = predictions
        if self.inc_var_loss and not mse_only:
            # Average over batch and dim, sum over ensembles.
            inv_var = torch.exp(-pred_logvars)
            mse_losses = torch.mean(
                torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=-1), dim=-1)
            var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        elif mse_only:
            mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            var_losses = None
        else:
            assert 0
        return mse_losses, var_losses

    def update(self, data_batch):
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            itemgetter("obs", "action", "next_obs", "reward")(data_batch)

        delta_obs_batch = next_obs_batch - obs_batch
        obs_batch, action_batch = self.transform_obs_action(obs_batch, action_batch)

        model_input = torch.cat([obs_batch, action_batch], dim=-1)
        predictions = self.model.predict(model_input)
        groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=-1)
        train_mse_losses, train_var_losses = self.model_loss(predictions, groundtruths)
        train_mse_loss = torch.sum(train_mse_losses)
        train_var_loss = torch.sum(train_var_losses)
        train_transition_loss = train_mse_loss + train_var_loss
        train_transition_loss += 0.01 * torch.sum(self.model.max_logvar) \
            - 0.01 * torch.sum(self.model.min_logvar)
        if self.use_weight_decay:
            decay_loss = self.model.get_decay_loss()
            train_transition_loss = train_transition_loss + decay_loss
        else:
            decay_loss = None
        self.model_optimizer.zero_grad()
        train_transition_loss.backward()
        self.model_optimizer.step()
        self.model_train_timesteps += 1

        # performance-only deviation: detached tensors instead of .item() per
        # call (host sync); materialised once per round by the trainer
        return {
            "loss/train_model_loss_mse": train_mse_loss.detach(),
            "loss/train_model_loss_var": train_var_loss.detach(),
            "loss/decay_loss": decay_loss.detach() if decay_loss is not None
            else 0.0,
            "misc/max_std": self.model.max_logvar.mean().detach(),
            "misc/min_std": self.model.min_logvar.mean().detach(),
        }

    @torch.no_grad()
    def eval_data(self, data, update_elite_models=False):
        obs_list, action_list, next_obs_list, reward_list = \
            itemgetter("obs", "action", "next_obs", "reward")(data)
        delta_obs_list = next_obs_list - obs_list
        obs_list, action_list = self.transform_obs_action(obs_list, action_list)
        model_input = torch.cat([obs_list, action_list], dim=-1)
        predictions = minibatch_inference(
            args=[model_input], rollout_fn=self.model.predict,
            batch_size=10000, cat_dim=1)
        groundtruths = torch.cat((delta_obs_list, reward_list), dim=1)
        eval_mse_losses, _ = self.model_loss(predictions, groundtruths, mse_only=True)
        if update_elite_models:
            elite_idx = np.argsort(eval_mse_losses.detach().cpu().numpy())
            self.model.elite_model_idxes = elite_idx[: self.model.num_elite]
        return eval_mse_losses.detach().cpu().numpy(), None

    # -- best-snapshot early stopping ---------------------------------------- #
    def reset_best_snapshots(self):
        self.model_best_snapshots = [
            copy.deepcopy(self.model.ensemble_models[idx].state_dict())
            for idx in range(self.model.ensemble_size)]
        self.best_snapshot_losses = [1e10 for _ in range(self.model.ensemble_size)]

    def update_best_snapshots(self, val_losses):
        updated = False
        for i in range(len(val_losses)):
            current_loss = val_losses[i]
            best_loss = self.best_snapshot_losses[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > 0.01:
                self.best_snapshot_losses[i] = current_loss
                self.save_model_snapshot(i)
                updated = True
        return updated

    def save_model_snapshot(self, idx):
        self.model_best_snapshots[idx] = copy.deepcopy(
            self.model.ensemble_models[idx].state_dict())

    def load_best_snapshots(self):
        self.model.load_state_dicts(self.model_best_snapshots)

    # -- rollout prediction -------------------------------------------------- #
    @torch.no_grad()
    def predict(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None,]
            act = act[None,]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(self.device)

        scaled_obs, scaled_act = self.transform_obs_action(obs, act)
        model_input = torch.cat([scaled_obs, scaled_act], dim=-1)
        pred_diff_means, pred_diff_logvars = self.model.predict(model_input)
        pred_diff_means = pred_diff_means.detach().cpu().numpy()
        obs = obs.detach().cpu().numpy()
        act = act.detach().cpu().numpy()
        ensemble_model_stds = pred_diff_logvars.exp().sqrt().detach().cpu().numpy()
        if deterministic:
            pass
        else:
            pred_diff_means = pred_diff_means + \
                np.random.normal(size=pred_diff_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = pred_diff_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)
        pred_diff_samples = pred_diff_means[model_idxes, batch_idxes]

        next_obs, rewards = pred_diff_samples[:, :-1] + obs, pred_diff_samples[:, -1]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        return next_obs, rewards, terminals


# --------------------------------------------------------------------------- #
# replay buffer (port of common/buffer.py ReplayBuffer + vectorized add_traj)
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Port of common/buffer.py ReplayBuffer.

    ``sample`` semantics replicated exactly (allow_duplicate=True uses
    ``np.random.choice`` with replacement; otherwise python ``random.sample``
    with batch clipped to the fill level); ``add_traj`` is vectorized with
    identical ring-buffer semantics."""

    def __init__(self, obs_dim: int, obs_dtype, act_dim: int, act_dtype,
                 max_buffer_size: int):
        self.max_buffer_size = int(max_buffer_size)
        self.curr = 0
        self.obs_shape = (obs_dim,)
        self.action_dim = act_dim
        self.obs_dtype = np.dtype(obs_dtype)
        self.action_dtype = np.dtype(act_dtype)
        self.obs_buffer = np.zeros((self.max_buffer_size,) + self.obs_shape,
                                   dtype=self.obs_dtype)
        self.action_buffer = np.zeros((self.max_buffer_size, self.action_dim),
                                      dtype=self.action_dtype)
        self.next_obs_buffer = np.zeros((self.max_buffer_size,) + self.obs_shape,
                                        dtype=self.obs_dtype)
        self.reward_buffer = np.zeros((self.max_buffer_size,), dtype=np.float32)
        self.done_buffer = np.zeros((self.max_buffer_size,), dtype=np.int8)
        self.truncated_buffer = np.zeros((self.max_buffer_size,), dtype=np.int8)
        self.max_sample_size = 0

    def clear(self):
        self.max_sample_size = 0
        self.curr = 0

    def add_transition(self, obs, action, next_obs, reward, done, truncated=None):
        if truncated is None:
            truncated = done
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        self.truncated_buffer[self.curr] = truncated
        self.curr = (self.curr + 1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size + 1, self.max_buffer_size)

    def add_traj(self, obs_list, action_list, next_obs_list, reward_list,
                 done_list, truncated_list=None):
        """Vectorized port of the upstream per-sample loop.

        Ring semantics preserved: sample j lands at (curr0 + j) % size, the
        surviving content is the last ``size`` samples, and the final pointer is
        (curr0 + n) % size.  The destination index set is a permutation (never
        has duplicates), so the batched assignment is order-independent and
        identical to the sequential loop."""
        if truncated_list is None:
            truncated_list = done_list
        n = len(obs_list)
        if n == 0:
            return
        size = self.max_buffer_size
        final_curr = (self.curr + n) % size
        m = min(n, size)
        idx = (final_curr - m + np.arange(m)) % size
        self.obs_buffer[idx] = np.asarray(obs_list)[n - m:]
        self.action_buffer[idx] = np.asarray(action_list)[n - m:]
        self.next_obs_buffer[idx] = np.asarray(next_obs_list)[n - m:]
        self.reward_buffer[idx] = np.asarray(reward_list, dtype=np.float64)[n - m:]
        self.done_buffer[idx] = np.asarray(done_list, dtype=np.float64)[n - m:]
        self.truncated_buffer[idx] = np.asarray(truncated_list, dtype=np.float64)[n - m:]
        self.curr = final_curr
        self.max_sample_size = min(self.max_sample_size + n, size)

    def sample(self, batch_size, to_tensor=True, device=None, allow_duplicate=False):
        if not allow_duplicate:
            if batch_size > self.max_sample_size:
                warnings.warn("Sampling larger than buffer size")
            batch_size = min(self.max_sample_size, batch_size)

        if allow_duplicate:
            indices = np.random.choice(range(self.max_sample_size), batch_size)
        else:
            indices = random.sample(range(self.max_sample_size), batch_size)

        obs_batch = self.obs_buffer[indices]
        action_batch = self.action_buffer[indices]
        next_obs_batch = self.next_obs_buffer[indices]
        reward_batch = self.reward_buffer[indices]
        done_batch = self.done_buffer[indices]
        truncated_batch = self.truncated_buffer[indices]

        if to_tensor:
            assert device is not None
            obs_batch = torch.FloatTensor(obs_batch).to(device)
            action_batch = torch.FloatTensor(action_batch).to(device)
            next_obs_batch = torch.FloatTensor(next_obs_batch).to(device)
            reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(done_batch).to(device).unsqueeze(1)
            truncated_batch = torch.FloatTensor(truncated_batch).to(device).unsqueeze(1)
        return dict(
            obs=obs_batch,
            action=action_batch,
            next_obs=next_obs_batch,
            reward=reward_batch,
            done=done_batch,
            truncated=truncated_batch,
        )

    def resize(self, new_size):
        if new_size < self.max_buffer_size:
            self.obs_buffer = self.obs_buffer[:new_size]
            self.action_buffer = self.action_buffer[:new_size]
            self.next_obs_buffer = self.next_obs_buffer[:new_size]
            self.reward_buffer = self.reward_buffer[:new_size]
            self.done_buffer = self.done_buffer[:new_size]
            self.truncated_buffer = self.truncated_buffer[:new_size]
            if self.curr >= new_size:  # buffer has overflowed
                self.curr = 0
                self.max_sample_size = new_size
            self.max_buffer_size = new_size
        elif new_size == self.max_buffer_size:
            return
        else:
            addition_size = new_size - self.max_buffer_size
            self.obs_buffer = np.concatenate(
                [self.obs_buffer, np.zeros((addition_size,) + self.obs_shape, dtype=self.obs_dtype)], axis=0)
            self.action_buffer = np.concatenate(
                [self.action_buffer, np.zeros((addition_size, self.action_dim), dtype=self.action_dtype)], axis=0)
            self.next_obs_buffer = np.concatenate(
                [self.next_obs_buffer, np.zeros((addition_size,) + self.obs_shape, dtype=self.obs_dtype)], axis=0)
            self.reward_buffer = np.concatenate(
                [self.reward_buffer, np.zeros((addition_size,), dtype=np.float32)], axis=0)
            self.done_buffer = np.concatenate(
                [self.done_buffer, np.zeros((addition_size,), dtype=np.int8)], axis=0)
            self.truncated_buffer = np.concatenate(
                [self.truncated_buffer, np.zeros((addition_size,), dtype=np.int8)], axis=0)
            if self.curr < self.max_sample_size:
                self.curr = self.max_sample_size
            self.max_buffer_size = new_size

    def arrays(self):
        return dict(obs=self.obs_buffer, action=self.action_buffer,
                    next_obs=self.next_obs_buffer, reward=self.reward_buffer,
                    done=self.done_buffer, truncated=self.truncated_buffer,
                    curr=self.curr, max_sample_size=self.max_sample_size,
                    max_buffer_size=self.max_buffer_size)

    def load_arrays(self, d):
        self.obs_buffer = np.asarray(d["obs"])
        self.action_buffer = np.asarray(d["action"])
        self.next_obs_buffer = np.asarray(d["next_obs"])
        self.reward_buffer = np.asarray(d["reward"])
        self.done_buffer = np.asarray(d["done"])
        self.truncated_buffer = np.asarray(d["truncated"])
        self.curr = int(d["curr"])
        self.max_sample_size = int(d["max_sample_size"])
        self.max_buffer_size = int(d["max_buffer_size"])
        assert self.obs_buffer.shape[0] == self.max_buffer_size


# --------------------------------------------------------------------------- #
# trainer (port of mbpo/trainer.py MBPOTrainer + phase-B plumbing)
# --------------------------------------------------------------------------- #
class MbpoTrainer:
    """Shared MBPO / RMBPO training skeleton.

    Port of ``mbpo/trainer.py`` (warmup random collection, model round every
    ``train_model_interval`` real steps followed by 100k rollout transitions,
    UTD=2 mixed SAC batches 95% model / 5% env, done-vs-truncated bookkeeping,
    model-buffer resize to one epoch of rollouts).  Phase-B additions: budget
    bookkeeping, counters.json / progress.jsonl, resumable checkpoints and the
    ``_after_model_training`` hook that RMBPO overrides to train the auxiliary
    model.  Everything an MBPO job touches is identical for an RMBPO job in
    bypass mode (no aux net is even constructed there).
    """

    ALGO = "mbpo"

    def __init__(self, task: str, seed: int, cfg: dict, job_dir: str,
                 device=None, use_aux: bool = False, bypass_aux: bool = False):
        self.task = task
        self.seed = int(seed)
        self.cfg = cfg
        self.job_dir = job_dir
        self.device = device if device is not None else resolve_device(cfg)
        self._ctor_flags = (bool(use_aux), bool(bypass_aux))
        self.use_aux = bool(use_aux) and not bool(bypass_aux)
        self.bypass_aux = bool(bypass_aux)

        # --- config unpacking (defaults mirror mujoco_baselines.config) ---- #
        sac = dict(cfg.get("sac", {}) or {})
        model_cfg = dict(cfg.get("model", {}) or {})
        rollout_cfg = dict(cfg.get("rollout", {}) or {})
        self.sac_cfg = sac
        self.model_cfg = model_cfg
        self.agent_batch_size = int(sac.get("batch", 256))
        self.model_batch_size = int(model_cfg.get("batch", 256))
        self.rollout_batch_size = int(rollout_cfg.get("batch", 100_000))
        self.rollout_mini_batch_size = int(rollout_cfg.get("mini_batch", 10_000))
        self.model_retain_epochs = int(rollout_cfg.get("retain_epochs", 1))
        self.model_env_ratio = float(rollout_cfg.get("model_env_ratio", 0.95))
        horizon = int(rollout_cfg.get("horizon", 1))
        self.num_env_steps_per_epoch = int(cfg.get("epoch_length", 1000))
        tmi = cfg.get("train_model_interval", 250)
        self.train_model_interval = int(tmi[task] if isinstance(tmi, dict) else tmi)
        self.num_agent_updates_per_env_step = int(cfg.get("utd", 2))
        self.max_model_update_epochs_to_improve = int(
            cfg.get("max_model_update_epochs_to_improve", 5))
        self.max_model_train_iterations = model_cfg.get("max_model_train_iterations", None)
        if self.max_model_train_iterations in ("None", None):
            self.max_model_train_iterations = float("inf")
        self.max_trajectory_length = int(cfg.get("max_trajectory_length", 1000))
        self.warmup_timesteps = int(cfg.get("warmup", 5000))
        self.holdout_ratio = float(model_cfg.get("holdout_ratio", 0.2))
        self.budget = int(cfg.get("budget", 3_000_000))
        self.save_every = int(cfg.get("save_state_every", 250_000))
        self.apply_model_buffer_resize = bool(cfg.get("apply_model_buffer_resize", True))

        # --- env ----------------------------------------------------------- #
        self.env = make_train_env(task, self.seed)
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.obs_dim = int(self.obs_space.shape[0])
        self.act_dim = int(self.action_space.shape[0])
        self.obs_dtype = self.obs_space.dtype
        self.act_dtype = self.action_space.dtype
        self.env_name = PERTURBED_IDS[task]

        # --- components ----------------------------------------------------- #
        self.agent = SACAgent(self.obs_dim, self.act_dim,
                              self.action_space.low, self.action_space.high,
                              sac, self.device)
        self.transition_model = TransitionModel(self.obs_dim, self.act_dim,
                                                 self.env_name, model_cfg, self.device)
        self.env_buffer = ReplayBuffer(self.obs_dim, self.obs_dtype, self.act_dim,
                                       self.act_dtype, int(cfg.get("env_buffer", 1_000_000)))
        self.model_buffer = ReplayBuffer(self.obs_dim, self.obs_dtype, self.act_dim,
                                         self.act_dtype, int(cfg.get("model_buffer", 2_000_000)))
        # upstream rollout_step_scheduler (initial=target=horizon => constant)
        self.rollout_step_generator = Scheduler(
            initial_val=horizon, target_val=horizon, start_timestep=20,
            end_timestep=150, schedule_type="linear")
        self._model_rollout_steps = horizon

        # --- counters / loop state ------------------------------------------- #
        self.real_transitions = 0
        self.model_transitions = 0
        self.sac_updates = 0
        self.model_train_iterations = 0
        self.aux_updates = 0
        self.model_rounds = 0
        self.aux_rounds = 0
        self.model_train_epochs_total = 0
        self._epoch = 0
        self._obs = None
        self._traj_length = 0
        self._traj_return = 0.0
        self._last_train_return = None
        self._ep_returns_in_epoch = []
        self._wall_s = 0.0
        self._next_save = self.save_every
        self._run_start = None

        os.makedirs(os.path.join(self.job_dir, "ckpt"), exist_ok=True)
        self.final_path = os.path.join(self.job_dir, "ckpt", "final.pt")
        self.counters_path = os.path.join(self.job_dir, "counters.json")
        self.progress_path = os.path.join(self.job_dir, "progress.jsonl")

    # ------------------------------------------------------------------ #
    # warmup (port of MBPOTrainer.warmup)
    # ------------------------------------------------------------------ #
    def warmup(self):
        obs, info = self.env.reset()
        for step in range(self.warmup_timesteps):
            if self.real_transitions >= self.budget:
                break
            action = self.env.action_space.sample()
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.real_transitions += 1
            self.env_buffer.add_transition(obs, action, next_obs, reward, done, truncated)
            obs = next_obs
            if done:
                obs, info = self.env.reset()

    # ------------------------------------------------------------------ #
    # model round (port of MBPOTrainer.train_model, returns the split for the aux hook)
    # ------------------------------------------------------------------ #
    def train_model(self):
        num_train_data = int(self.env_buffer.max_sample_size * (1.0 - self.holdout_ratio))
        env_data = self.env_buffer.sample(self.env_buffer.max_sample_size,
                                          device=self.device)
        train_data, eval_data = {}, {}
        for key in env_data.keys():
            train_data[key] = env_data[key][:num_train_data]
            eval_data[key] = env_data[key][num_train_data:]
        self.transition_model.reset_normalizers()
        self.transition_model.update_normalizer(train_data['obs'], train_data['action'])

        model_train_iters = 0
        model_train_epochs = 0
        num_epochs_since_prev_best = 0
        model_log_infos = {}
        self.transition_model.reset_best_snapshots()
        eval_mse_losses, _ = self.transition_model.eval_data(eval_data, update_elite_models=False)
        self.transition_model.update_best_snapshots(eval_mse_losses)
        while True:
            for train_data_batch in dict_batch_generator(train_data, self.model_batch_size):
                model_log_infos = self.transition_model.update(train_data_batch)
                model_train_iters += 1
                self.model_train_iterations += 1
            eval_mse_losses, _ = self.transition_model.eval_data(eval_data, update_elite_models=False)
            updated = self.transition_model.update_best_snapshots(eval_mse_losses)
            num_epochs_since_prev_best += 1
            if updated is True:
                model_train_epochs += num_epochs_since_prev_best
                num_epochs_since_prev_best = 0
            if num_epochs_since_prev_best >= self.max_model_update_epochs_to_improve \
                    or model_train_iters > self.max_model_train_iterations:
                break
        self.transition_model.load_best_snapshots()

        # evaluate data to update the elite models
        eval_mse_losses, _ = self.transition_model.eval_data(eval_data, update_elite_models=True)
        logs = materialize(model_log_infos)
        logs["misc/norm_obs_mean"] = float(torch.mean(self.transition_model.obs_normalizer.mean).item())
        logs["misc/norm_obs_var"] = float(torch.mean(self.transition_model.obs_normalizer.var).item())
        logs["misc/norm_act_mean"] = float(torch.mean(self.transition_model.act_normalizer.mean).item())
        logs["misc/norm_act_var"] = float(torch.mean(self.transition_model.act_normalizer.var).item())
        logs["misc/model_train_epochs"] = model_train_epochs
        logs["misc/model_train_train_steps"] = model_train_iters
        logs["model_holdout_mse"] = float(np.mean(eval_mse_losses))
        return logs, train_data, eval_data

    def _after_model_training(self, train_data, eval_data):
        """Hook: RMBPO trains the auxiliary model here (same cadence)."""
        return {}

    # ------------------------------------------------------------------ #
    # rollouts (port of MBPOTrainer.rollout_model)
    # ------------------------------------------------------------------ #
    def resize_model_buffer(self, rollout_length):
        """Upstream MBPOTrainer.resize_model_buffer (vendored verbatim).

        NOTE (regime consequence): with the budget-scaled train_model_interval
        this sizes the model replay to int(rollout_batch * 1000 / interval)
        model transitions — hopper 16,666 / halfcheetah 53,333 / walker2d 40,000
        (the paper regime's 400,000 at interval 250) — i.e. one epoch worth of
        model data, which is exactly the retain_epochs=1 semantics of the
        upstream trainer.  The 2M ``model_buffer`` config value is the initial
        allocation, overridden here just like upstream.  Set
        ``cfg['apply_model_buffer_resize'] = False`` to keep the configured
        size instead (the generated-then-overwritten rollout count and the
        model-data reuse per real step are identical either way; only the
        freshness window of the model replay changes)."""
        if not getattr(self, "apply_model_buffer_resize", True):
            return
        rollouts_per_epoch = self.rollout_batch_size * self.num_env_steps_per_epoch \
            / self.train_model_interval
        new_model_buffer_size = int(rollout_length * rollouts_per_epoch * self.model_retain_epochs)
        self.model_buffer.resize(new_model_buffer_size)

    @torch.no_grad()
    def rollout_model(self):
        rollout_data_batch = self.env_buffer.sample(
            self.rollout_batch_size, to_tensor=False, allow_duplicate=True)
        obs_batch = rollout_data_batch['obs']
        rollout_batch_nums = int(np.ceil(self.rollout_batch_size / self.rollout_mini_batch_size))
        for rollout_batch_id in range(rollout_batch_nums):
            obs_minibatch = obs_batch[
                rollout_batch_id * self.rollout_mini_batch_size:
                min(len(obs_batch), (rollout_batch_id + 1) * self.rollout_mini_batch_size)]
            for rollout_step in range(self._model_rollout_steps):
                action_minibatch = self.agent.select_action(obs_minibatch)['action']
                next_obs_minibatch, reward_minibatch, done_minibatch = \
                    self.transition_model.predict(obs_minibatch, action_minibatch)
                done_minibatch = np.asarray([float(d) for d in done_minibatch])
                self.model_buffer.add_traj(obs_minibatch, action_minibatch,
                                           next_obs_minibatch, reward_minibatch,
                                           done_minibatch)
                self.model_transitions += len(obs_minibatch)
                obs_minibatch = np.array([next_obs_pred for next_obs_pred, done_pred
                                          in zip(next_obs_minibatch, done_minibatch)
                                          if not done_pred])
                if len(obs_minibatch) == 0:
                    break

    # ------------------------------------------------------------------ #
    # SAC batch (port of MBPOTrainer.train_agent)
    # ------------------------------------------------------------------ #
    def train_agent(self):
        train_agent_model_batch_size = int(self.agent_batch_size * self.model_env_ratio)
        train_agent_env_batch_size = self.agent_batch_size - train_agent_model_batch_size
        model_data_batch = self.model_buffer.sample(
            train_agent_model_batch_size, device=self.device)
        env_data_batch = self.env_buffer.sample(
            train_agent_env_batch_size, device=self.device)
        data_batch = merge_data_batch(model_data_batch, env_data_batch)
        loss_dict = self.agent.update(data_batch)
        return loss_dict

    # ------------------------------------------------------------------ #
    # checkpointing / counters / progress
    # ------------------------------------------------------------------ #
    def counters(self) -> dict:
        return {
            "algo": self.ALGO,
            "task": self.task,
            "seed": self.seed,
            "real_transitions": int(self.real_transitions),
            "model_transitions": int(self.model_transitions),
            "sac_updates": int(self.sac_updates),
            "model_train_iterations": int(self.model_train_iterations),
            "aux_updates": int(self.aux_updates),
            "model_rollout_rounds": int(self.model_rounds),
            "aux_rounds": int(self.aux_rounds),
            "model_train_epochs": int(getattr(self, "model_train_epochs_total", 0)),
            "last_train_return": None if self._last_train_return is None
            else round(float(self._last_train_return), 4),
            "budget": int(self.budget),
            "wall_s": round(self._wall_s_total(), 3),
            "finished": bool(self.real_transitions >= self.budget),
            "device": str(self.device),
        }

    def _wall_s_total(self):
        total = self._wall_s
        if self._run_start is not None:
            total += time.time() - self._run_start
        return total

    def _write_counters(self):
        tmp = self.counters_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.counters(), f, indent=2)
        os.replace(tmp, self.counters_path)

    def _append_progress(self, rec: dict):
        with open(self.progress_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _checkpoint_state(self) -> dict:
        tm = self.transition_model
        rng = {
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        base = self.env.unwrapped
        st = {
            "format": 1,
            "algo": self.ALGO,
            "task": self.task,
            "seed": self.seed,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_low": np.asarray(self.action_space.low, dtype=np.float32),
            "act_high": np.asarray(self.action_space.high, dtype=np.float32),
            "sac_cfg": self.sac_cfg,
            "counters": self.counters(),
            "epoch": self._epoch,
            "scheduler_timestep": self.rollout_step_generator.curr_timestep,
            "model_rollout_steps": self._model_rollout_steps,
            "next_save": self._next_save,
            "agent": self.agent.state_dict(),
            "agent_opt": {
                "q1": self.agent.q1_optimizer.state_dict(),
                "q2": self.agent.q2_optimizer.state_dict(),
                "policy": self.agent.policy_optimizer.state_dict(),
                "alpha": self.agent.alpha_optim.state_dict(),
                "log_alpha": self.agent.log_alpha.detach().cpu(),
            },
            "transition_model": {
                "model": tm.model.state_dict(),
                "optimizer": tm.model_optimizer.state_dict(),
                "obs_norm_mean": tm.obs_normalizer.mean,
                "obs_norm_var": tm.obs_normalizer.var,
                "obs_norm_tot_count": int(tm.obs_normalizer.tot_count),
                "act_norm_mean": tm.act_normalizer.mean,
                "act_norm_var": tm.act_normalizer.var,
                "act_norm_tot_count": int(tm.act_normalizer.tot_count),
                "elites": np.asarray(tm.model.elite_model_idxes),
                "model_train_timesteps": tm.model_train_timesteps,
            },
            "env_buffer": self.env_buffer.arrays(),
            "model_buffer": self.model_buffer.arrays(),
            "env_state": {
                "obs": self._obs,
                "traj_length": self._traj_length,
                "traj_return": float(self._traj_return),
                "elapsed_steps": int(self.env.env._elapsed_steps),
                "qpos": np.copy(base.data.qpos),
                "qvel": np.copy(base.data.qvel),
                "current_params": tuple(self.env.current_params)
                if self.env.current_params is not None else None,
                "env_rng": base._np_random.bit_generator.state,
                "sampler_rng": self.env._sampler.rng.bit_generator.state,
            },
            "rng": rng,
        }
        st.update(self._aux_checkpoint_state())
        return st

    def _aux_checkpoint_state(self) -> dict:
        return {}

    def _state_files(self):
        files = glob.glob(os.path.join(self.job_dir, "ckpt", "state_*.pt"))
        def step_of(p):
            try:
                return int(os.path.basename(p)[len("state_"):-3])
            except ValueError:
                return -1
        return sorted(files, key=step_of)

    def _save_checkpoint(self, tag=None):
        step = tag if tag is not None else self.real_transitions
        path = os.path.join(self.job_dir, "ckpt", f"state_{int(step)}.pt")
        tmp = path + ".tmp"
        torch.save(self._checkpoint_state(), tmp)
        os.replace(tmp, path)
        # keep only the newest two state files
        files = self._state_files()
        for old in files[:-2]:
            try:
                os.remove(old)
            except OSError:
                pass
        self._write_counters()
        return path

    def _load_checkpoint(self, path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        c = st["counters"]
        self.real_transitions = int(c["real_transitions"])
        self.model_transitions = int(c["model_transitions"])
        self.sac_updates = int(c["sac_updates"])
        self.model_train_iterations = int(c["model_train_iterations"])
        self.aux_updates = int(c.get("aux_updates", 0))
        self.model_rounds = int(c.get("model_rollout_rounds", 0))
        self.aux_rounds = int(c.get("aux_rounds", 0))
        self.model_train_epochs_total = int(c.get("model_train_epochs", 0))
        self._wall_s = float(c.get("wall_s", 0.0))
        self._epoch = int(st["epoch"])
        self.rollout_step_generator.curr_timestep = int(st["scheduler_timestep"])
        self._model_rollout_steps = int(st["model_rollout_steps"])
        self._next_save = int(st.get("next_save", self.save_every))
        self._last_train_return = c.get("last_train_return", None)

        self.agent.load_state_dict(st["agent"])
        ao = st["agent_opt"]
        self.agent.q1_optimizer.load_state_dict(ao["q1"])
        self.agent.q2_optimizer.load_state_dict(ao["q2"])
        self.agent.policy_optimizer.load_state_dict(ao["policy"])
        self.agent.alpha_optim.load_state_dict(ao["alpha"])
        self.agent.log_alpha = ao["log_alpha"].to(self.device).requires_grad_(True)
        self.agent.alpha = self.agent.log_alpha.detach().exp()

        tm = st["transition_model"]
        self.transition_model.model.load_state_dict(tm["model"])
        self.transition_model.model.to(self.device)
        self.transition_model.model_optimizer.load_state_dict(tm["optimizer"])
        if tm["obs_norm_mean"] is None:
            self.transition_model.obs_normalizer.reset()
            self.transition_model.act_normalizer.reset()
        else:
            self.transition_model.obs_normalizer.mean = tm["obs_norm_mean"].to(self.device)
            self.transition_model.obs_normalizer.var = tm["obs_norm_var"].to(self.device)
            self.transition_model.obs_normalizer.tot_count = int(tm["obs_norm_tot_count"])
            self.transition_model.act_normalizer.mean = tm["act_norm_mean"].to(self.device)
            self.transition_model.act_normalizer.var = tm["act_norm_var"].to(self.device)
            self.transition_model.act_normalizer.tot_count = int(tm["act_norm_tot_count"])
        self.transition_model.model.elite_model_idxes = np.asarray(tm["elites"])
        self.transition_model.model_train_timesteps = int(tm["model_train_timesteps"])

        self.env_buffer.load_arrays(st["env_buffer"])
        self.model_buffer.load_arrays(st["model_buffer"])

        # env state (perturbation params, physics state, rng streams)
        es = st["env_state"]
        self.env.reset()
        base = self.env.unwrapped
        if es.get("current_params") is not None:
            self.env._apply(*es["current_params"])
        base.set_state(np.asarray(es["qpos"]), np.asarray(es["qvel"]))
        self.env.env._elapsed_steps = int(es["elapsed_steps"])
        base._np_random.bit_generator.state = es["env_rng"]
        self.env._sampler.rng.bit_generator.state = es["sampler_rng"]
        self._obs = np.asarray(es["obs"])
        self._traj_length = int(es["traj_length"])
        self._traj_return = float(es["traj_return"])

        # global RNGs (torch RNG states must stay on CPU — torch requirement)
        np.random.set_state(st["rng"]["numpy"])
        random.setstate(st["rng"]["python"])
        torch_state = st["rng"]["torch"]
        if torch.is_tensor(torch_state) and torch_state.device.type != "cpu":
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
        if self.device.type == "cuda" and "cuda" in st["rng"]:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])

        self._load_aux_checkpoint(st)
        return st

    def _load_aux_checkpoint(self, st):
        pass

    def _save_final(self):
        st = self._checkpoint_state()
        st["counters"] = self.counters()
        st["counters"]["finished"] = True
        tmp = self.final_path + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, self.final_path)
        self._write_counters()
        # state files are superseded by final.pt; keep the last one for debugging
        files = self._state_files()
        for old in files[:-1]:
            try:
                os.remove(old)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # main loop (port of MBPOTrainer.train, restructured around the budget)
    # ------------------------------------------------------------------ #
    def train(self):
        # finished already?  (idempotent entry for the runner)
        if os.path.exists(self.final_path):
            try:
                with open(self.counters_path) as f:
                    c = json.load(f)
                if int(c.get("real_transitions", 0)) >= self.budget:
                    print(f"[{self.ALGO}/{self.task}/seed{self.seed}] already finished "
                          f"({c['real_transitions']} transitions) — skipping", flush=True)
                    return
            except (OSError, ValueError):
                pass

        self._run_start = time.time()
        files = self._state_files()
        resumed = False
        if files:
            try:
                self._load_checkpoint(files[-1])
                resumed = True
                print(f"[{self.ALGO}/{self.task}/seed{self.seed}] resumed from "
                      f"{os.path.basename(files[-1])} at step {self.real_transitions}",
                      flush=True)
            except Exception as e:  # corrupted snapshot: restart from scratch
                print(f"[{self.ALGO}/{self.task}/seed{self.seed}] resume failed ({e}); "
                      f"restarting", flush=True)
                use_aux, bypass_aux = self._ctor_flags
                self.__init__(self.task, self.seed, self.cfg, self.job_dir,
                              self.device, use_aux=use_aux, bypass_aux=bypass_aux)
                self._run_start = time.time()
        if self.real_transitions >= self.budget:
            self._save_final()
            return

        if not resumed:
            set_global_seed(self.seed)
            obs, info = self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
            self.warmup()
            obs, info = self.env.reset()
            self._obs = obs
            self._traj_length = 0
            self._traj_return = 0.0
            # pre-resize the model buffer to its regime size (the epoch-0 resize
            # would do the same) so the post-warmup snapshot is not bloated by
            # the 2M-row initial allocation
            self.resize_model_buffer(self._model_rollout_steps)
            if self.real_transitions > 0:
                self._save_checkpoint()
            self._next_save = max(self.save_every,
                                  (self.real_transitions // self.save_every + 1) * self.save_every)

        print(f"[{self.ALGO}/{self.task}/seed{self.seed}] training to "
              f"{self.budget} real transitions on {self.device} "
              f"(interval={self.train_model_interval}, warmup={self.warmup_timesteps})",
              flush=True)

        while self.real_transitions < self.budget:
            # ---- epoch start (upstream: scheduler + model-buffer resize) ---- #
            new_model_rollout_steps = int(self.rollout_step_generator.next())
            if self._epoch == 0 or new_model_rollout_steps != self._model_rollout_steps:
                self.resize_model_buffer(new_model_rollout_steps)
                self._model_rollout_steps = new_model_rollout_steps

            epoch_t0 = time.time()
            round_logs = {}
            train_model_s = 0.0
            rollout_model_s = 0.0
            train_agent_s = 0.0
            agent_logs = {}
            self._ep_returns_in_epoch = []

            for env_step in range(self.num_env_steps_per_epoch):
                if self.real_transitions >= self.budget:
                    break
                action = self.agent.select_action(self._obs)['action'][0]
                next_obs, reward, done, truncated, info = self.env.step(action)
                self.real_transitions += 1
                self._traj_length += 1
                self._traj_return += reward
                if truncated or self._traj_length >= self.max_trajectory_length:
                    done = False
                self.env_buffer.add_transition(self._obs, action, next_obs, reward,
                                               done, truncated)
                self._obs = next_obs
                if done or truncated or self._traj_length >= self.max_trajectory_length:
                    obs, info = self.env.reset()
                    self._obs = obs
                    self._ep_returns_in_epoch.append(self._traj_return)
                    self._last_train_return = self._traj_return
                    self._traj_length = 0
                    self._traj_return = 0.0

                if self.real_transitions % self.train_model_interval == 0 \
                        and self.model_env_ratio > 0.0:
                    t0 = time.time()
                    model_logs, train_data, eval_data = self.train_model()
                    self.model_train_epochs_total += int(
                        model_logs.get("misc/model_train_epochs", 0) or 0)
                    aux_logs = self._after_model_training(train_data, eval_data)
                    train_model_s += time.time() - t0
                    t0 = time.time()
                    self.rollout_model()
                    rollout_model_s += time.time() - t0
                    self.model_rounds += 1
                    round_logs = {**model_logs, **aux_logs}
                    print(f"[{self.ALGO}/{self.task}/seed{self.seed}] step "
                          f"{self.real_transitions}: model round {self.model_rounds} "
                          f"holdout_mse={round_logs.get('model_holdout_mse', float('nan')):.3g} "
                          f"epochs={round_logs.get('misc/model_train_epochs', 0)} "
                          f"iters={round_logs.get('misc/model_train_train_steps', 0)}"
                          + (f" aux_kl={round_logs.get('aux/kl_loss', float('nan')):.3g} "
                             f"aux_rob={round_logs.get('aux/robustness_loss', float('nan')):.3g} "
                             f"aux_iters={round_logs.get('aux/iters', 0)}"
                             if aux_logs else "") + f" ({train_model_s:.1f}s+{rollout_model_s:.1f}s)",
                          flush=True)

                t0 = time.time()
                for agent_update_step in range(self.num_agent_updates_per_env_step):
                    agent_logs = self.train_agent()
                    self.sac_updates += 1
                train_agent_s += time.time() - t0

                if self.real_transitions >= self._next_save:
                    self._save_checkpoint()
                    self._next_save += self.save_every

            epoch_s = time.time() - epoch_t0
            self._epoch += 1
            agent_logs_f = materialize(agent_logs) if agent_logs else {}
            rec = {
                "step": int(self.real_transitions),
                "epoch": int(self._epoch),
                "train_return": None if self._last_train_return is None
                else round(float(self._last_train_return), 4),
                "train_return_epoch_mean": (
                    None if not self._ep_returns_in_epoch
                    else round(float(np.mean(self._ep_returns_in_epoch)), 4)),
                "train_episodes": len(self._ep_returns_in_epoch),
                "train_length": int(self._traj_length),
                "model_holdout_mse": round_logs.get("model_holdout_mse"),
                "model_train_epochs_last_round": round_logs.get("misc/model_train_epochs"),
                "model_train_iters_last_round": round_logs.get("misc/model_train_train_steps"),
                "aux/kl_loss": round_logs.get("aux/kl_loss"),
                "aux/robustness_loss": round_logs.get("aux/robustness_loss"),
                "aux/holdout_combined": round_logs.get("aux/holdout_combined"),
                "aux/epochs_last_round": round_logs.get("aux/epochs"),
                "aux/iters_last_round": round_logs.get("aux/iters"),
                "model_buffer_size": int(self.model_buffer.max_sample_size),
                "sac/q1_loss": agent_logs_f.get("loss/q1"),
                "sac/q2_loss": agent_logs_f.get("loss/q2"),
                "sac/policy_loss": agent_logs_f.get("loss/policy"),
                "sac/alpha": agent_logs_f.get("misc/entropy_alpha"),
                "times/train_model_s": round(train_model_s, 3),
                "times/rollout_model_s": round(rollout_model_s, 3),
                "times/train_agent_s": round(train_agent_s, 3),
                "times/epoch_s": round(epoch_s, 3),
                "steps_per_s": round(self.num_env_steps_per_epoch / max(epoch_s, 1e-9), 3),
                "counters": self.counters(),
                "wall_s": round(self._wall_s_total(), 1),
            }
            self._append_progress(rec)
            self._write_counters()

        self._save_final()
        print(f"[{self.ALGO}/{self.task}/seed{self.seed}] DONE: "
              f"{self.real_transitions} real transitions, "
              f"{self.model_transitions} model transitions, "
              f"{self.sac_updates} SAC updates, {self.model_train_iterations} model "
              f"iters, {self.aux_updates} aux updates, "
              f"{self._wall_s_total():.0f}s wall", flush=True)


# --------------------------------------------------------------------------- #
# frozen-evaluation adapter (adapter protocol of eval/evaluator.py)
# --------------------------------------------------------------------------- #
class SacEvalAdapter:
    """Deterministic SAC mean-action adapter (act / reset_episodes protocol)."""

    def __init__(self, agent: SACAgent, device: torch.device):
        self.agent = agent
        self.device = device

    def act(self, obs, active):
        return self.agent.deterministic_action(obs)

    def reset_episodes(self, mask):
        return None


def load_sac_adapter(path: str, cfg: dict | None = None,
                     device=None) -> SacEvalAdapter:
    """Rebuild the SAC agent from a checkpoint (final.pt or a state_*.pt)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    sac_cfg = st.get("sac_cfg") or (cfg or {}).get("sac") or {}
    if device is None:
        device = resolve_device(cfg or {})
    agent = SACAgent(int(st["obs_dim"]), int(st["act_dim"]),
                     st.get("act_low"), st.get("act_high"), sac_cfg, device)
    agent.load_state_dict(st["agent"])
    agent.eval()
    return SacEvalAdapter(agent, device)
