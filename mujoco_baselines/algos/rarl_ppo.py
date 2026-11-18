"""RARL-PPO for phase B — vendored from robustrllib @ a665ddf
(baselines/rarl/{game,train}.py, configs/algorithm/rarl_ppo.yaml), adapted to:

  * unified v4 PerturbEnv (robustness-gymnasium semantics, no action noise)
    instead of robustrllib.tasks.make_env v5 envs
  * adversary force fixed to ±1 N (world x, z; task §4.1) on the same
    per-task body mapping as RobustRLlib's rarl_ppo_*.yaml
  * shared 3,000,000-transition counter: the protagonist and the adversary
    each run 1:1 alternating blocks in their own (separate) environment
    stacks, every transition in either stack increments the shared counter
    (task §4.2); the protagonist's last PPO block is trimmed to an updatable
    length so the counter lands exactly on the budget.

Everything else (PPO hyperparameters, update blocks, player views, obs/reward
normalisation of the protagonist, monitor logging) follows upstream.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..constants import ADVERSARY_BODIES, BUDGET_SHARED, EPISODE_CAP
from ..envs.perturbed import make_train_env
from ..envs.force import BodyForce

AXES = {"x": 0, "y": 1, "z": 2}


class Player:
    """Current policy of one player, queried from inside the other's view."""

    def __init__(self):
        self.model = None
        self.vecnorm = None
        self.active = True

    def act(self, obs, space):
        if not self.active:
            return np.zeros(space.shape, dtype=np.float32)
        if self.model is None:
            return np.asarray(space.sample(), dtype=np.float32)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = self.model.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.float32).reshape(space.shape)


class PlayerView(gym.Wrapper):
    """One player's view of the two-player game (upstream semantics).

    role="protagonist": task action space and reward (adversary policy fills
    the perturbation). role="adversary": perturbation action space and the
    negated reward (protagonist policy fills the task action).
    """

    def __init__(self, env, bodies, axes, force, role, other: Player,
                 zero_force_prob=0.0, seed=None):
        super().__init__(env)
        self.adversary = BodyForce(env, bodies, axes, force)
        self.role = role
        self.other = other
        self.zero_force_prob = float(zero_force_prob)
        self._rng = np.random.default_rng(seed)
        self._obs = None
        if role == "adversary":
            self.action_space = self.adversary.space

    def reset(self, **kwargs):
        self.adversary.reset()
        self._obs, info = self.env.reset(**kwargs)
        return self._obs, info

    def step(self, action):
        space = self.adversary.space
        if self.role == "protagonist":
            pro = action
            if self.zero_force_prob > 0.0 and self._rng.random() < self.zero_force_prob:
                adv = np.zeros(space.shape, dtype=np.float32)
            else:
                adv = self.other.act(self._obs, space)
        else:
            pro = self.other.act(self._obs, self.env.action_space)
            adv = action
        self.adversary.apply(adv)
        self._obs, reward, terminated, truncated, info = self.env.step(pro)
        self.adversary.clear()
        if self.role == "adversary":
            reward = -float(reward)
        return self._obs, reward, terminated, truncated, info


def _make_agent(hp, venv, seed, lr):
    kwargs = {k: hp[k] for k in ("n_steps", "batch_size", "n_epochs", "learning_rate",
                                 "gamma", "gae_lambda", "clip_range", "ent_coef",
                                 "vf_coef", "max_grad_norm")}
    kwargs["learning_rate"] = lr
    policy = "CnnPolicy" if len(venv.observation_space.shape) == 3 else "MlpPolicy"
    return PPO(policy, venv, seed=seed, verbose=0, device="cpu", **kwargs)


def _learn_exact(agent, n_envs, target, batch_size):
    """Drive the agent to exactly `target` num_timesteps with updatable blocks.

    learn(total_timesteps=T, reset_num_timesteps=False) is incremental in SB3;
    when the remaining T is below one block, n_steps is trimmed to an
    exactly-updatable length and the rollout buffer reallocated to match (SB3
    allocates it only at construction, so a trimmed block would otherwise
    never mark itself full)."""
    from stable_baselines3.common.buffers import RolloutBuffer
    while agent.num_timesteps < target:
        block = agent.n_steps * n_envs
        remaining = target - agent.num_timesteps
        if remaining < block:
            rem_env = max(remaining // n_envs, batch_size)
            agent.n_steps = int(rem_env)
            agent.rollout_buffer = RolloutBuffer(
                agent.n_steps, agent.observation_space, agent.action_space,
                device=agent.device, gamma=agent.gamma, gae_lambda=agent.gae_lambda,
                n_envs=n_envs)
        agent.learn(total_timesteps=(agent.n_steps * n_envs), reset_num_timesteps=False)


def train(job_dir: str, task: str, seed: int, cfg: dict) -> None:
    ck = os.path.join(job_dir, "ckpt")
    counters_path = os.path.join(job_dir, "counters.json")
    if os.path.exists(os.path.join(ck, "step_final.zip")) and os.path.exists(counters_path):
        c = json.load(open(counters_path))
        if c.get("real_transitions") >= cfg.get("budget", BUDGET_SHARED):
            return
    os.makedirs(ck, exist_ok=True)
    hp = cfg["ppo"]
    n_envs = int(cfg["n_envs"])
    ab = cfg["adversary_bodies"]
    bodies = ab[task] if isinstance(ab, dict) else (ab or ADVERSARY_BODIES[task])
    axes = cfg.get("adversary_axes", ("x", "z"))
    force = float(cfg["adversary_force"])
    pro_budget = int(cfg.get("protagonist_steps", BUDGET_SHARED // 2))
    set_random_seed(seed)
    torch.set_num_threads(2)

    protagonist, adversary = Player(), Player()

    def view(role, other, env_seed):
        def build():
            env = make_train_env(task, env_seed)
            v = PlayerView(env, bodies, axes, force, role, other,
                           zero_force_prob=cfg["zero_force_prob"] if role == "protagonist" else 0.0,
                           seed=env_seed)
            return Monitor(v) if role == "protagonist" else v
        return build

    pro_venv = DummyVecEnv([view("protagonist", adversary, seed + i) for i in range(n_envs)])
    adv_venv = DummyVecEnv([view("adversary", protagonist, seed + n_envs + i) for i in range(n_envs)])
    if cfg["normalize"]:
        pro_venv = VecNormalize(pro_venv, norm_obs=True, norm_reward=True,
                                clip_obs=10.0, gamma=hp["gamma"])
        protagonist.vecnorm = pro_venv
    lr = float(hp["learning_rate"])
    protagonist.model = _make_agent(hp, pro_venv, seed, lr)
    adversary.model = _make_agent(hp, adv_venv, seed + n_envs, lr)
    pro, adv = protagonist.model, adversary.model

    t0 = time.time()
    prog = open(os.path.join(job_dir, "progress.jsonl"), "a")

    while pro.num_timesteps < pro_budget:
        _learn_exact(pro, n_envs, pro_budget, hp["batch_size"])
        ret = np.mean([e["r"] for e in pro.ep_info_buffer]) if pro.ep_info_buffer else float("nan")
        prog.write(json.dumps({"protagonist_steps": pro.num_timesteps,
                               "adversary_steps": adv.num_timesteps,
                               "episode_return": float(ret),
                               "wall_s": time.time() - t0}) + "\n")
        prog.flush()
        # 1:1 adversary share, interleaved
        adversary.active = True
        _learn_exact(adv, n_envs, pro_budget, hp["batch_size"])
    # final checkpoint at exact budget
    os.makedirs(ck, exist_ok=True)
    saved = os.path.join(ck, "step_final.zip")
    pro.save(saved)
    if protagonist.vecnorm is not None:
        protagonist.vecnorm.save(os.path.join(ck, "vecnorm.pkl"))
    prog.close()

    total_transitions = int(pro.num_timesteps + adv.num_timesteps)
    json.dump({"real_transitions": total_transitions,
               "protagonist_steps": int(pro.num_timesteps),
               "adversary_steps": int(adv.num_timesteps)},
              open(counters_path, "w"), indent=1)
    pro_venv.close()
    adv_venv.close()
    print(f"[rarl_ppo] done: pro {pro.num_timesteps} + adv {adv.num_timesteps} "
          f"= {total_transitions} transitions, ckpt {saved}", flush=True)


class _RarlEvalAdapter:
    def __init__(self, model, vecnorm):
        self.model = model
        self.vecnorm = vecnorm

    def reset_episodes(self, mask):
        pass

    def act(self, obs, active):
        obs = np.asarray(obs, dtype=np.float32)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def load_eval_adapter(job_dir: str, task: str, seed: int, cfg: dict):
    from stable_baselines3 import PPO as _PPO
    ck = os.path.join(job_dir, "ckpt")
    model = _PPO.load(os.path.join(ck, "step_final.zip"), device=cfg.get("device", "cpu"))
    vecnorm = None
    vn = os.path.join(ck, "vecnorm.pkl")
    if cfg.get("normalize", True) and os.path.exists(vn):
        vecnorm = VecNormalize.load(vn, DummyVecEnv([lambda: make_train_env(task, seed)]))
        vecnorm.training = False
        vecnorm.norm_reward = False
    return _RarlEvalAdapter(model, vecnorm)
