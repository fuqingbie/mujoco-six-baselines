"""DR-PPO for phase B — SB3 PPO on the shared per-episode physical sampler.

Domain randomization = the shared PhysSampler (m/m0 ~ U[0.90,1.10], friction
mult ~ U[0.90,1.10]) resampled at every reset, holding for the whole episode
(task §4.1). The PPO stack (hyperparameters, VecNormalize obs+reward,
n_envs=4) is identical to the RARL-PPO protagonist so the DR-vs-adversary
comparison isolates the robustness mechanism. [REBUILD choice: AD-RRL's
public tree trains AD-RRL, not DR-PPO; the paper describes SB3 PPO + DR
intervals. Phase B uses the unified narrow distribution instead of the
paper's wide intervals (phase A reconstructs those separately).]
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..constants import BUDGET_SHARED
from ..envs.perturbed import make_train_env


class _BudgetStop(BaseCallback):
    def __init__(self, steps, ckpt_dir, vecnorm=None):
        super().__init__()
        self.steps, self.ckpt_dir, self.vecnorm = int(steps), ckpt_dir, vecnorm
        self.saved = None

    def _on_step(self) -> bool:
        if self.num_timesteps < self.steps:
            return True
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.saved = os.path.join(self.ckpt_dir, "final.zip")
        self.model.save(self.saved)
        if self.vecnorm is not None:
            self.vecnorm.save(os.path.join(self.ckpt_dir, "vecnorm.pkl"))
        return False


def train(job_dir: str, task: str, seed: int, cfg: dict) -> None:
    ck = os.path.join(job_dir, "ckpt")
    counters_path = os.path.join(job_dir, "counters.json")
    budget = int(cfg.get("steps", BUDGET_SHARED))
    if os.path.exists(os.path.join(ck, "final.zip")) and os.path.exists(counters_path):
        c = json.load(open(counters_path))
        if c.get("real_transitions") >= budget:
            return
    os.makedirs(ck, exist_ok=True)
    hp = cfg["ppo"]
    n_envs = int(cfg["n_envs"])
    set_random_seed(seed)
    torch.set_num_threads(8)

    venv = DummyVecEnv([lambda i=i: Monitor(make_train_env(task, seed + i))
                        for i in range(n_envs)])
    if cfg["normalize"]:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=hp["gamma"])
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, **{
        k: hp[k] for k in ("n_steps", "batch_size", "n_epochs", "learning_rate",
                           "gamma", "gae_lambda", "clip_range", "ent_coef",
                           "vf_coef", "max_grad_norm")})
    stop = _BudgetStop(budget, ck, venv if cfg["normalize"] else None)
    block = int(hp["n_steps"]) * n_envs
    t0 = time.time()
    prog = open(os.path.join(job_dir, "progress.jsonl"), "a")
    while stop.saved is None:
        remaining = budget - model.num_timesteps
        total = block if remaining >= block else remaining
        if total < block:
            rem_env = max(total // n_envs, hp["batch_size"])
            model.n_steps = int(rem_env)
            total = rem_env * n_envs
        model.learn(total_timesteps=total, callback=stop, reset_num_timesteps=False)
        ret = np.mean([e["r"] for e in model.ep_info_buffer]) if model.ep_info_buffer else float("nan")
        prog.write(json.dumps({"steps": model.num_timesteps, "episode_return": float(ret),
                               "wall_s": time.time() - t0}) + "\n")
        prog.flush()
    prog.close()
    json.dump({"real_transitions": int(model.num_timesteps)}, open(counters_path, "w"), indent=1)
    venv.close()
    print(f"[dr_ppo] done: {model.num_timesteps} transitions, ckpt {stop.saved}", flush=True)


class _DrEvalAdapter:
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
    ck = os.path.join(job_dir, "ckpt")
    model = PPO.load(os.path.join(ck, "final.zip"), device=cfg.get("device", "cuda"))
    vecnorm = None
    vn = os.path.join(ck, "vecnorm.pkl")
    if cfg.get("normalize", True) and os.path.exists(vn):
        vecnorm = VecNormalize.load(vn, DummyVecEnv([lambda: make_train_env(task, seed)]))
        vecnorm.training = False
        vecnorm.norm_reward = False
    return _DrEvalAdapter(model, vecnorm)
