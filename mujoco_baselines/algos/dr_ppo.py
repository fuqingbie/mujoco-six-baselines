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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..constants import BUDGET_SHARED
from ..envs.perturbed import make_train_env



def _learn_exact(agent, n_envs, target, batch_size):
    """Exact-budget PPO blocks; the final block is trimmed to an updatable
    length with the rollout buffer reallocated to match (see rarl_ppo)."""
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
    budget = int(cfg.get("steps", BUDGET_SHARED))
    if os.path.exists(os.path.join(ck, "final.zip")) and os.path.exists(counters_path):
        c = json.load(open(counters_path))
        if c.get("real_transitions") >= budget:
            return
    os.makedirs(ck, exist_ok=True)
    hp = cfg["ppo"]
    n_envs = int(cfg["n_envs"])
    set_random_seed(seed)
    torch.set_num_threads(2)

    venv = DummyVecEnv([lambda i=i: Monitor(make_train_env(task, seed + i))
                        for i in range(n_envs)])
    if cfg["normalize"]:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=hp["gamma"])
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **{
        k: hp[k] for k in ("n_steps", "batch_size", "n_epochs", "learning_rate",
                           "gamma", "gae_lambda", "clip_range", "ent_coef",
                           "vf_coef", "max_grad_norm")})
    t0 = time.time()
    prog = open(os.path.join(job_dir, "progress.jsonl"), "a")
    while model.num_timesteps < budget:
        _learn_exact(model, n_envs, budget, hp["batch_size"])
        ret = np.mean([e["r"] for e in model.ep_info_buffer]) if model.ep_info_buffer else float("nan")
        prog.write(json.dumps({"steps": model.num_timesteps, "episode_return": float(ret),
                               "wall_s": time.time() - t0}) + "\n")
        prog.flush()
    os.makedirs(ck, exist_ok=True)
    final = os.path.join(ck, "final.zip")
    model.save(final)
    if cfg["normalize"]:
        venv.save(os.path.join(ck, "vecnorm.pkl"))
    prog.close()
    json.dump({"real_transitions": int(model.num_timesteps)}, open(counters_path, "w"), indent=1)
    venv.close()
    print(f"[dr_ppo] done: {model.num_timesteps} transitions, ckpt {final}", flush=True)


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
    model = PPO.load(os.path.join(ck, "final.zip"), device=cfg.get("device", "cpu"))
    vecnorm = None
    vn = os.path.join(ck, "vecnorm.pkl")
    if cfg.get("normalize", True) and os.path.exists(vn):
        vecnorm = VecNormalize.load(vn, DummyVecEnv([lambda: make_train_env(task, seed)]))
        vecnorm.training = False
        vecnorm.norm_reward = False
    return _DrEvalAdapter(model, vecnorm)
