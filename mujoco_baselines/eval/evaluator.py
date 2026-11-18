"""Unified frozen evaluator (task §4.1 / §4.2 / §6).

Protocol, identical for every method and both phases of scoring:
  * 30 reset seeds [20000..20029] per condition per checkpoint; each env runs
    exactly one episode per condition (30 episodes per cell).
  * conditions = full unified grid (90/154/64) + one nominal condition
    (nominal torso mass read from the raw env, friction multiplier 1.0).
  * deterministic policies (SAC mean action / PPO deterministic prediction);
    any per-episode state a method keeps (RMA history) is reset at episode start.
  * no adversary, no action noise, no exploration; weights and normalization
    statistics frozen.
  * raw return: sum of environment rewards up to and INCLUDING the first
    terminated or truncated step (the final step's reward is counted — this is
    the unified scorer; the RMBPO authors' evaluate_v2 zeroes the final step's
    reward and is kept only for the phase-A released-weight re-evaluation).
  * episode cap 1000 via the v4 TimeLimit: `truncated` is True when the cap
    ends a not-terminated episode; `terminated` reflects the env's own flag.
  * only the base observation is exposed to the policy (history/temperature
    inputs are maintained by the method adapter itself).

Policy adapter protocol:
    adapter.act(obs: np.ndarray[n, obs_dim], active: np.ndarray[n] of bool)
        -> np.ndarray[n, act_dim]
        (deterministic action; per-env state such as RMA history must only be
        advanced for envs where active is True — the vector env auto-resets
        finished envs and their obs rows are stale)
    adapter.reset_episodes(mask: np.ndarray[n] of bool)  # called after resets
"""
from __future__ import annotations

import time

import numpy as np

from ..constants import EVAL_RESET_SEEDS, EPISODE_CAP
from ..envs.perturbed import make_eval_venv
from .. import grids as G


def evaluate_condition(adapter, task: str, condition: dict, seeds=EVAL_RESET_SEEDS):
    """Run one condition: 30 envs x 1 episode. Returns list of episode dicts."""
    n = len(seeds)
    venv = make_eval_venv(task, (condition["mass_kg"], condition["friction_multiplier"]),
                          list(seeds))
    obs, _ = venv.reset()
    adapter.reset_episodes(np.ones(n, dtype=bool))
    returns = np.zeros(n, dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    steps = 0
    while active.any() and steps < EPISODE_CAP:
        actions = adapter.act(obs, active)
        obs, rewards, term, trunc, _ = venv.step(actions)
        # accumulate the final step's reward for episodes that end here
        returns[active] += np.asarray(rewards)[active]
        lengths[active] += 1
        terminated |= term
        truncated |= trunc
        active &= ~(term | trunc)
        steps += 1
    venv.close()
    out = []
    for i, s in enumerate(seeds):
        out.append({
            "task": task,
            "condition_id": condition["condition_id"],
            "condition_str": condition["condition_str"],
            "mass_kg": float(condition["mass_kg"]),
            "friction_multiplier": float(condition["friction_multiplier"]),
            "is_ood": bool(condition["is_ood"]),
            "is_nominal": bool(condition.get("is_nominal", False)),
            "eval_seed": int(s),
            "return": float(returns[i]),
            "length": int(lengths[i]),
            "terminated": bool(terminated[i]),
            "truncated": bool(truncated[i]),
        })
    return out


def evaluate_checkpoint(adapter, task: str, conditions=None, seeds=EVAL_RESET_SEEDS,
                        log_every=20):
    """Evaluate all conditions for one checkpoint. Returns (records, wall_seconds)."""
    if conditions is None:
        conditions = G.grid(task, with_nominal=True)
    t0 = time.time()
    records = []
    for k, cond in enumerate(conditions):
        records.extend(evaluate_condition(adapter, task, cond, seeds))
        if (k + 1) % log_every == 0:
            el = time.time() - t0
            print(f"[eval:{task}] {k + 1}/{len(conditions)} conditions, "
                  f"{el:.0f}s elapsed", flush=True)
    return records, time.time() - t0
