"""Phase-A RMA grid evaluator (runs in the SPARC env, task §3).

Evaluates a trained adapter policy on the 9x9 wind grid with the authors'
evaluation semantics (sampled actions via get_action, as in
utils/log_utils.eval_agent — their Table 5 protocol), 30 episodes per
condition with reset seeds [20000..20029], Hopper termination disabled
(inside the Wind envs, per make_env).

Grids (task §3):
  hopper/walker2d: x in [-20,20] step 5, z in [-5,5] step 1.25
  halfcheetah:     x in [-5,5] step 1.25, z in [-10,10] step 2.5
OOD = strictly outside the training rectangle:
  hopper/walker2d: x in [-10,10], z in [-2.5,2.5]
  halfcheetah:     x in [-2.5,2.5], z in [-5,5]

Usage (sparc env, PYTHONPATH=<sparc checkout>):
    python -m mujoco_baselines.phase_a.rma_eval_grid <task> <adapter.pth> \
        --out <records.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.environ.get("SPARC_CHECKOUT", ""))

from utils.modules import AdapterPolicy  # noqa: E402
from environments import wind_hopper, wind_halfcheetah, wind_walker2d  # noqa: E402,F401

GRIDS = {
    "hopper": {"xs": [(-20 + 5 * i) for i in range(9)], "zs": [(-5 + 1.25 * i) for i in range(9)],
               "train_rect": (-10, 10, -2.5, 2.5), "env_id": "WindHopper-v5"},
    "walker2d": {"xs": [(-20 + 5 * i) for i in range(9)], "zs": [(-5 + 1.25 * i) for i in range(9)],
                 "train_rect": (-10, 10, -2.5, 2.5), "env_id": "WindWalker2d-v5"},
    "halfcheetah": {"xs": [(-5 + 1.25 * i) for i in range(9)], "zs": [(-10 + 2.5 * i) for i in range(9)],
                    "train_rect": (-2.5, 2.5, -5, 5), "env_id": "WindHalfCheetah-v5"},
}
SEEDS = list(range(20000, 20030))


def make_env_thunk(task, wx, wz, history_length=50):
    env_id = GRIDS[task]["env_id"]

    def thunk():
        kwargs = dict(wind_speed_interval_x=(wx, wx), wind_speed_interval_z=(wz, wz),
                      history_length=history_length,
                      context_in_obs=True, history_in_obs=True)
        if "WindHopper" in env_id:
            kwargs["terminate_when_unhealthy"] = False
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def eval_condition(adapter, task, wx, wz, device):
    n = len(SEEDS)
    envs = gym.vector.SyncVectorEnv([make_env_thunk(task, wx, wz) for _ in range(n)])
    obs, _ = envs.reset(seed=SEEDS)
    returns = np.zeros(n)
    lengths = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    while active.any():
        o = obs["obs"]
        h = obs["history"]
        with torch.no_grad():
            actions, _, _ = adapter.get_action(torch.Tensor(o).to(device),
                                               torch.Tensor(h).to(device))
        obs, rewards, term, trunc, infos = envs.step(actions.cpu().numpy())
        returns[active] += np.asarray(rewards)[active]
        lengths[active] += 1
        active &= ~(term | trunc)
    envs.close()
    return returns, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("adapter_path")
    p.add_argument("--out", required=True)
    p.add_argument("--history-length", type=int, default=50)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = GRIDS[args.task]
    envs = gym.vector.SyncVectorEnv([make_env_thunk(args.task, 0.0, 0.0,
                                                    args.history_length) for _ in range(1)])
    obs_shape = envs.single_observation_space["obs"].shape
    action_shape = envs.single_action_space.shape
    adapter = AdapterPolicy(envs, args.history_length, obs_shape, action_shape).to(device)
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=device))
    adapter.eval()
    envs.close()

    train_x = g["train_rect"]
    records = []
    for wx in g["xs"]:
        for wz in g["zs"]:
            ood = not (train_x[0] <= wx <= train_x[1] and train_x[2] <= wz <= train_x[3])
            rets, lens = eval_condition(adapter, args.task, wx, wz, device)
            for i, s in enumerate(SEEDS):
                records.append({
                    "task": args.task, "condition_id": len(records),
                    "condition_str": f"wx{wx}_wz{wz}",
                    "mass_kg": float("nan"), "friction_multiplier": float("nan"),
                    "is_ood": bool(ood), "is_nominal": (wx == 0 and wz == 0),
                    "eval_seed": int(s), "return": float(rets[i]),
                    "length": int(lens[i]), "terminated": False, "truncated": False,
                    "extra": json.dumps({"wind_x": wx, "wind_z": wz,
                                         "scoring": "sparc eval_agent semantics (sampled actions)"}),
                })
            print(f"[rma-eval] {args.task} wx={wx} wz={wz} mean={rets.mean():.1f}",
                  flush=True)
    json.dump(records, open(args.out, "w"))
    print(f"[rma-eval] wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
