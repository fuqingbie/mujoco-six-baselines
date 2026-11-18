"""Phase-A lane: released-weight re-evaluation of MBPO / RMBPO with the
authors' evaluator (rmbpo-tmlr @ d58283f, evaluate/main_eval.py).

Runs, per (agent, task):
    MPLBACKEND=Agg python evaluate/main_eval.py configs/robust_mbpo/<task>-friction-torso.py \
        --load-dir ./weights/<task>-v4/<agent>/ --num-eval-runs 30
inside the rmbpo-tmlr checkout with the mjb env (robustness-gymnasium +
unstable_baselines installed), then converts the authors' pkl output into
per-episode records + phase-A summary values.

Author-protocol notes (kept as-is per task §3, with the scoring caveat):
  * the authors' evaluate_v2 zeroes the reward of the step on which an episode
    ends (rewards *= (1 - dones) before accumulation) — every reported return
    excludes the final step's reward; our unified phase-B evaluator includes
    it. Author outputs are stored unmodified.
  * eval seed default 30 (env i seeded 30+i at creation, first episode starts
    from the second reset of the RNG stream).
  * num-eval-runs 30 reproduces the paper (their note: default 10, use 30).
  * Walker2d author grid = 80 cells; the 64-cell common subset (aligning with
    QARL/HTML) is aggregated additionally.

Usage (driver, run from anywhere):
    python -m mujoco_baselines.phase_a.rmbpo_eval <agent> <task>
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time

from ..constants import EXP_ROOT, RUNS_DIR

CHECKOUT = os.path.join(EXP_ROOT, "upstreams/rmbpo-tmlr")
MJB_PY = os.path.join(EXP_ROOT, "envs/mjb/bin/python")

AGENTS = {"mbpo": "mbpo", "rmbpo": {"hopper": "rmbpo_4", "halfcheetah": "rmbpo_05",
                                    "walker2d": "rmbpo_05"}}

# author grid: (task -> (friction offsets list, mass numerators list))
AUTHOR_GRID = {
    "hopper": ([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
               [250 + 25 * i for i in range(10)]),
    "halfcheetah": ([-0.3 + 0.3 * j for j in range(11)],
                    [300 + 50 * i for i in range(14)]),
    "walker2d": ([-1.0, -0.75, -0.5, -0.25, 0.0, 0.2, 0.25, 0.5, 0.75, 1.0],
                 [100 * (i + 1) for i in range(8)]),
}
# common 64-cell subset for walker2d (8 friction x 8 mass)
WALKER_COMMON_FRICTION = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
NOMINAL_FRICTION = 1.0  # robustness-gymnasium applies geom_friction * (1 + friction_p)


def condition_id(mass_num: int, friction_off_num: int) -> str:
    return f"m{mass_num}_f{round((NOMINAL_FRICTION + friction_off_num / 100) * 100)}"


def run_agent(agent: str, task: str) -> int:
    sub = AGENTS[agent] if agent == "mbpo" else AGENTS[agent][task]
    job_dir = os.path.join(RUNS_DIR, "source", agent, task, "seed0")
    os.makedirs(job_dir, exist_ok=True)
    status = os.path.join(job_dir, "status.json")
    if os.path.exists(status) and json.load(open(status)).get("state") == "DONE":
        print(f"[rmbpo_eval] already done: {job_dir}")
        return 0
    json.dump({"state": "RUNNING", "pid": os.getpid(), "started": time.time()},
              open(status, "w"))
    cfg = f"configs/robust_mbpo/{task}-friction-torso.py"
    load_dir = f"./weights/{task}-v4/{sub}/"
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mjb-no-mps")
    cmd = [MJB_PY, "evaluate/main_eval.py", cfg, "--load-dir", load_dir,
           "--num-eval-runs", "30"]
    t0 = time.time()
    with open(os.path.join(job_dir, "stdout.log"), "a") as out, \
         open(os.path.join(job_dir, "stderr.log"), "a") as err:
        rc = subprocess.call(cmd, cwd=CHECKOUT, env=env, stdout=out, stderr=err)
    el = time.time() - t0
    if rc != 0:
        json.dump({"state": "FAILED", "rc": rc, "error": "author evaluator failed",
                   "finished": time.time()}, open(status, "w"), indent=1)
        print(f"[rmbpo_eval] FAILED {agent}/{task} rc={rc}", file=sys.stderr)
        return rc
    # copy author pkl into the job dir (results_data/<env_name>/<sub>/ — env_name
    # casing follows each config: Hopper-v4 is capitalised, others lowercase;
    # the glob is anchored to THIS task's env_name because concurrent lanes of
    # the same agent share the checkout's results_data tree)
    import glob
    import shutil
    env_name = {"hopper": "Hopper-v4", "halfcheetah": "halfcheetah-v4",
                "walker2d": "walker2d-v4"}[task]
    hits = glob.glob(os.path.join(CHECKOUT, "results_data", env_name, sub,
                                  "friction_p_torso_mass-mean.pkl"))
    dst = os.path.join(job_dir, "author_output.pkl")
    if hits:
        shutil.copy(hits[0], dst)
    records = []
    if not os.path.exists(dst):
        raise FileNotFoundError(f"author pkl not found after eval (globbed {CHECKOUT}/results_data/*/{sub}/)")
    records = parse_author_pkl(dst, agent, task)
    from ..eval.records import save_records
    save_records(records, "source", agent, task, train_seed=0, checkpoint=sub,
                 wall_seconds=el)
    summary = summarize(records, task)
    json.dump(summary, open(os.path.join(job_dir, "summary.json"), "w"), indent=1)
    json.dump({"state": "DONE", "rc": 0, "finished": time.time(), "wall_s": el},
              open(status, "w"), indent=1)
    print(f"[rmbpo_eval] DONE {agent}/{task}: {summary}")
    return 0


def parse_author_pkl(pkl_path: str, agent: str, task: str):
    """Author pkl: (perturb_params, perturb_values, means[5 seeds x cells],
    ci_low, ci_high, mean over seeds) — convert to episode-shaped records.

    The authors' pkl holds per-seed per-cell MEAN returns (30 episodes each),
    not per-episode values; per-episode detail is not recoverable from their
    output, so records here are seed-level means annotated as such (30-episode
    means, eval_seed=None, n_episodes=30 noted in extra).
    """
    with open(pkl_path, "rb") as f:
        perturb_params, perturb_values, eval_mean_returns, lo, hi = pickle.load(f)
    friction_offs, mass_nums = AUTHOR_GRID[task]
    records = []
    # author iteration: friction outer x mass inner (main_eval.py builds
    # [(i, j) for i in frictions for j in masses] then envs in that order)
    cells = [(fo, mn) for fo in friction_offs for mn in mass_nums]
    assert len(cells) == eval_mean_returns.shape[1], \
        f"grid mismatch: {len(cells)} vs {eval_mean_returns.shape[1]}"
    for seed_i in range(eval_mean_returns.shape[0]):
        for c, (fo, mn) in enumerate(cells):
            mass_kg = mn / 100.0
            fmult = NOMINAL_FRICTION + fo
            records.append({
                "task": task,
                "condition_id": c,
                "condition_str": condition_id(mn, int(round(fo * 100))),
                "mass_kg": mass_kg,
                "friction_multiplier": fmult,
                "is_ood": bool(mass_kg < 0.9 * _nominal(task) - 1e-9
                               or mass_kg > 1.1 * _nominal(task) + 1e-9
                               or fmult < 0.9 - 1e-9 or fmult > 1.1 + 1e-9),
                "is_nominal": False,
                "eval_seed": seed_i,  # author seed_a..e index
                "return": float(eval_mean_returns[seed_i, c]),
                "length": -1,
                "terminated": False, "truncated": False,
                "extra": json.dumps({"author_cell": f"friction_p={fo},torso_mass={mass_kg}",
                                     "level": "seed_mean_of_30_episodes",
                                     "scoring": "author_evaluate_v2_excludes_final_step"}),
            })
    return records


def _nominal(task: str) -> float:
    from ..grids import nominal_torso_mass
    return nominal_torso_mass(task)


def summarize(records, task: str) -> dict:
    import numpy as np
    rets = np.array([r["return"] for r in records])
    n_cells = len(records) // 5
    by_seed = rets.reshape(5, n_cells)
    out = {"author_grid_mean": float(by_seed.mean()),
           "author_grid_mean_per_seed": [float(x) for x in by_seed.mean(axis=1)],
           "n_cells_author": n_cells}
    if task == "walker2d":
        # common 64-cell subset
        friction_offs, mass_nums = AUTHOR_GRID[task]
        cells = [(fo, mn) for fo in friction_offs for mn in mass_nums]
        keep = [i for i, (fo, mn) in enumerate(cells) if fo in WALKER_COMMON_FRICTION]
        out["common64_mean"] = float(by_seed[:, keep].mean())
        out["n_cells_common"] = len(keep)
    # nominal-condition cell (friction 0, nominal mass)
    nom = _nominal(task)
    nom_cells = [i for i, r in enumerate(records)
                 if abs(r["friction_multiplier"] - 1.0) < 1e-9
                 and abs(r["mass_kg"] - nom) < 0.01]
    if nom_cells:
        out["nominal_mean"] = float(np.mean([records[i]["return"] for i in nom_cells]))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m mujoco_baselines.phase_a.rmbpo_eval <mbpo|rmbpo> <task>")
        sys.exit(2)
    sys.exit(run_agent(sys.argv[1], sys.argv[2]))
