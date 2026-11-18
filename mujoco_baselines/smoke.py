"""Smoke tests (task §5.2): every method walks a short Hopper run plus
reset/step/short-train/checkpoint-load/small-grid-eval on the other tasks.

    python -m mujoco_baselines smoke [--algo rarl_ppo,dr_ppo,...]

Generic checks per (algo, task):
  * train ~cfg["smoke_steps"] transitions; counters.json consistent
  * checkpoint loads; frozen evaluator produces records on a 2-cell grid
  * records: mass/friction of the condition actually entered the model
    (checked in envs tests); returns finite; lengths ≤ 1000
Algo-specific checks live in each algo module's `smoke_checks(job_dir, task,
seed, cfg)` when provided.
"""
from __future__ import annotations

import json
import os
import sys

from . import config as CF
from .constants import MPS_PIPE_DIR, RUNS_DIR
from .eval.evaluator import evaluate_checkpoint
from . import grids as G

SMOKE_STEPS = 20_000
SHORT_STEPS = 2_000


def smoke_one(algo: str, task: str, seed: int, steps: int) -> bool:
    import importlib
    mod = importlib.import_module(f"mujoco_baselines.algos.{algo}")
    job_dir = os.path.join(RUNS_DIR, "smoke", algo, task, f"seed{seed}")
    os.makedirs(job_dir, exist_ok=True)
    from .run.job import _algo_cfg
    cfg = _algo_cfg(algo, task)
    cfg["train_seed"] = seed
    # shrink budgets
    for key in ("steps", "budget", "phase1_steps", "phase2_steps",
                "protagonist_steps"):
        if key in cfg and isinstance(cfg[key], int):
            if key in ("phase1_steps", "phase2_steps"):
                cfg[key] = steps // 2
            elif key == "protagonist_steps":
                cfg[key] = steps // 2
            else:
                cfg[key] = steps
    if "learning_starts" in cfg:
        cfg["learning_starts"] = min(500, cfg["learning_starts"])
    if "warmup" in cfg:
        cfg["warmup"] = min(500, cfg["warmup"])
    if "train_model_interval" in cfg and isinstance(cfg["train_model_interval"], dict):
        cfg["train_model_interval"] = {t: 500 for t in cfg["train_model_interval"]}
    print(f"[smoke] {algo}/{task}/seed{seed}: {steps} steps -> {job_dir}", flush=True)
    mod.train(job_dir, task, seed, cfg)
    counters = json.load(open(os.path.join(job_dir, "counters.json")))
    real = counters.get("real_transitions", 0)
    assert real >= steps - 5, f"counters {real} < {steps}"
    adapter = mod.load_eval_adapter(job_dir, task, seed, cfg)
    conds = [G.grid(task)[0], G.grid(task)[-1], G.grid(task, with_nominal=True)[-1]]
    records, _ = evaluate_checkpoint(adapter, task, conditions=conds, seeds=[20000, 20001])
    assert all(r["length"] <= 1000 for r in records)
    assert all(r["return"] == r["return"] for r in records)  # no NaN
    checks = getattr(mod, "smoke_checks", None)
    extra = checks(job_dir, task, seed, cfg) if checks else {}
    json.dump({"counters": counters, "records": len(records), "extra": extra},
              open(os.path.join(job_dir, "smoke.json"), "w"), indent=1, default=str)
    print(f"[smoke] PASS {algo}/{task}: counters={counters} "
          f"mean_returns={[round(r['return'], 1) for r in records]}", flush=True)
    return True


def main(argv) -> int:
    import argparse
    os.makedirs(MPS_PIPE_DIR, exist_ok=True)
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", MPS_PIPE_DIR)
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="rarl_ppo,dr_ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=SMOKE_STEPS)
    args = p.parse_args(argv)
    ok = True
    for algo in args.algo.split(","):
        # hopper full smoke; other tasks short smoke (task §5.2)
        try:
            smoke_one(algo, "hopper", args.seed, args.steps)
            for task in ("halfcheetah", "walker2d"):
                smoke_one(algo, task, args.seed, SHORT_STEPS)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[smoke] FAIL {algo}: {e}", flush=True)
            ok = False
    print("[smoke]", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1
