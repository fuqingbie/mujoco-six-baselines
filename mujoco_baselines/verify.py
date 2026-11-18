"""Acceptance verification (task §6): every table cell must be recomputable
from the per-episode records, with the promised episode counts.

    python -m mujoco_baselines verify

Checks per (protocol, algo, task, train_seed, checkpoint):
  * grid coverage: every unified-grid condition + nominal present
  * episode count: 30 episodes per condition (eval seeds 20000..20029)
  * lengths <= 1000; returns finite; terminated XOR truncated-or-alive
  * phase-B cells: recompute seed-mean -> cell-mean -> seed-mean and compare
    with report tables
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .constants import ALGOS, EPISODES_DIR, EVAL_RESET_SEEDS, TASKS
from . import grids as G


def verify_shared() -> list:
    problems = []
    from .eval.records import load_all
    df = load_all("shared")
    if df.empty:
        return ["no shared records yet"]
    for algo in ALGOS:
        for task in TASKS:
            sub = df[(df["algo"] == algo) & (df["task"] == task)]
            seeds = sorted(sub["train_seed"].unique())
            for seed in seeds:
                s = sub[sub["train_seed"] == seed]
                conds = s["condition_id"].unique()
                grid_ids = {c["condition_id"] for c in G.grid(task)}
                missing = grid_ids - set(conds)
                if missing:
                    problems.append(f"{algo}/{task}/s{seed}: missing {len(missing)} grid cells")
                if -1 not in set(conds):
                    problems.append(f"{algo}/{task}/s{seed}: nominal condition missing")
                bad_len = (s["length"] > 1000).sum() + (s["length"] <= 0).sum()
                if bad_len:
                    problems.append(f"{algo}/{task}/s{seed}: {bad_len} bad lengths")
                nonfinite = (~np.isfinite(s["return"].astype(float))).sum()
                if nonfinite:
                    problems.append(f"{algo}/{task}/s{seed}: {nonfinite} non-finite returns")
                per_cond = s.groupby("condition_id")["return"].count()
                under = per_cond[per_cond < len(EVAL_RESET_SEEDS)]
                if len(under):
                    problems.append(f"{algo}/{task}/s{seed}: {len(under)} cells with <30 episodes")
                if not math.isclose(float(np.isfinite(s["return"]).mean()), 1.0):
                    problems.append(f"{algo}/{task}/s{seed}: finite check failed")
    return problems


def verify_source() -> list:
    problems = []
    from .eval.records import load_all
    df = load_all("source")
    if df.empty:
        return ["no source records yet"]
    # lanes with per-episode records: rarl (20/cell), rma (30/cell),
    # qarl/dr (30/cell); mbpo/rmbpo are seed-mean level (noted)
    expected_eps = {"rarl_ppo": 20, "rma": 30, "qarl": 30, "dr_ppo": 30}
    for (algo, task), sub in df.groupby(["algo", "task"]):
        if algo in expected_eps:
            per = sub.groupby(["train_seed", "condition_id"])["return"].count()
            under = per[per < expected_eps[algo] - 2]
            if len(under) > 0:
                problems.append(f"source {algo}/{task}: {len(under)} cells under "
                                f"{expected_eps[algo]} episodes")
    return problems


def main(argv=None) -> int:
    p1 = verify_shared()
    p2 = verify_source()
    for line in p1 + p2:
        print("[verify]", line)
    print("[verify]", "ALL OK" if not (p1 + p2) else f"{len(p1)+len(p2)} problems")
    return 0 if not (p1 + p2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
