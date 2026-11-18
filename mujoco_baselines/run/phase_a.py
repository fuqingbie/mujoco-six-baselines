"""Phase-A (source protocol) job list: one heterogeneous driver per lane."""
from __future__ import annotations

import os

from ..constants import EXP_ROOT, TASKS, TRAIN_SEEDS
from .registry import Job, MJB_PY, MJB5_PY

SPARC_PY = os.path.join(EXP_ROOT, "envs/sparc/bin/python")

LANES = {
    # algo: (python, module, seeds) — GPU lane (mjb env)
    "qarl": (MJB_PY, "mujoco_baselines.phase_a.qarl"),
    "dr_ppo": (MJB_PY, "mujoco_baselines.phase_a.dr_v3"),
    # GPU lane (mjb5 env, robustrllib native)
    "rarl_ppo": (MJB5_PY, "mujoco_baselines.phase_a.rarl"),
    # GPU lane (sparc env)
    "rma": (SPARC_PY, "mujoco_baselines.phase_a.rma"),
}

EVAL_ONLY = [
    ("mbpo", "hopper"), ("mbpo", "halfcheetah"), ("mbpo", "walker2d"),
    ("rmbpo", "hopper"), ("rmbpo", "halfcheetah"), ("rmbpo", "walker2d"),
]


def build_jobs():
    jobs = []
    for algo, (py, module) in LANES.items():
        for task in TASKS:
            for seed in TRAIN_SEEDS:
                jobs.append(Job(
                    algo, task, seed, "source",
                    cmd=[py, "-m", module, task, str(seed)],
                    cwd=EXP_ROOT, py=py))
    for algo, task in EVAL_ONLY:
        jobs.append(Job(
            algo, task, 0, "source",
            cmd=[MJB_PY, "-m", "mujoco_baselines.phase_a.rmbpo_eval", algo, task],
            cwd=EXP_ROOT, py=MJB_PY))
    return jobs
