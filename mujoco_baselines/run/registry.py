"""Job registry for both protocols.

A Job is one schedulable unit: a subprocess command run in a fixed cwd with a
fixed environment (python interpreter, GPU, MPS bypass). Phase-B jobs run the
unified package in the `mjb` env; phase-A jobs run per-lane drivers that may
use `mjb5` (v5 stack) or the authors' checkouts.
"""
from __future__ import annotations

import os

from ..constants import ALGOS, EXP_ROOT, RUNS_DIR, TASKS, TRAIN_SEEDS

MJB_PY = os.path.join(EXP_ROOT, "envs/mjb/bin/python")
MJB5_PY = os.path.join(EXP_ROOT, "envs/mjb5/bin/python")

# longest-first so the heavy families start early
ALGO_ORDER = ("mbpo", "rmbpo", "rma", "qarl", "rarl_ppo", "dr_ppo")
TASK_ORDER = {"hopper": 0, "halfcheetah": 1, "walker2d": 2}


class Job:
    def __init__(self, algo, task, seed, protocol, cmd, cwd, py, gpu_any=True):
        self.algo, self.task, self.seed = algo, task, seed
        self.protocol = protocol
        self.cmd = cmd
        self.cwd = cwd
        self.py = py
        self.dir = os.path.join(RUNS_DIR, protocol, algo, task, f"seed{seed}")
        self.gpu_any = gpu_any
        self.id = f"{protocol}:{algo}:{task}:{seed}"

    def sort_key(self):
        first = 0 if (self.task == "hopper" and self.seed == 0) else 1
        return (first, ALGO_ORDER.index(self.algo) if self.algo in ALGO_ORDER else 99,
                TASK_ORDER.get(self.task, 9), self.seed)


def build(protocol: str, phases: tuple = ("b",)) -> list:
    jobs = []
    if protocol == "shared" and "b" in phases:
        for algo in ALGO_ORDER:
            for task in TASKS:
                for seed in TRAIN_SEEDS:
                    cmd = [MJB_PY, "-m", "mujoco_baselines.run.job", "shared",
                           algo, task, str(seed)]
                    jobs.append(Job(algo, task, seed, "shared", cmd, EXP_ROOT, MJB_PY))
    if protocol == "source" or "a" in phases:
        from . import phase_a
        jobs += phase_a.build_jobs()
    jobs.sort(key=lambda j: j.sort_key())
    return jobs
