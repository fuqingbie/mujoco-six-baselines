"""MuJoCo six-baseline reproduction & unified evaluation package.

Layout:
  grids.py      unified test grids (task §4.3) + nominal masses
  envs/         perturbed env construction (robustness-gymnasium fork, noise-free step)
                shared per-episode physical parameter sampler
                body-force adversary wrapper (RARL-PPO / QARL)
  algos/        six method implementations for phase B (shared v4 stack)
  eval/         unified frozen evaluator + per-episode record IO
  runs/         job runner, status files, runs.jsonl manifest
  report.py     aggregation into the two 6x3 tables
"""
import os

EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAMS = os.path.join(EXP_ROOT, "upstreams")
RUNS_DIR = os.path.join(EXP_ROOT, "runs")
LOGS_DIR = os.path.join(EXP_ROOT, "logs")
REPORTS_DIR = os.path.join(EXP_ROOT, "reports")
CONFIGS_DIR = os.path.join(EXP_ROOT, "configs")
EPISODES_DIR = os.path.join(EXP_ROOT, "episodes")
MANIFEST = os.path.join(EXP_ROOT, "runs.jsonl")
LOCK = os.path.join(EXP_ROOT, "upstreams.lock.json")

TASKS = ("hopper", "halfcheetah", "walker2d")
ALGOS = ("rarl_ppo", "dr_ppo", "rma", "qarl", "mbpo", "rmbpo")

# phase-B unified budget (task §4.2)
BUDGET_SHARED = 3_000_000
TRAIN_SEEDS = (0, 1, 2, 3, 4)
EVAL_RESET_SEEDS = tuple(range(20000, 20030))  # 30 episodes per condition

# phase-B shared training distribution (task §4.1)
MASS_MULT_RANGE = (0.90, 1.10)
FRICTION_MULT_RANGE = (0.90, 1.10)

# episode cap (task §4.1)
EPISODE_CAP = 1000

# MPS bypass for every CUDA process we spawn (kev stack owns the default pipe)
MPS_PIPE_DIR = "/tmp/mjb-no-mps"

# adversary force cap for phase B (task §4.1: ±1 N on world x,z)
ADVERSARY_FORCE = 1.0

# body-force adversary mappings, taken from RobustRLlib rarl_ppo_*.yaml
ADVERSARY_BODIES = {
    "hopper": ("foot",),
    "halfcheetah": ("torso", "bfoot", "ffoot"),
    "walker2d": ("foot", "foot_left"),
}
