"""Frozen experiment configuration (single source of truth).

Phase B ("shared"): every method trains from scratch on the unified
robustness-gymnasium v4 stack with the shared PhysSampler
(m/m0 ~ U[0.90,1.10], friction mult ~ U[0.90,1.10]) to the budget of
3,000,000 counted real simulation transitions per (algo, task, seed)
(task §4). Choices below that are not dictated by the task book are
marked  [REBUILD]  and echoed into the report.

Model-based budget scaling [REBUILD]: MBPO/RMBPO keep the RMBPO-paper
per-real-step training cadence (UTD 2, horizon 1, 100k rollouts per model
refresh, model refresh every 250 steps at the paper's 125k/300k/400k
budgets) and scale the model-refresh interval by budget/orig so the
number of refreshes, rollouts and buffer ratios match the paper's regime
at every point of the run.
"""
from __future__ import annotations

from .constants import ADVERSARY_FORCE, ADVERSARY_BODIES, BUDGET_SHARED

SHARED_BUDGET = BUDGET_SHARED

PPO_COMMON = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5,
)  # RobustRLlib rarl_ppo.yaml base PPO hyperparameters (also SB3 defaults)

RARL_PPO = dict(
    ppo=PPO_COMMON,
    n_envs=4,
    normalize=True,           # VecNormalize(obs, reward) on the protagonist
    update_ratio=(1, 1),      # protagonist : adversary blocks
    warmup=0,
    zero_force_prob=0.0,
    adversary_force=ADVERSARY_FORCE,       # ±1 N on world x,z (task §4.1)
    adversary_bodies=ADVERSARY_BODIES,     # RobustRLlib per-task mapping
    adversary_axes=("x", "z"),
    # protagonist and adversary each spend half of the shared 3M counter
    # (1:1 blocks); the last protagonist block is trimmed to land exactly.
    protagonist_steps=SHARED_BUDGET // 2,
)

DR_PPO = dict(
    ppo=PPO_COMMON,
    n_envs=4,
    normalize=True,           # same PPO stack as RARL-PPO (isolates DR vs adversary)
    resample="reset",         # resample physical params every reset (task §4.1)
    steps=SHARED_BUDGET,
)

RMA = dict(
    # SPARC defaults (utils/args.py) carried over; two phases x 1.5M (task §4.2)
    phase1_steps=SHARED_BUDGET // 2,
    phase2_steps=SHARED_BUDGET // 2,
    gamma=0.99, tau=0.005, batch_size=32, learning_starts=5000,
    policy_lr=3e-4, q_lr=3e-4, policy_frequency=2, autotune=True,
    grad_clip_critic=10.0,
    # QR-SAC (reconstructed per SPARC paper §4.1 / task §2)
    num_quantiles=32, quantile_kappa=1.0, top_quantiles_to_drop=0,
    # RMA core
    history_length=50,
    adapter_lr=3e-4,          # SPARC args default
    context_dim=2,            # (mass mult, friction mult)
    # phase-2 adapter trains by regression on the expert latent embedding
    # (representation MSE, task §2 "保留…表征 MSE"); expert frozen.
    device="cpu",  # measured: tiny-batch SAC is 3-4x faster on CPU; GPU+MPS adds per-launch latency
)

QARL = dict(
    # per RMBPO paper appendix D (MuJoCo adaptation of Reddi et al.):
    actor_lr=3e-4, utd=1, adversary_force=ADVERSARY_FORCE,  # force cap 1
    adversary_bodies=ADVERSARY_BODIES, adversary_axes=("x", "z"),
    # gamma temperature teacher (QARLExperiment defaults)
    initial_conc=50.0, target_conc=1.0, fixed_rate=1000.0,
    conc_lower_bound=1e-8, max_kl=0.5, target_kl_threshold=0.01,
    n_samples_for_dist_update=30,
    performance_lower_bound={"hopper": 1.0, "halfcheetah": 10.0, "walker2d": 10.0},
    # episode-based iterations as upstream: 30 episodes/iteration, eval temp = mean
    n_episodes_per_iteration=30,
    # shared env: both players act in the same transition; 3M transitions total
    steps=SHARED_BUDGET,
    device="cpu",  # measured: tiny-batch SAC is 3-4x faster on CPU; GPU+MPS adds per-launch latency
)

# paper budgets per task for the model-based pair
_MBPO_ORIG = {"hopper": 125_000, "halfcheetah": 400_000, "walker2d": 300_000}
_MBPO_SCALE = {t: SHARED_BUDGET / b for t, b in _MBPO_ORIG.items()}

MODEL_BASED_COMMON = dict(
    # Unstable-Baselines/RMBPO config (rmbpo-tmlr configs/robust_mbpo/default.py)
    sac=dict(gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, batch=256,
             hidden=(256, 256), target_entropy="auto"),
    model=dict(hidden=(200, 200, 200, 200), ensemble=7, elites=5, lr=1e-3,
               batch=256, holdout_ratio=0.2, inc_var_loss=True, epsilon=1e-4,
               # upstream compute envelope: at the paper's budgets a model round
               # never exceeds ~10-12k iterations (buffer/256 x early-stopped
               # epochs); the cap only binds in the 3M stretched regime where
               # uncapped rounds would train 10-100x longer than any paper round
               max_model_train_iterations=12000),
    rollout=dict(horizon=1, batch=100_000, mini_batch=10_000, retain_epochs=1,
                 model_env_ratio=0.95),
    warmup=5000,
    save_state_every=100_000,  # OOM-risk insurance: 250k default lost ~114k steps in the 11-19 storm
    utd=2,                     # num_agent_updates_per_env_step (kept per-real-step)
    train_model_interval={t: int(round(250 * _MBPO_SCALE[t])) for t in _MBPO_ORIG},
    max_model_update_epochs_to_improve=5,
    epoch_length=1000,
    device="cpu",  # measured: tiny-batch SAC is 3-4x faster on CPU; GPU+MPS adds per-launch latency
    env_buffer=1_000_000,
    model_buffer=2_000_000,
    budget=SHARED_BUDGET,
)

RMBPO_ETA = {"hopper": 4.0, "halfcheetah": 0.5, "walker2d": 0.5}  # fig 3/7 (task §3)
RMBPO_AUX = dict(
    hidden=(200, 200, 200, 200), lr=1e-3, batch=256,  # robust_head from default.py
    max_train_iterations=2000,  # hard cap per aux round: the paper's 'while
    # improving' rule alone does not terminate when the holdout combined loss
    # keeps improving >1% (robustness term is genuinely improvable); capped to
    # keep the aux a 'small fraction' of the round cost as the paper reports
)


# ---------------------------------------------------------------- phase A ---
PHASE_A = {
    "rarl_ppo": dict(env="v5", steps=2_000_000, adversary_force=5.0,
                     seeds=(0, 1, 2, 3, 4), episodes_per_cell=20,
                     grid="robustrllib part1_mujoco (24 perturbed cells + nominal)"),
    "rma": dict(env="v5-wind", phase_steps=3_000_000, episodes_per_cell=30,
                seeds=(0, 1, 2, 3, 4),
                grid="9x9 wind grid per task; aggregate strictly-outside-train-rect"),
    "dr_ppo": dict(env="v3-recon", steps=1_500_000, episodes_per_cell=30,
                   seeds=(0, 1, 2, 3, 4), eval_point="mass x1.2 (nominal friction)",
                   mass_range={"hopper": (0.5, 6.5), "halfcheetah": (3.5, 9.5),
                               "walker2d": (0.5, 6.5)},
                   friction_range={"hopper": (0.1, 3.0), "halfcheetah": (0.2, 0.8),
                                   "walker2d": (0.5, 2.0)}),
    "qarl": dict(env="v4", steps={"hopper": 1_000_000, "halfcheetah": 3_000_000,
                                  "walker2d": 1_000_000},
                 eval_grid="unified mass x friction grid (§4.3)", episodes_per_cell=30,
                 seeds=(0, 1, 2, 3, 4)),
    "mbpo": dict(env="v4", weights="upstreams/rmbpo-tmlr/weights/<task>-v4/mbpo/",
                 episodes_per_cell=30, evaluator="author main_eval.py"),
    "rmbpo": dict(env="v4",
                  weights={"hopper": "rmbpo_4", "halfcheetah": "rmbpo_05",
                           "walker2d": "rmbpo_05"},
                  episodes_per_cell=30, evaluator="author main_eval.py"),
}
