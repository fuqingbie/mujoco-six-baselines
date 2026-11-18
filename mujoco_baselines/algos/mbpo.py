"""MBPO (phase B, shared v4 stack).

``train(job_dir, task, seed, cfg)`` runs the Unstable-Baselines MBPO port in
``mbpo_core.MbpoTrainer`` (use_aux=False) to ``cfg['budget']`` counted real
transitions.  ``load_eval_adapter`` rebuilds the deterministic SAC policy from
``ckpt/final.pt`` for the frozen evaluator.
"""
from __future__ import annotations

import os

from ..config import MODEL_BASED_COMMON
from .mbpo_core import MbpoTrainer, load_sac_adapter

__all__ = ["train", "load_eval_adapter", "MbpoTrainer"]


def _full_cfg(cfg: dict | None) -> dict:
    """cfg = dict(MODEL_BASED_COMMON) plus job overrides; fill any gaps from the
    frozen config so a partial cfg still trains with the frozen recipe."""
    merged = dict(MODEL_BASED_COMMON)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def train(job_dir: str, task: str, seed: int, cfg: dict | None = None) -> None:
    cfg = _full_cfg(cfg)
    # seed before construction so network initialization is reproducible
    from .mbpo_core import set_global_seed
    set_global_seed(seed)
    trainer = MbpoTrainer(task=task, seed=seed, cfg=cfg, job_dir=job_dir,
                          use_aux=False, bypass_aux=False)
    trainer.train()


def load_eval_adapter(job_dir: str, task: str, seed: int,
                      cfg: dict | None = None):
    path = os.path.join(job_dir, "ckpt", "final.pt")
    if not os.path.exists(path):
        # fall back to the newest intermediate state file (partial run);
        # sort numerically — state_1000000.pt < state_5200.pt lexicographically
        import glob
        import re
        states = glob.glob(os.path.join(job_dir, "ckpt", "state_*.pt"))

        def step_of(p):
            m = re.search(r"state_(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else -1
        if not states:
            raise FileNotFoundError(f"no checkpoint under {job_dir}/ckpt")
        path = max(states, key=step_of)
    return load_sac_adapter(path, cfg=cfg)
