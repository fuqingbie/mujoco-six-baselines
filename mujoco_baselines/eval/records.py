"""Per-episode record IO: episodes.parquet writer/reader (task §6).

Schema (phase B):
  protocol, algo, task, train_seed, checkpoint, condition_id, condition_str,
  mass_kg, friction_multiplier, eval_seed, return, length, terminated,
  truncated, is_ood, is_nominal, wall_seconds
Phase A records add source-protocol columns where available (e.g. force
components for RARL/RMA wind lanes are stored in `extra` as JSON).
"""
from __future__ import annotations

import json
import os

import pandas as pd

from ..constants import EPISODES_DIR

_COLUMNS = ["protocol", "algo", "task", "train_seed", "checkpoint",
            "condition_id", "condition_str", "mass_kg", "friction_multiplier",
            "eval_seed", "return", "length", "terminated", "truncated",
            "is_ood", "is_nominal", "wall_seconds", "extra"]


def episode_path(protocol: str, algo: str, task: str) -> str:
    d = os.path.join(EPISODES_DIR, protocol, algo, task)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "episodes.parquet")


def save_records(records, protocol: str, algo: str, task: str, train_seed,
                 checkpoint: str, wall_seconds: float | None = None):
    """Append (idempotent per condition) episode records for one checkpoint."""
    path = episode_path(protocol, algo, task)
    df = pd.DataFrame(records)
    if df.empty:
        return path
    df["protocol"] = protocol
    df["algo"] = algo
    df["train_seed"] = train_seed
    df["checkpoint"] = checkpoint
    df["wall_seconds"] = wall_seconds
    if "extra" not in df:
        df["extra"] = None
    df = df[[c for c in _COLUMNS if c in df.columns]]
    if os.path.exists(path):
        old = pd.read_parquet(path)
        # idempotency: replace rows for the same checkpoint (reruns overwrite)
        key = ["train_seed", "checkpoint", "condition_id", "eval_seed"]
        df = pd.concat([old.merge(df[key].drop_duplicates(), on=key, how="left",
                                  indicator=True).query('_merge == "left_only"').drop(columns="_merge"),
                        df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def load_all(protocol: str | None = None) -> pd.DataFrame:
    frames = []
    if not os.path.isdir(EPISODES_DIR):
        return pd.DataFrame(columns=_COLUMNS)
    for root, _dirs, files in os.walk(EPISODES_DIR):
        if "episodes.parquet" in files:
            frames.append(pd.read_parquet(os.path.join(root, "episodes.parquet")))
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if protocol is not None:
        df = df[df["protocol"] == protocol]
    return df
