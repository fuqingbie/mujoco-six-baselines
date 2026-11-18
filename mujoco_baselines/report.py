"""Score aggregation and the two 6x3 report tables (task §6).

Phase-B main-table cell: mean over 5 training seeds of the equal-weight mean
over grid cells of the 30-episode mean. Also reported per task: unseen-subset
(OOD) mean, nominal-condition mean, low-tail mean (per seed, worst
ceil(0.10 * n_cells) cells by per-cell average), 95% bootstrap CI over
training seeds, and compute cost (transitions, updates, wall seconds).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from .constants import ALGOS, EPISODES_DIR, MANIFEST, REPORTS_DIR, TASKS
from . import grids as G

RNG = np.random.default_rng(20261118)


def seed_cell_mean(df: pd.DataFrame) -> pd.Series:
    """Series indexed (algo, task, train_seed) -> equal-weight cell mean."""
    grid_only = df[~df["is_nominal"]]
    per_cell = grid_only.groupby(["algo", "task", "train_seed", "condition_id"])["return"].mean()
    per_seed = per_cell.groupby(level=["algo", "task", "train_seed"]).mean()
    return per_seed


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000) -> tuple:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float(values.mean()) if len(values) else float("nan"),) * 2
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ood_mean(df: pd.DataFrame) -> pd.Series:
    ood = df[(df["is_ood"]) & (~df["is_nominal"])]
    per_cell = ood.groupby(["algo", "task", "train_seed", "condition_id"])["return"].mean()
    return per_cell.groupby(level=["algo", "task", "train_seed"]).mean()


def nominal_mean(df: pd.DataFrame) -> pd.Series:
    nom = df[df["is_nominal"]]
    per_seed = nom.groupby(["algo", "task", "train_seed"])["return"].mean()
    return per_seed


def lowtail_mean(df: pd.DataFrame, frac: float = 0.10) -> pd.Series:
    """Per seed: worst ceil(frac * n_cells) cells (by per-cell mean), averaged."""
    grid_only = df[~df["is_nominal"]]
    per_cell = grid_only.groupby(["algo", "task", "train_seed", "condition_id"])["return"].mean()
    out = {}
    for (algo, task, seed), cells in per_cell.groupby(level=["algo", "task", "train_seed"]):
        n = len(cells)
        k = math.ceil(frac * G.n_conditions(task))
        k = min(k, n)
        worst = cells.nsmallest(k)
        out[(algo, task, seed)] = float(worst.mean())
    return pd.Series(out)


def phase_b_tables(df: pd.DataFrame) -> dict:
    main = {}
    for algo in ALGOS:
        for task in TASKS:
            sub = df[(df["algo"] == algo) & (df["task"] == task)]
            if sub.empty:
                main[(algo, task)] = None
                continue
            cm = seed_cell_mean(sub)
            om = ood_mean(sub)
            nm = nominal_mean(sub)
            lt = lowtail_mean(sub)
            main[(algo, task)] = {
                "mean": float(cm.mean()),
                "ci95": bootstrap_ci(cm.values),
                "n_seeds": int(len(cm)),
                "ood_mean": float(om.mean()) if len(om) else float("nan"),
                "nominal_mean": float(nm.mean()) if len(nm) else float("nan"),
                "lowtail_mean": float(lt.mean()) if len(lt) else float("nan"),
            }
    return main


def cost_table() -> dict:
    costs = {}
    if not os.path.exists(MANIFEST):
        return costs
    for line in open(MANIFEST):
        try:
            m = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if m.get("protocol") != "shared":
            continue
        key = (m["algo"], m["task"])
        c = costs.setdefault(key, {"jobs": 0, "real_transitions": 0, "wall_s": 0.0})
        c["jobs"] += 1
        c["real_transitions"] += int(m.get("real_transitions", 0))
        c["wall_s"] += float(m.get("wall_s", 0.0))
    return costs


def render_report(main: dict, costs: dict) -> str:
    lines = ["# MuJoCo 六基线 · 阶段 B 统一重训成绩表", "",
             "每格：5 训练种子 × 30 评测回合/格点，先格点内平均、后 90/154/64 格等权、"
             "再对种子平均；95% CI 为训练种子层级 bootstrap。", ""]
    header = "| 方法 | " + " | ".join(t for t in TASKS) + " |"
    sep = "|---|" + "---:|" * len(TASKS)
    lines += [header, sep]
    for algo in ALGOS:
        row = [f"`{algo}`"]
        for task in TASKS:
            c = main.get((algo, task))
            if c is None:
                row.append("—")
            else:
                lo, hi = c["ci95"]
                row.append(f"{c['mean']:.0f} [{lo:.0f}, {hi:.0f}]")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 未见子集 / 名义 / 低尾 / 成本", ""]
    header2 = "| 方法·任务 | 全网格 | 未见子集 | 名义 | 低尾10% | 训练交互 | 墙钟 |"
    lines += [header2, "|---|---:|---:|---:|---:|---:|---:|"]
    for algo in ALGOS:
        for task in TASKS:
            c = main.get((algo, task))
            k = costs.get((algo, task))
            if c is None:
                lines.append(f"| {algo}·{task} | — | — | — | — | — | — |")
                continue
            cost = f"{k['real_transitions']:,}" if k else "—"
            wall = f"{k['wall_s'] / 3600:.1f}h" if k else "—"
            lines.append(
                f"| {algo}·{task} | {c['mean']:.0f} | {c['ood_mean']:.0f} | "
                f"{c['nominal_mean']:.0f} | {c['lowtail_mean']:.0f} | {cost} | {wall} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    from .eval.records import load_all
    df = load_all("shared")
    if df.empty:
        print("no shared-phase episode records yet")
        return 1
    main_t = phase_b_tables(df)
    costs = cost_table()
    text = render_report(main_t, costs)
    path = os.path.join(REPORTS_DIR, "report.md")
    open(path, "w").write(text)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in main_t.items()},
              open(os.path.join(REPORTS_DIR, "phase_b_main.json"), "w"), indent=1)
    print(text)
    print(f"\nwritten: {path}")
    return 0
