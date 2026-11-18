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

from .constants import ALGOS, EPISODES_DIR, MANIFEST, REPORTS_DIR, RUNS_DIR, TASKS
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
    if argv and argv[0] == "full":
        return main_full()
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


# ------------------------------------------------------------------ phase A
SOURCE_REFERENCE = {
    # task -> per-algo reference values from the task book (source labels)
    "rarl_ppo": {"hopper": 1145, "halfcheetah": 1781, "walker2d": 2233,
                 "source": "RobustRLlib All (normalized score back-converted)"},
    "rma": {"hopper": 1307.96, "halfcheetah": 9033.87, "walker2d": 2306.23,
            "source": "SPARC paper Table 5 mean ± s.e.m. (OOD wind cells)"},
    "dr_ppo": {"hopper": 1589, "halfcheetah": 2199, "walker2d": 2196,
               "source": "AD-RRL fig 2 interpolation at 1.2x mass"},
    "qarl": {"hopper": 1330, "halfcheetah": 4080, "walker2d": 1560,
             "source": "RMBPO paper heat-map estimates"},
    "mbpo": {"hopper": 1630, "halfcheetah": 6560, "walker2d": 2110,
             "source": "RMBPO paper heat-map estimates"},
    "rmbpo": {"hopper": 2100, "halfcheetah": 6030, "walker2d": 2850,
              "source": "RMBPO paper heat-map estimates"},
}

REPRO_KIND = {
    "rarl_ppo": "按原协议重训（RobustRLlib v5 官方入口）",
    "rma": "按原协议重训（SPARC v5 风场 + QR-SAC 重建组件）",
    "dr_ppo": "基于公开信息重建（SB3 PPO + AD-RRL v3 FullObs 语义 + DR 区间）",
    "qarl": "上游实现移植重跑（v4，RMBPO 论文口径）",
    "mbpo": "发布权重复测（作者评估器原样）",
    "rmbpo": "发布权重复测（作者评估器原样）",
}


def phase_a_table() -> str:
    import glob
    lines = ["# 阶段 A · 按来源协议复测表", ""]
    header = "| 方法 | " + " | ".join(TASKS) + " | 复现口径 | 参照 |"
    lines += [header, "|---|" + "---:|" * len(TASKS) + "|---|---|"]
    for algo in ALGOS:
        row = [f"`{algo}`"]
        for task in TASKS:
            # prefer the lane's own summary; fall back to episodes parquet
            val = None
            for d in glob.glob(os.path.join(RUNS_DIR, "source", algo, task,
                                             "seed*", "summary.json")):
                s = json.load(open(d))
                for key in ("all_24cell_mean", "ood_mean", "point_mean",
                            "grid_mean", "author_grid_mean", "common64_mean"):
                    if s.get(key) is not None:
                        val = s[key]
                        break
                if val is None:
                    continue
                break
            if val is None:
                from .eval.records import load_all
                df = load_all("source")
                sub = df[(df["algo"] == algo) & (df["task"] == task)]
                if not sub.empty:
                    val = float(sub["return"].mean())
            row.append("—" if val is None else f"{val:.0f}")
        ref = SOURCE_REFERENCE.get(algo, {})
        row.append(REPRO_KIND.get(algo, ""))
        row.append(" / ".join(f"{t}:{ref[t]:.0f}" for t in TASKS if t in ref))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_full(main_b: dict, costs: dict) -> str:
    """Full report.md: phase-A table + phase-B table + protocol notes."""
    parts = [phase_a_table(), "", "---", ""]
    parts.append(render_report(main_b, costs))
    parts += ["", "## 口径说明", "",
              "- 阶段 A 各方法按其来源论文协议复测；MBPO/RMBPO 用作者评估器原样"
              "（其 evaluate_v2 不计 done 步奖励——末步计分口径差异已注明），"
              "RARL-PPO 用 RobustRLlib 官方 v5 入口重训，RMA 用 SPARC 官方入口"
              "（QR-SAC 组件按论文重建），QARL 为上游实现移植重跑，DR-PPO 为"
              "公开信息重建。",
              "- 阶段 B 为统一质量/摩擦分布重训：六法同一 PhysSampler"
              "（m/m0、摩擦倍率均 U[0.90,1.10]），同一 3,000,000 真实仿真转移预算，"
              "同一冻结评估器（含末步奖励、30 复位种子 20000-20029、确定性策略、"
              "无对手无动作噪声）。",
              "- 主表单元 = 5 训练种子 × (格点内 30 回合平均 → 90/154/64 格等权平均)"
              " → 种子平均；95% CI 为种子层 bootstrap。",
              "- 参照值为任务书给出的历史读数（RobustLLib All 反算 / SPARC 表 5 / "
              "AD-RRL 图 2 插值 / RMBPO 热图估读），仅作复现对照，不进入统一表。"]
    return "\n".join(parts)


def main_full() -> int:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    from .eval.records import load_all
    df = load_all("shared")
    main_t = phase_b_tables(df) if not df.empty else {}
    costs = cost_table()
    text = render_full(main_t, costs)
    path = os.path.join(REPORTS_DIR, "report.md")
    open(path, "w").write(text)
    print(text[:3000])
    print(f"\nwritten: {path}")
    return 0
