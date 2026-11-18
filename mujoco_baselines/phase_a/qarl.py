"""Phase-A lane: QARL per the RMBPO paper protocol (task §3).

The RMBPO paper appendix D: QARL (Reddi et al. implementation) trained on
the DEFAULT v4 env (no mass/friction randomisation) with the adversary, UTD
1, actor lr 3e-4, adversary force cap 1, ~1M steps (HalfCheetah 3M); scored
on the unified mass x friction grid with the frozen protagonist and the
no-adversary evaluation default (alpha input 0).

Uses the mujoco_baselines QARL port with cfg["env_mode"]="nominal".

Driver: python -m mujoco_baselines.phase_a.qarl <task> <seed>
"""
from __future__ import annotations

import json
import os
import sys
import time

from .. import config as CF
from ..constants import EXP_ROOT, RUNS_DIR
from .. import grids as G

STEPS = {"hopper": 1_000_000, "halfcheetah": 3_000_000, "walker2d": 1_000_000}


def main(task: str, seed: int) -> int:
    from ..algos import qarl as qarl_mod
    from ..eval.evaluator import evaluate_checkpoint
    from ..eval.records import save_records
    from ..run.job import _sha256

    job_dir = os.path.join(RUNS_DIR, "source", "qarl", task, f"seed{seed}")
    os.makedirs(job_dir, exist_ok=True)
    status_p = os.path.join(job_dir, "status.json")
    if os.path.exists(status_p) and json.load(open(status_p)).get("state") == "DONE":
        print(f"[qarl-a] already done {job_dir}")
        return 0
    json.dump({"state": "RUNNING", "pid": os.getpid(), "started": time.time()},
              open(status_p, "w"))
    cfg = dict(CF.QARL)
    cfg["env_mode"] = "nominal"          # default v4 physics, adversary only
    cfg["steps"] = STEPS[task]
    cfg["task"] = task
    cfg["train_seed"] = seed
    json.dump(cfg, open(os.path.join(job_dir, "config.json"), "w"),
              indent=1, default=str)
    t0 = time.time()
    try:
        qarl_mod.train(job_dir, task, seed, cfg)
        counters = json.load(open(os.path.join(job_dir, "counters.json")))
        real = int(counters.get("real_transitions", 0))
        assert real >= cfg["steps"] - 5, f"budget {real} < {cfg['steps']}"
        adapter = qarl_mod.load_eval_adapter(job_dir, task, seed, cfg)
        records, eval_s = evaluate_checkpoint(adapter, task)
        ck = os.path.join(job_dir, "ckpt", "final.pt")
        save_records(records, "source", "qarl", task, seed, "final", eval_s)
        import numpy as np
        grid_recs = [r for r in records if not r["is_nominal"]]
        cell = np.mean([r["return"] for r in grid_recs])
        ood = np.mean([r["return"] for r in records if r["is_ood"]])
        nom = np.mean([r["return"] for r in records if r["is_nominal"]])
        summary = {"grid_mean": float(cell), "ood_mean": float(ood),
                   "nominal_mean": float(nom), "steps": cfg["steps"],
                   "reference": "RMBPO paper fig 3/7 heatmap estimates "
                                "~1330/~4080/~1560"}
        json.dump(summary, open(os.path.join(job_dir, "summary.json"), "w"), indent=1)
        manifest = {
            "protocol": "source", "algo": "qarl", "task": task, "train_seed": seed,
            "state": "DONE", "real_transitions": real, "wall_s": time.time() - t0,
            "eval_wall_s": eval_s, "checkpoint": ck,
            "checkpoint_sha256": _sha256(ck), "command": " ".join(sys.argv),
            "n_eval_episodes": len(records),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(EXP_ROOT, "runs.jsonl"), "a") as f:
            f.write(json.dumps(manifest) + "\n")
        json.dump({"state": "DONE", "rc": 0, "finished": time.time(),
                   "summary": summary}, open(status_p, "w"), indent=1)
        print(f"[qarl-a] DONE {task}/{seed}: {summary}")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        open(os.path.join(job_dir, "stderr_tail.txt"), "a").write(traceback.format_exc())
        json.dump({"state": "FAILED", "error": str(e),
                   "finished": time.time()}, open(status_p, "w"), indent=1)
        print(f"[qarl-a] FAILED {task}/{seed}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m mujoco_baselines.phase_a.qarl <task> <seed>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
