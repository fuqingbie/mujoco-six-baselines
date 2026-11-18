"""Phase-A lane: RMA per the SPARC protocol (task §3).

Runs the SPARC checkout natively (envs/sparc, gymnasium 1.2.0 stack) with the
reconstructed algorithms/qr_sac.py (see upstreams.lock.json local changes):
  phase 1: train.py --alg rma_phase1 --env Wind<Task>-v5
           --sparc-expert-alg qr_sac --total-timesteps 3000000  (QR-SAC expert)
  phase 2: train.py --alg rma_phase2 --expert-model-path <phase1 expert>
           --total-timesteps 3000000                              (frozen expert
           + 50-step-history adapter, representation MSE)
Then the 9x9 wind grid evaluation (rma_eval_grid.py, 30 episodes/condition,
reset seeds 20000..20029, authors' sampled-action eval semantics).

Checkpoint rule: phase-1 and phase-2 budget endpoints are LOCKED (the paper's
Pareto selection rule for the phase-1 checkpoint is not recoverable from the
public materials; noted in the report as a selection-rule difference).

Driver: python -m mujoco_baselines.phase_a.rma <task> <seed>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from ..constants import EXP_ROOT, RUNS_DIR

SPARC_PY = os.path.join(EXP_ROOT, "envs/sparc/bin/python")
CHECKOUT = os.path.join(EXP_ROOT, "upstreams/sparc")
STEPS = 3_000_000

ENV_IDS = {"hopper": "WindHopper-v5", "halfcheetah": "WindHalfCheetah-v5",
           "walker2d": "WindWalker2d-v5"}


def main(task: str, seed: int) -> int:
    job_dir = os.path.join(RUNS_DIR, "source", "rma", task, f"seed{seed}")
    os.makedirs(job_dir, exist_ok=True)
    status_p = os.path.join(job_dir, "status.json")
    if os.path.exists(status_p) and json.load(open(status_p)).get("state") == "DONE":
        print(f"[rma-a] already done {job_dir}")
        return 0
    json.dump({"state": "RUNNING", "pid": os.getpid(), "started": time.time()},
              open(status_p, "w"))
    env = dict(os.environ)
    env["PYTHONPATH"] = CHECKOUT
    env["SPARC_CHECKOUT"] = CHECKOUT
    env["WANDB_MODE"] = "disabled"
    env.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mjb-no-mps")
    env.setdefault("OMP_NUM_THREADS", "8")
    env_id = ENV_IDS[task]
    t0 = time.time()
    try:
        # ---- phase 1: QR-SAC expert (sparc_expert_alg=qr_sac per paper §4.1)
        expert_path = os.path.join(job_dir, "models", f"{env_id}_rma_phase1_{seed}_expert.pth")
        if not os.path.exists(expert_path):
            cmd1 = [SPARC_PY, "train.py", "--alg", "rma_phase1", "--env", env_id,
                    "--sparc-expert-alg", "qr_sac", "--total-timesteps", str(STEPS),
                    "--seed", str(seed), "--wandb-mode", "disabled"]
            with open(os.path.join(job_dir, "phase1.stdout.log"), "a") as fo, \
                 open(os.path.join(job_dir, "phase1.stderr.log"), "a") as fe:
                rc = subprocess.call(cmd1, cwd=job_dir, env=env, stdout=fo, stderr=fe)
            assert rc == 0, f"phase1 rc={rc}"
        # ---- phase 2: adapter against the frozen expert
        adapter_path = os.path.join(job_dir, "models", f"{env_id}_rma_phase2_{seed}_adapter.pth")
        if not os.path.exists(adapter_path):
            cmd2 = [SPARC_PY, "train.py", "--alg", "rma_phase2", "--env", env_id,
                    "--total-timesteps", str(STEPS), "--seed", str(seed),
                    "--wandb-mode", "disabled",
                    "--expert-model-path", expert_path]
            with open(os.path.join(job_dir, "phase2.stdout.log"), "a") as fo, \
                 open(os.path.join(job_dir, "phase2.stderr.log"), "a") as fe:
                rc = subprocess.call(cmd2, cwd=job_dir, env=env, stdout=fo, stderr=fe)
            assert rc == 0, f"phase2 rc={rc}"
        # ---- 9x9 wind grid evaluation
        rec_json = os.path.join(job_dir, "grid_records.json")
        if not os.path.exists(rec_json):
            cmd3 = [SPARC_PY, os.path.join(EXP_ROOT, "mujoco_baselines/phase_a/rma_eval_grid.py"),
                    task, adapter_path, "--out", rec_json]
            with open(os.path.join(job_dir, "eval.stdout.log"), "a") as fo, \
                 open(os.path.join(job_dir, "eval.stderr.log"), "a") as fe:
                rc = subprocess.call(cmd3, cwd=job_dir, env=env, stdout=fo, stderr=fe)
            assert rc == 0, f"eval rc={rc}"
        records = json.load(open(rec_json))
        from ..eval.records import save_records
        save_records(records, "source", "rma", task, seed, "phase2_final",
                     time.time() - t0)
        import numpy as np
        ood = [r["return"] for r in records if r["is_ood"]]
        ind = [r["return"] for r in records if not r["is_ood"]]
        summary = {
            "ood_mean": float(np.mean(ood)) if ood else None,
            "ind_mean": float(np.mean(ind)) if ind else None,
            "n_ood_cells": len({r["condition_str"] for r in records if r["is_ood"]}),
            "phase_steps": STEPS, "phases": 2,
            "selection_rule": "budget endpoints locked (paper Pareto rule not public)",
            "reference": "SPARC paper Table 5 (mean ± s.e.m. over 5 seeds, OOD cells)",
        }
        json.dump(summary, open(os.path.join(job_dir, "summary.json"), "w"), indent=1)
        manifest = {
            "protocol": "source", "algo": "rma", "task": task, "train_seed": seed,
            "state": "DONE", "real_transitions": 2 * STEPS,
            "wall_s": time.time() - t0, "command": " ".join(sys.argv),
            "checkpoint": "phase2_final", "n_eval_episodes": len(records),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(EXP_ROOT, "runs.jsonl"), "a") as f:
            f.write(json.dumps(manifest) + "\n")
        json.dump({"state": "DONE", "rc": 0, "finished": time.time(),
                   "summary": summary}, open(status_p, "w"), indent=1)
        print(f"[rma-a] DONE {task}/{seed}: {summary}")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        open(os.path.join(job_dir, "stderr_tail.txt"), "a").write(traceback.format_exc())
        json.dump({"state": "FAILED", "error": str(e),
                   "finished": time.time()}, open(status_p, "w"), indent=1)
        print(f"[rma-a] FAILED {task}/{seed}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m mujoco_baselines.phase_a.rma <task> <seed>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
