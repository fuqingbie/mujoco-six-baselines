"""Phase-A lane: RARL-PPO per the RobustRLlib protocol (task §3).

Official entry points, run natively in the mjb5 env from the pinned checkout:
  1. resolve the experiment card for (task, seed) with train.steps=2,000,000
     and out_dir under runs/source/rarl_ppo/<task>/seed<k>/
  2. `python baselines/rarl/train.py --config <resolved>`  (protagonist 2M
     steps, 1:1 alternating adversary blocks — both step counts recorded from
     the run's progress.csv / checkpoint state)
  3. `python baselines/evaluate.py --run <out> --eval part1_mujoco.yaml
     --checkpoint last --episodes 20` (final checkpoint, 20 paired episodes
     per condition, eval seed 10000, observation normalisation kept)
  4. parse eval/part1_mujoco.json (per-episode returns per condition) into
     episodes records + summary: 24-perturbed-cell mean ("All" alignment) and
     the nominal cell.

Driver: python -m mujoco_baselines.phase_a.rarl <task> <seed>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from ..constants import EXP_ROOT, RUNS_DIR

REPO = os.path.join(EXP_ROOT, "upstreams/robustrllib")
MJB5_PY = os.path.join(EXP_ROOT, "envs/mjb5/bin/python")
STEPS = 2_000_000
EPISODES = 20

GRID = "robustrllib/configs/eval/part1_mujoco.yaml"


def main(task: str, seed: int) -> int:
    from ..config import PHASE_A
    card = f"robustrllib/configs/experiment/rarl_ppo_{task}.yaml"
    job_dir = os.path.join(RUNS_DIR, "source", "rarl_ppo", task, f"seed{seed}")
    os.makedirs(job_dir, exist_ok=True)
    status_p = os.path.join(job_dir, "status.json")
    if os.path.exists(status_p) and json.load(open(status_p)).get("state") == "DONE":
        print(f"[rarl-a] already done {job_dir}")
        return 0
    json.dump({"state": "RUNNING", "pid": os.getpid(), "started": time.time()},
              open(status_p, "w"))
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    env.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mjb-no-mps")
    env["CUDA_VISIBLE_DEVICES"] = ""  # SB3 auto-device -> CPU (GPU shared with
    # an external tenant; PPO's small ops are faster on CPU under that load)
    env.setdefault("OMP_NUM_THREADS", "8")
    t0 = time.time()
    try:
        if not os.path.exists(os.path.join(job_dir, "ckpt")):
            # 1) materialise the resolved run config
            out = subprocess.run(
                [MJB5_PY, "-c",
                 f"from baselines.common import resolve, write_run_config; "
                 f"cfg = resolve({card!r}, seed={seed}, out={job_dir!r}); "
                 f"cfg['train']['steps'] = {STEPS}; "
                 f"write_run_config(cfg, {job_dir!r})"],
                cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
            assert out.returncode == 0, out.stderr[-2000:]
            # 2) official training entry point
            with open(os.path.join(job_dir, "stdout.log"), "a") as fo, \
                 open(os.path.join(job_dir, "stderr.log"), "a") as fe:
                rc = subprocess.call(
                    [MJB5_PY, "baselines/rarl/train.py", "--config",
                     os.path.join(job_dir, "config.yaml")],
                    cwd=REPO, env=env, stdout=fo, stderr=fe)
            assert rc == 0, f"train.py rc={rc} (see stderr.log)"
        # 3) official evaluator
        with open(os.path.join(job_dir, "stdout.log"), "a") as fo, \
             open(os.path.join(job_dir, "stderr.log"), "a") as fe:
            rc = subprocess.call(
                [MJB5_PY, "baselines/evaluate.py", "--run", job_dir,
                 "--eval", GRID, "--checkpoint", "last",
                 "--episodes", str(EPISODES)],
                cwd=REPO, env=env, stdout=fo, stderr=fe)
        assert rc == 0, f"evaluate.py rc={rc}"
        # 4) parse
        eval_json = os.path.join(job_dir, "eval", "part1_mujoco.json")
        data = json.load(open(eval_json))
        records, summary = parse(data, task, seed)
        from ..eval.records import save_records
        save_records(records, "source", "rarl_ppo", task, seed, "last",
                     time.time() - t0)
        # protagonist/adversary step split from the run's progress.csv
        prog = os.path.join(job_dir, "progress.csv")
        steps_split = {}
        if os.path.exists(prog):
            import csv
            with open(prog) as f:
                rows = list(csv.DictReader(f))
            if rows:
                steps_split = {"protagonist_steps": int(float(rows[-1]["steps"]))}
        steps_split["adversary_steps_approx"] = steps_split.get(
            "protagonist_steps", STEPS)  # 1:1 alternating blocks; exact from log
        summary.update(steps_split)
        json.dump(summary, open(os.path.join(job_dir, "summary.json"), "w"),
                  indent=1)
        manifest = {
            "protocol": "source", "algo": "rarl_ppo", "task": task,
            "train_seed": seed, "state": "DONE",
            "real_transitions": int(STEPS + STEPS),
            "wall_s": time.time() - t0, "command": " ".join(sys.argv),
            "checkpoint": "last", "n_eval_episodes": len(records),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(EXP_ROOT, "runs.jsonl"), "a") as f:
            f.write(json.dumps(manifest) + "\n")
        json.dump({"state": "DONE", "rc": 0, "finished": time.time(),
                   "summary": summary}, open(status_p, "w"), indent=1)
        print(f"[rarl-a] DONE {task}/{seed}: {summary}")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        open(os.path.join(job_dir, "stderr_tail.txt"), "a").write(
            traceback.format_exc())
        json.dump({"state": "FAILED", "error": str(e),
                   "finished": time.time()}, open(status_p, "w"), indent=1)
        print(f"[rarl-a] FAILED {task}/{seed}: {e}", file=sys.stderr)
        return 1


def parse(data, task, seed):
    """eval JSON -> episode records + summary (24-cell 'All' + nominal)."""
    from ..grids import nominal_torso_mass
    m0 = nominal_torso_mass(task)
    records = []
    perturbed_means = []
    nominal_mean = None
    for i, (name, cond) in enumerate(data.items()):
        if name == "nominal":
            nominal_mean = cond["return_mean"]
        else:
            perturbed_means.append(cond["return_mean"])
        returns = cond.get("returns", [])
        extra = {"condition_name": name,
                 "axis": cond.get("axis"), "quartile": cond.get("quartile"),
                 "severity": cond.get("severity"),
                 "protocol": "robustrllib part1_mujoco v5, 20 episodes, eval seed 10000"}
        for ep_i, r in enumerate(returns):
            records.append({
                "task": task, "condition_id": i,
                "condition_str": name,
                "mass_kg": m0, "friction_multiplier": 1.0,
                "is_ood": name != "nominal",
                "is_nominal": name == "nominal",
                "eval_seed": 10000 + ep_i,
                "return": float(r), "length": -1,
                "terminated": False, "truncated": False,
                "extra": json.dumps(extra),
            })
    summary = {
        "all_24cell_mean": sum(perturbed_means) / max(len(perturbed_means), 1),
        "nominal_mean": nominal_mean,
        "n_conditions": len(data),
        "reference": "RobustRLlib All (normalized-score back-conversion ~ raw return)",
    }
    return records, summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m mujoco_baselines.phase_a.rarl <task> <seed>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
