"""Single phase-B job: train one (algo, task, seed) to budget, then frozen-eval.

    python -m mujoco_baselines.run.job shared <algo> <task> <seed>
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time

from .. import config as CF
from .. import constants as C
from .. import grids as G
from ..eval.evaluator import evaluate_checkpoint
from ..eval.records import save_records


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _algo_cfg(algo: str, task: str) -> dict:
    if algo == "rarl_ppo":
        cfg = dict(CF.RARL_PPO)
    elif algo == "dr_ppo":
        cfg = dict(CF.DR_PPO)
    elif algo == "rma":
        cfg = dict(CF.RMA)
    elif algo == "qarl":
        cfg = dict(CF.QARL)
    elif algo in ("mbpo", "rmbpo"):
        cfg = json.loads(json.dumps(CF.MODEL_BASED_COMMON))
        if algo == "rmbpo":
            cfg["eta"] = CF.RMBPO_ETA[task]
            cfg["aux"] = dict(CF.RMBPO_AUX)
    else:
        raise ValueError(algo)
    cfg["task"] = task
    cfg["train_seed"] = None  # filled by caller
    return cfg


def main(algo: str, task: str, seed: int, protocol: str = "shared") -> int:
    job_dir = os.path.join(C.RUNS_DIR, protocol, algo, task, f"seed{seed}")
    os.makedirs(job_dir, exist_ok=True)
    status_path = os.path.join(job_dir, "status.json")
    if os.path.exists(status_path):
        st = json.load(open(status_path))
        if st.get("state") == "DONE":
            print(f"[job] already DONE: {job_dir}")
            return 0

    cfg = _algo_cfg(algo, task)
    cfg["train_seed"] = seed
    budget = cfg.get("budget") or cfg.get("steps") or C.BUDGET_SHARED
    json.dump(cfg, open(os.path.join(job_dir, "config.json"), "w"), indent=1, default=str)
    json.dump({"argv": sys.argv, "cwd": os.getcwd(),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(job_dir, "cmd.json"), "w"), indent=1)
    json.dump({"state": "RUNNING", "pid": os.getpid(),
               "started": time.time()}, open(status_path, "w"))
    t0 = time.time()

    import importlib
    mod = importlib.import_module(f"mujoco_baselines.algos.{algo}")
    try:
        mod.train(job_dir, task, seed, cfg)
        counters = json.load(open(os.path.join(job_dir, "counters.json")))
        real = int(counters.get("real_transitions", 0))
        assert real >= budget, f"budget not reached: {real} < {budget}"
        adapter = mod.load_eval_adapter(job_dir, task, seed, cfg)
        records, eval_s = evaluate_checkpoint(adapter, task)
        ck = _newest_ckpt(job_dir)
        save_records(records, protocol, algo, task, seed, ck, eval_s)
        manifest = {
            "protocol": protocol, "algo": algo, "task": task, "train_seed": seed,
            "state": "DONE", "real_transitions": real,
            "counters": counters, "wall_s": time.time() - t0, "eval_wall_s": eval_s,
            "checkpoint": ck, "checkpoint_sha256": _sha256(ck),
            "python": platform.python_version(),
            "n_eval_episodes": len(records),
            "command": " ".join(sys.argv),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(C.MANIFEST, "a") as f:
            f.write(json.dumps(manifest) + "\n")
        json.dump({"state": "DONE", "pid": os.getpid(), "started": st_started(status_path),
                   "finished": time.time(), "manifest": manifest},
                  open(status_path, "w"), indent=1)
        print(f"[job] DONE {job_dir}: {real} transitions, {len(records)} eval episodes")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        with open(os.path.join(job_dir, "stderr_tail.txt"), "a") as f:
            f.write(f"--- {time.strftime('%F %T')} ---\n{tb}\n")
        json.dump({"state": "FAILED", "pid": os.getpid(), "error": f"{type(e).__name__}: {e}",
                   "finished": time.time()}, open(status_path, "w"), indent=1)
        print(f"[job] FAILED {job_dir}: {e}", file=sys.stderr)
        return 1


def st_started(status_path):
    try:
        return json.load(open(status_path)).get("started")
    except Exception:  # noqa: BLE001
        return None


def _newest_ckpt(job_dir: str) -> str:
    ck = os.path.join(job_dir, "ckpt")
    if not os.path.isdir(ck):
        return ""
    files = [os.path.join(ck, f) for f in os.listdir(ck) if f.endswith((".pt", ".zip"))]
    return max(files, key=os.path.getmtime) if files else ""


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "shared":
        sys.exit(main(sys.argv[2], sys.argv[3], int(sys.argv[4]), "shared"))
    print("usage: python -m mujoco_baselines.run.job shared <algo> <task> <seed>")
    sys.exit(2)
