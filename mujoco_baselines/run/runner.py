"""Queue runner: schedules jobs across the two H20s (task §5).

    python -m mujoco_baselines run --protocol shared --workers 2 --resume

Workers pull the next PENDING job and spawn
    python -m mujoco_baselines.run.job shared <algo> <task> <seed>
as a subprocess with a pinned GPU, the MPS-bypass pipe dir, and bounded
thread counts. Status lives in runs/<protocol>/<algo>/<task>/seed<k>/
status.json; RUNNING entries with dead pids are re-queued (resume).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from ..constants import (ALGOS, LOGS_DIR, MPS_PIPE_DIR, RUNS_DIR, TASKS,
                         TRAIN_SEEDS)

# longest-first so the heavy families start early; hopper seed 0 first overall
# (task §5.3: first finish Hopper's first formal seed, other card runs the rest)
ALGO_ORDER = ("mbpo", "rmbpo", "rma", "qarl", "rarl_ppo", "dr_ppo")
TASK_ORDER = {"hopper": 0, "halfcheetah": 1, "walker2d": 2}


def job_dir(algo, task, seed, protocol="shared"):
    return os.path.join(RUNS_DIR, protocol, algo, task, f"seed{seed}")


def job_state(d):
    try:
        return json.load(open(os.path.join(d, "status.json"))).get("state", "PENDING")
    except Exception:  # noqa: BLE001
        return "PENDING"


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def build_jobs(protocol="shared"):
    jobs = []
    for algo in ALGO_ORDER:
        for task in TASKS:
            for seed in TRAIN_SEEDS:
                jobs.append((algo, task, seed))
    if protocol != "shared":
        raise NotImplementedError(f"protocol {protocol} lanes are defined in phase_a.py")
    # hopper seed 0 of every algo first, then longest-family-first
    jobs.sort(key=lambda j: (0 if (j[1] == "hopper" and j[2] == 0) else 1,
                             ALGO_ORDER.index(j[0]), TASK_ORDER[j[1]], j[2]))
    return jobs


def main(argv) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="shared", choices=["shared", "source"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", default=None,
                   help="comma list of algo[:task[:seed]] filters")
    args = p.parse_args(argv)

    os.makedirs(MPS_PIPE_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = build_jobs(args.protocol)
    if args.only:
        filt = args.only.split(",")
        def keep(j):
            key = f"{j[0]}:{j[1]}:{j[2]}"
            key2 = f"{j[0]}:{j[1]}"
            return any(key.startswith(f.rstrip(':')) or key2 == f.rstrip(':')
                       for f in filt)
        jobs = [j for j in jobs if keep(j)]

    queue = []
    for j in jobs:
        d = job_dir(*j, protocol=args.protocol)
        st = job_state(d)
        if st == "DONE":
            continue
        if st == "RUNNING":
            pid = json.load(open(os.path.join(d, "status.json"))).get("pid")
            if pid and _pid_alive(pid):
                print(f"[runner] skip live job {j}")
                continue
        queue.append(j)
    print(f"[runner] protocol={args.protocol} jobs pending: {len(queue)} "
          f"(of {len(jobs)}), workers={args.workers}, gpus={gpus}", flush=True)
    if not queue:
        return 0

    running = {}  # worker_idx -> (job, proc, t0)
    qi = 0
    wi = 0
    log = open(os.path.join(LOGS_DIR, f"runner_{args.protocol}.log"), "a")

    def spawn(job, worker_idx):
        algo, task, seed = job
        d = job_dir(*job, protocol=args.protocol)
        os.makedirs(d, exist_ok=True)
        gpu = gpus[worker_idx % len(gpus)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE_DIR
        env["OMP_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        env["OPENBLAS_NUM_THREADS"] = "8"
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))  # exp root
        cmd = [sys.executable, "-m", "mujoco_baselines.run.job",
               args.protocol, algo, task, str(seed)]
        out = open(os.path.join(d, "stdout.log"), "a")
        err = open(os.path.join(d, "stderr.log"), "a")
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, cwd=env["PYTHONPATH"])
        line = f"[runner] spawn worker{worker_idx} gpu{gpu} {job} pid {proc.pid}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        return proc, time.time()

    retries = {}
    while qi < len(queue) or running:
        # reap
        for w in list(running):
            job, proc, t0 = running[w]
            if proc.poll() is None:
                continue
            del running[w]
            algo, task, seed = job
            d = job_dir(*job, protocol=args.protocol)
            st = job_state(d)
            el = time.time() - t0
            line = f"[runner] exit {job} rc={proc.returncode} state={st} {el:.0f}s"
            print(line, flush=True)
            log.write(line + "\n")
            if proc.returncode != 0 and st != "DONE" and retries.get(job, 0) < 2:
                retries[job] = retries.get(job, 0) + 1
                queue.append(job)  # resume from checkpoint
                line = f"[runner] requeue {job} (retry {retries[job]})"
                print(line, flush=True)
                log.write(line + "\n")
        # fill
        while len(running) < args.workers and qi < len(queue):
            job = queue[qi]
            qi += 1
            if job_state(job_dir(*job, protocol=args.protocol)) == "DONE":
                continue
            running[wi] = (job, *spawn(job, wi))
            wi = (wi + 1) % max(args.workers, 1)
        time.sleep(5)
    log.close()
    print("[runner] queue drained", flush=True)
    return 0
