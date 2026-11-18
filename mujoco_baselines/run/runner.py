"""Queue runner: schedules jobs across the two H20s (task §5).

    python -m mujoco_baselines run --protocol shared --workers 2 --resume
    python -m mujoco_baselines run --protocol source --workers 6 --resume

Workers pull the next PENDING job from the registry (shared protocol: the
unified package in the mjb env; source protocol: per-lane drivers with their
own interpreters) and spawn it as a subprocess with a pinned GPU, the
MPS-bypass pipe dir, and bounded thread counts. Status lives in
runs/<protocol>/<algo>/<task>/seed<k>/status.json; RUNNING entries with dead
pids are re-queued (resume). Failed jobs are retried up to twice from their
checkpoints.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from ..constants import EXP_ROOT, LOGS_DIR, MPS_PIPE_DIR


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
    from .registry import build as build_jobs
    jobs = build_jobs(args.protocol)
    if args.only:
        filt = args.only.split(",")

        def keep(j):
            key = f"{j.algo}:{j.task}:{j.seed}"
            key2 = f"{j.algo}:{j.task}"
            return any(key.startswith(f.rstrip(':')) or key2 == f.rstrip(':')
                       for f in filt)
        jobs = [j for j in jobs if keep(j)]

    queue = []
    for j in jobs:
        st = job_state(j.dir)
        if st == "DONE":
            continue
        if st == "RUNNING":
            pid = json.load(open(os.path.join(j.dir, "status.json"))).get("pid")
            if pid and _pid_alive(pid):
                print(f"[runner] skip live job {j.id}")
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
        d = job.dir
        os.makedirs(d, exist_ok=True)
        gpu = gpus[worker_idx % len(gpus)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE_DIR
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["NUMEXPR_NUM_THREADS"] = "2"
        env["PYTHONPATH"] = EXP_ROOT
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("SPARC_CHECKOUT", None)
        cmd = list(job.cmd)
        out = open(os.path.join(d, "stdout.log"), "a")
        err = open(os.path.join(d, "stderr.log"), "a")
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, cwd=job.cwd)
        line = f"[runner] spawn worker{worker_idx} gpu{gpu} {job.id} pid {proc.pid}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        return proc, time.time()

    retries = {}
    while qi < len(queue) or running:
        for w in list(running):
            job, proc, t0 = running[w]
            rc = proc.poll()
            if rc is None and _zombie_reap(proc):
                # poll() can stall on zombie children (observed when jobs are
                # killed externally); reap directly and synthesise the rc
                rc = proc.returncode if proc.returncode is not None else -15
            if rc is None:
                continue
            proc.returncode = rc
            del running[w]
            st = job_state(job.dir)
            el = time.time() - t0
            line = f"[runner] exit {job.id} rc={proc.returncode} state={st} {el:.0f}s"
            print(line, flush=True)
            log.write(line + "\n")
            if proc.returncode != 0 and st != "DONE" and retries.get(job.id, 0) < 2:
                retries[job.id] = retries.get(job.id, 0) + 1
                queue.append(job)  # resume from checkpoint
                line = f"[runner] requeue {job.id} (retry {retries[job.id]})"
                print(line, flush=True)
                log.write(line + "\n")
        while len(running) < args.workers and qi < len(queue):
            job = queue[qi]
            qi += 1
            if job_state(job.dir) == "DONE":
                continue
            running[wi] = (job, *spawn(job, wi))
            wi = (wi + 1) % max(args.workers, 1)
        time.sleep(5)
    log.close()
    print("[runner] queue drained", flush=True)
    return 0


def _zombie_reap(proc) -> bool:
    """True if the child is a zombie; reaps it via waitpid directly."""
    import os as _os
    try:
        with open(f"/proc/{proc.pid}/status") as f:
            state = next(l for l in f if l.startswith("State:"))
    except (FileNotFoundError, StopIteration, ProcessLookupError):
        return False
    if "Z" not in state.split(":", 1)[1]:
        return False
    try:
        _pid, status = _os.waitpid(proc.pid, _os.WNOHANG)
        if _pid == proc.pid:
            proc.returncode = _os.waitstatus_to_exitcode(status)
    except ChildProcessError:
        return True
    return True
