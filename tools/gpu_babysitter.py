#!/usr/bin/env python3
"""GPU-node babysitter for the 22 model jobs kept on this node after the
four-machine handoff (round gpu-release-20261119-01).

The jobs run as orphan processes (their spawning runner was frozen for the
handoff). This babysitter:
  * every 120s verifies each job's process is alive
  * respawns dead jobs via `mujoco_baselines.run.job` (which resumes from the
    job's newest checkpoint; HANDOFF/DONE jobs are never touched)
  * never spawns anything else; the watchdog still handles freeze/cont and
    memory guarding.
"""
import json
import os
import re
import subprocess
import time

ROOT = "/root/mjbexp"
GPU_JOBS = [
    # 11 mbpo kept on GPU
    ("mbpo", "hopper", s) for s in range(5)
] + [
    ("mbpo", "halfcheetah", s) for s in range(5)
] + [
    ("mbpo", "walker2d", 4),
] + [
    # 11 rmbpo kept on GPU
    ("rmbpo", "hopper", s) for s in (0, 3, 4)
] + [
    ("rmbpo", "halfcheetah", s) for s in (2, 3, 4)
] + [
    ("rmbpo", "walker2d", s) for s in range(5)
]
LOG = os.path.join(ROOT, "logs/gpu_babysitter.log")


def live_pids():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines()[1:]:
        m = re.search(r"run\.job shared (\S+) (\S+) (\d+)", line)
        if m:
            pids.add((m.group(1), m.group(2), int(m.group(3))))
    return pids


def state_of(algo, task, seed):
    try:
        return json.load(open(f"{ROOT}/runs/shared/{algo}/{task}/seed{seed}/status.json")).get("state")
    except FileNotFoundError:
        return "PENDING"


def spawn(algo, task, seed):
    d = f"{ROOT}/runs/shared/{algo}/{task}/seed{seed}"
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_MPS_PIPE_DIRECTORY="/tmp/mjb-mps-pipe",
                PYTHONPATH=ROOT, OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    cmd = ["/root/mjbexp/envs/mjb/bin/python", "-m", "mujoco_baselines.run.job",
           "shared", algo, task, str(seed)]
    out = open(os.path.join(d, "stdout.log"), "a")
    err = open(os.path.join(d, "stderr.log"), "a")
    proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, cwd=ROOT,
                            start_new_session=True)
    return proc.pid


def all_shared_done():
    """All 22 GPU-owned model jobs DONE (HANDOFF/DONE both count as settled;
    HANDOFF jobs were handed to CPU nodes and are not ours to wait for)."""
    for (algo, task, seed) in GPU_JOBS:
        if state_of(algo, task, seed) != "DONE":
            return False
    return True


def launch_source_runner():
    """shared -> source transition (task: shared first, then source): start the
    controlled source-protocol queue once our shared jobs are settled. The
    runner is HANDOFF-aware and will never re-claim CPU-owned jobs."""
    env = dict(os.environ)
    env.update(CUDA_MPS_PIPE_DIRECTORY="/tmp/mjb-mps-pipe",
                PYTHONPATH=ROOT, OMP_NUM_THREADS="2")
    cmd = ["/root/mjbexp/envs/mjb/bin/python", "-m", "mujoco_baselines",
           "run", "--protocol", "source", "--workers", "20", "--resume"]
    out = open(os.path.join(ROOT, "logs/runner_source_5.log"), "a")
    err = open(os.path.join(ROOT, "logs/runner_source_5.log"), "a")
    proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, cwd=ROOT,
                            start_new_session=True)
    return proc.pid


def main():
    source_started = False
    with open(LOG, "a") as log:
        log.write(f"[babysitter] start {time.strftime('%F %T')} watching {len(GPU_JOBS)} jobs\n")
    while True:
        live = live_pids()
        events = []
        for (algo, task, seed) in GPU_JOBS:
            key = (algo, task, seed)
            st = state_of(algo, task, seed)
            if st in ("DONE", "HANDOFF"):
                continue
            if key not in live:
                pid = spawn(algo, task, seed)
                events.append(f"respawn {algo}/{task}/s{seed} pid {pid}")
        if not source_started and all_shared_done():
            pid = launch_source_runner()
            events.append(f"SOURCE_RUNNER started pid {pid}")
            source_started = True
        line = f"{time.strftime('%F %T')} live={len(live)} source={'on' if source_started else 'off'} " + ("; ".join(events) if events else "")
        with open(LOG, "a") as log:
            log.write(line + "\n")
        time.sleep(120)


if __name__ == "__main__":
    main()
