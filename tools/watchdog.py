#!/usr/bin/env python3
"""Resource watchdog: keeps total active jobs within the 94-core cgroup quota.

Priority order (highest first):
  P0 phase-B non-model (shared rma/qarl/rarl/dr)
  P1 phase-B model (shared mbpo/rmbpo)  [currently SIGSTOPped]
  P2 phase-A source lanes               [currently SIGSTOPped]

Every 120s: count active (state R/S, not T) job processes; if below TARGET,
SIGCONT the highest-priority stopped ones; if above TARGET+4, SIGSTOP the
lowest-priority active ones. Logs to logs/watchdog.log. PIDs are discovered
fresh each cycle via ps + /proc/<pid>/cmdline matching, so restarted jobs are
picked up automatically.
"""
import os
import subprocess
import time

ROOT = "/root/mjbexp"
LOG = os.path.join(ROOT, "logs/watchdog.log")
TARGET = 44


def pids_matching(*needles):
    out = subprocess.run(["ps", "-eo", "pid,stat,args"], capture_output=True,
                         text=True).stdout
    pids = {}
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, stat, args = int(parts[0]), parts[1], parts[2]
        if "grep" in args or "watchdog" in args:
            continue
        for n in needles:
            if n in args:
                pids[pid] = (stat, args)
                break
    return pids


def classify(pid_args):
    args = pid_args
    if "run.job shared" in args:
        for a in ("mbpo", "rmbpo"):
            if f"shared {a}" in args:
                return 1
        return 0
    if "phase_a." in args:
        return 2
    return 99  # child process (train.py / main_eval): inherits tree priority


JOB_MARKERS = ("run.job", "phase_a.", "train.py --alg", "evaluate/main_eval",
               "sparc/train.py")


def _ppid(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for l in f:
                if l.startswith("PPid:"):
                    return int(l.split()[1])
    except FileNotFoundError:
        pass
    return 0


RUNNER_MARK = ("mujoco_baselines run", "runner.py", "watchdog", "install_")


def _args_of(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return ""


def _top_ancestor(pid):
    """Walk up until the runner/nohup shell; the last job-side pid is the tree
    root (so each run.job / phase_a driver is its own tree)."""
    seen = set()
    cur = pid
    root = pid
    while cur not in seen:
        seen.add(cur)
        pp = _ppid(cur)
        if pp <= 1:
            return root
        pa = _args_of(pp)
        if any(m in pa for m in RUNNER_MARK):
            return root
        root = pp
        cur = pp
    return root


def get_pids():
    """All job-ish processes (incl. sparc train children and eval subprocesses)."""
    out = subprocess.run(["ps", "-eo", "pid,stat,args"], capture_output=True,
                         text=True).stdout
    found = {}
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, stat, args = int(parts[0]), parts[1], parts[2]
        if "grep" in args or "watchdog" in args or "runner" in args:
            continue
        if any(m in args for m in JOB_MARKERS):
            found[pid] = (stat, args)
    return found


def main():
    while True:
        jobs = get_pids()
        trees = {}  # top pid -> dict(active=[...], stopped=[...], pri)
        for pid, (stat, args) in jobs.items():
            top = _top_ancestor(pid)
            t = trees.setdefault(top, {"active": [], "stopped": [], "pri": 99})
            rec = (classify(args), pid, args)
            if "T" in stat:
                t["stopped"].append(rec)
            else:
                t["active"].append(rec)
        for t in trees.values():
            pris = [r[0] for r in t["active"] + t["stopped"]]
            t["pri"] = min(pris) if pris else 99
        active_trees = [t for t in trees.values() if t["active"]]
        stopped_trees = [t for t in trees.values() if not t["active"] and t["stopped"]]
        active_trees.sort(key=lambda t: t["pri"])
        stopped_trees.sort(key=lambda t: t["pri"])
        line = (f"{time.strftime('%F %T')} active_trees={len(active_trees)} "
                f"stopped_trees={len(stopped_trees)} pri_active="
                + str([sum(1 for t in active_trees if t["pri"] == p) for p in (0, 1, 2)]))
        if len(active_trees) < TARGET and stopped_trees:
            for t in stopped_trees[:TARGET - len(active_trees)]:
                for pri, pid, _ in t["stopped"]:
                    try:
                        os.kill(pid, 18)
                        line += f" cont:{pid}"
                    except ProcessLookupError:
                        pass
        elif len(active_trees) > TARGET + 4:
            for t in sorted(active_trees, key=lambda t: -t["pri"]):
                if len(active_trees) <= TARGET:
                    break
                if t["pri"] == 0:
                    continue
                for pri, pid, _ in t["active"]:
                    try:
                        os.kill(pid, 19)
                        line += f" stop:{pid}"
                    except ProcessLookupError:
                        pass
                active_trees.remove(t)
        with open(LOG, "a") as f:
            f.write(line + "\n")
        time.sleep(120)


if __name__ == "__main__":
    main()
