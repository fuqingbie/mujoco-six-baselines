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
    return None


def get_pids():
    jobs = pids_matching("run.job", "phase_a.")
    out = {}
    for pid, (stat, args) in jobs.items():
        if "run.job shared" in args or "phase_a." in args:
            out[pid] = (stat, args)
    return out


def main():
    while True:
        jobs = get_pids()
        active, stopped = [], []
        for pid, (stat, args) in jobs.items():
            pri = classify(args)
            if pri is None:
                continue
            rec = (pri, pid, args)
            if "T" in stat:  # stopped (T / TL)
                stopped.append(rec)
            else:
                active.append(rec)
        active.sort(key=lambda r: r[0])
        stopped.sort(key=lambda r: r[0])
        line = (f"{time.strftime('%F %T')} active={len(active)} "
                f"stopped={len(stopped)} "
                f"by_pri_active=" + str([sum(1 for r in active if r[0] == p) for p in (0, 1, 2)]))
        if len(active) < TARGET and stopped:
            # resume highest-priority stopped jobs up to target
            for pri, pid, _ in stopped[:TARGET - len(active)]:
                try:
                    os.kill(pid, 18)  # SIGCONT
                    line += f" cont:{pid}"
                except ProcessLookupError:
                    pass
        elif len(active) > TARGET + 4:
            # suspend lowest-priority active jobs
            for pri, pid, _ in sorted(active, key=lambda r: -r[0])[:len(active) - TARGET]:
                if pri == 0:  # never suspend P0
                    continue
                try:
                    os.kill(pid, 19)  # SIGSTOP
                    line += f" stop:{pid}"
                except ProcessLookupError:
                    pass
        with open(LOG, "a") as f:
            f.write(line + "\n")
        time.sleep(120)


if __name__ == "__main__":
    main()
