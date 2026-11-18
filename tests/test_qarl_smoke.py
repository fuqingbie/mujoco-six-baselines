"""Smoke test for mujoco_baselines/algos/qarl.py (QARL port).

Run (from /root/mjbexp):
    PYTHONPATH=/root/mjbexp /root/mjbexp/envs/mjb/bin/python \
        tests/test_qarl_smoke.py

Covers:
  A. 3000-step hopper/seed-0 run with the production constants (upstream
     warmup 3000/5000 -> no gradient updates inside 3000 steps, but the
     teacher curriculum, counters, progress log, final checkpoint, eval
     adapter and idempotent re-entry must all work).
  B. 3000-step run with smoke-only warmup overrides
     (initial_replay_size=100, warmup_transitions=300) so that BOTH players
     actually take gradient updates: protagonist actor+alpha updates,
     adversary actor updates with alpha set per batch from the state
     temperature (and never gradient-updated), teacher concentration moves.
  C. resume: train to 1500 steps, then re-enter with steps=3000 (budget
     raised) -> resumes from ckpt/state.pt and lands exactly on 3000.
  D. frozen evaluator: load_eval_adapter + evaluate_checkpoint on two grid
     conditions with eval seeds [20000, 20001] + save_records.

No pytest test_* functions are defined on purpose (each run takes minutes);
this is a standalone script that exits non-zero on failure.
"""
import json
import os
import sys

# must be set before torch initialises CUDA
os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mjb-no-mps")
os.makedirs("/tmp/mjb-no-mps", exist_ok=True)

if "/root/mjbexp" not in sys.path:
    sys.path.insert(0, "/root/mjbexp")

import tempfile

import numpy as np

from mujoco_baselines import config
from mujoco_baselines import grids as G
from mujoco_baselines.algos import qarl

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}", flush=True)
    if not cond:
        FAILURES.append(name)


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def _progress_lines(job):
    p = os.path.join(job, "progress.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def train_job(tmp, tag, task, seed, steps, extra=None):
    cfg = dict(config.QARL)
    cfg["steps"] = steps
    if extra:
        cfg.update(extra)
    job = os.path.join(tmp, f"{tag}_{task}_seed{seed}")
    qarl.train(job, task, seed, cfg)
    return job, cfg


def run_A(tmp):
    print("== A: 3000 steps, production constants ==", flush=True)
    job, cfg = train_job(tmp, "A", "hopper", 0, 3000)
    c = _load_json(os.path.join(job, "counters.json"))
    check("A counters real_transitions == 3000",
          c["real_transitions"] == 3000, str(c))
    check("A protagonist_steps == adversary_steps == 3000",
          c["protagonist_steps"] == 3000 and c["adversary_steps"] == 3000)
    check("A temperature_samples multiple of 30",
          c["temperature_samples"] > 0 and c["temperature_samples"] % 30 == 0,
          str(c["temperature_samples"]))
    check("A final.pt exists",
          os.path.exists(os.path.join(job, "ckpt", "final.pt")))
    lines = _progress_lines(job)
    check("A progress.jsonl non-empty", len(lines) >= 1)
    check("A last progress step == 3000", lines[-1]["step"] == 3000)
    check("A progress returns logged on final line",
          lines[-1]["mean_return_vs_adv"] is not None
          and lines[-1]["mean_return_no_adv"] is not None)
    check("A progress mean_temp finite",
          np.isfinite(lines[-1]["mean_temp"]),
          f"mean_temp={lines[-1]['mean_temp']:.5f}")
    ck = qarl.load_eval_adapter  # noqa: F841 (exercise import path)
    return job, cfg, c


def run_A_idempotent(job, cfg, c_before, lines_before):
    print("== A2: idempotent re-entry (budget reached) ==", flush=True)
    qarl.train(job, "hopper", 0, cfg)
    c_after = _load_json(os.path.join(job, "counters.json"))
    lines_after = _progress_lines(job)
    check("A2 counters unchanged", c_after == c_before)
    check("A2 progress lines unchanged", len(lines_after) == len(lines_before))


def run_B(tmp):
    print("== B: 3000 steps, smoke warmup overrides ==", flush=True)
    job, cfg = train_job(tmp, "B", "hopper", 0, 3000,
                         {"initial_replay_size": 100,
                          "warmup_transitions": 300})
    c = _load_json(os.path.join(job, "counters.json"))
    check("B counters real_transitions == 3000", c["real_transitions"] == 3000)
    ckpt = qarl.torch.load(os.path.join(job, "ckpt", "final.pt"),
                           map_location="cpu", weights_only=False)
    pu = ckpt["prot"]["update_counters"]
    au = ckpt["adv"]["update_counters"]
    check("B protagonist actor updated", pu["actor"] > 0, str(pu))
    check("B protagonist alpha updated", pu["alpha"] > 0)
    check("B protagonist critic updated", pu["critic"] > 0)
    check("B adversary actor updated", au["actor"] > 0, str(au))
    check("B adversary critic updated", au["critic"] > 0)
    check("B adversary alpha set from temperature (never grad-updated)",
          au["alpha_from_temp"] > 0 and au["alpha"] == 0, str(au))
    tw = ckpt["teacher"]["weights"]
    check("B teacher concentration moved from initial 50.0",
          abs(tw[0] - 50.0) > 1e-6, f"conc={tw[0]:.4f}")
    means = ckpt["sampled_temp_means"]
    check("B sampled temp means recorded", len(means) >= 1, str(means))
    conc_hist = ckpt["teacher"]["conc_data"]
    check("B teacher conc history shows update",
          len(conc_hist) > 1 and any(abs(x - 50.0) > 1e-9 for x in conc_hist),
          str(conc_hist))
    check("B temperature changes over iterations",
          len(means) >= 2 or len(conc_hist) > 1)
    return job, cfg


def run_C(tmp):
    print("== C: resume (1500 -> 3000 steps) ==", flush=True)
    job, cfg = train_job(tmp, "C", "hopper", 0, 1500,
                         {"initial_replay_size": 100,
                          "warmup_transitions": 300})
    c1 = _load_json(os.path.join(job, "counters.json"))
    check("C first phase stops at 1500", c1["real_transitions"] == 1500)
    n_lines_1 = len(_progress_lines(job))
    cfg2 = dict(cfg)
    cfg2["steps"] = 3000
    qarl.train(job, "hopper", 0, cfg2)  # budget raised -> resume from state.pt
    c2 = _load_json(os.path.join(job, "counters.json"))
    check("C resumed run lands exactly on 3000",
          c2["real_transitions"] == 3000, str(c2))
    lines = _progress_lines(job)
    check("C progress continues after resume", len(lines) > n_lines_1)
    check("C last progress step == 3000", lines[-1]["step"] == 3000)
    return job


def run_D(job, cfg):
    print("== D: frozen evaluator (2 grid conditions, 2 seeds) ==", flush=True)
    adapter = qarl.load_eval_adapter(job, "hopper", 0, cfg)
    conds = G.grid("hopper")[:2]
    check("D grid conditions built", len(conds) == 2,
          str([c["condition_str"] for c in conds]))
    from mujoco_baselines.eval.evaluator import evaluate_checkpoint
    from mujoco_baselines.eval.records import save_records
    records, wall = evaluate_checkpoint(adapter, "hopper", conditions=conds,
                                        seeds=[20000, 20001])
    check("D 4 episode records (2 conditions x 2 seeds)", len(records) == 4)
    rets = [r["return"] for r in records]
    check("D returns finite", all(np.isfinite(r) for r in rets), str(rets))
    lens = [r["length"] for r in records]
    check("D lengths sane (0 < len <= 1000)",
          all(0 < l <= 1000 for l in lens), str(lens))
    # adapter protocol details
    obs = np.zeros((3, 11), dtype=np.float32)
    act = adapter.act(obs, np.ones(3, dtype=bool))
    check("D adapter act shape", act.shape == (3, 3), str(act.shape))
    check("D adapter act within action box",
          np.all(act >= -1.0) and np.all(act <= 1.0))
    adapter.reset_episodes(np.ones(3, dtype=bool))
    path = save_records(records, "smoke", "qarl", "hopper", 0, "final",
                        wall_seconds=wall)
    check("D save_records wrote parquet", os.path.exists(path), path)
    return adapter


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="qarl_smoke_")
    print(f"smoke scratch dir: {tmp}", flush=True)
    jobA, cfgA, cA = run_A(tmp)
    run_A_idempotent(jobA, cfgA, cA, _progress_lines(jobA))
    run_B(tmp)
    run_C(tmp)
    # frozen evaluator on the production-constants run (A)
    run_D(jobA, cfgA)
    print(flush=True)
    if FAILURES:
        print(f"SMOKE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
