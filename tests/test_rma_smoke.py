"""Smoke test for the RMA two-phase port (algos/rma.py).

Short two-phase run (4k + 4k steps, learning_starts=1000) on hopper seed 0,
then frozen-adapter evaluation through the unified evaluator (2 conditions x
2 reset seeds), plus the history-reset determinism check (the same condition
evaluated twice on the same adapter must give identical episode records).

Run:  envs/mjb/bin/python -m pytest tests/test_rma_smoke.py -x -q
  or: envs/mjb/bin/python tests/test_rma_smoke.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mujoco_baselines import config, grids as G          # noqa: E402
from mujoco_baselines.algos import rma                   # noqa: E402
from mujoco_baselines.eval import evaluator              # noqa: E402

SMOKE_CFG = dict(config.RMA)
SMOKE_CFG.update(
    phase1_steps=4000,
    phase2_steps=4000,
    learning_starts=1000,
    save_interval=2500,   # exercise the mid-phase checkpoint path
    log_freq=100,
)


def _probe_mse(job_dir, task, seed, cfg, steps=400):
    """Regression probe: MSE(trained adapter) vs MSE(fresh random adapter).

    Collects (history, context) pairs from a seeded rollout (the exact
    history convention of phase 2), computes the expert's context latent,
    and compares the final adapter against a freshly initialized one.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mujoco_baselines.algos.rma import AdapterPolicy, ExpertPolicy
    from mujoco_baselines.envs.perturbed import make_train_env

    ec = torch.load(os.path.join(job_dir, "ckpt", "expert.pt"),
                    map_location="cpu", weights_only=False)
    fc = torch.load(os.path.join(job_dir, "ckpt", "final.pt"),
                    map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = ExpertPolicy(ec["obs_dim"], ec["context_dim"], ec["act_dim"],
                          ec["action_low"], ec["action_high"]).to(device)
    expert.load_state_dict(ec["state_dict"])

    H = int(fc["history_length"])
    D = int(fc["obs_dim"]) + int(fc["act_dim"])
    env = make_train_env(task, seed, context_out=True)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    ctx = np.asarray(env.current_context, dtype=np.float32)
    hist = np.zeros((H, D), dtype=np.float32)
    hists, ctxs = [], []
    for _ in range(steps):
        hists.append(hist.copy())
        ctxs.append(ctx.copy())
        a = env.action_space.sample().astype(np.float32)
        nobs, _, term, trunc, _ = env.step(a)
        nobs = np.asarray(nobs, dtype=np.float32)
        hist[:-1] = hist[1:]
        hist[-1] = np.concatenate([nobs, a])
        if term or trunc:
            obs, _ = env.reset()
            ctx = np.asarray(env.current_context, dtype=np.float32)
            hist[:] = 0.0
    env.close()
    hists = torch.tensor(np.asarray(hists), dtype=torch.float32, device=device)
    ctxs = torch.tensor(np.asarray(ctxs), dtype=torch.float32, device=device)
    with torch.no_grad():
        z = expert.encoder(ctxs)

    def mse_of(policy):
        with torch.no_grad():
            z_hat = policy.adapter(hists)
        return float(F.mse_loss(z_hat, z).item())

    trained = AdapterPolicy(fc["obs_dim"], fc["act_dim"], H,
                            fc["action_low"], fc["action_high"]).to(device)
    trained.load_state_dict(fc["state_dict"])
    torch.manual_seed(seed)
    fresh = AdapterPolicy(fc["obs_dim"], fc["act_dim"], H,
                          fc["action_low"], fc["action_high"]).to(device)
    return mse_of(trained), mse_of(fresh)


def _read_progress(job_dir):
    entries = []
    with open(os.path.join(job_dir, "progress.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_smoke(job_dir=None, task="hopper", seed=0, cfg=None):
    cfg = dict(SMOKE_CFG if cfg is None else cfg)
    if job_dir is None:
        job_dir = tempfile.mkdtemp(prefix="rma_smoke_")
    t0 = time.time()

    # ---- phase 1 + phase 2 ----
    rma.train(job_dir, task, seed, cfg)
    wall = time.time() - t0

    P = dict(
        expert=os.path.join(job_dir, "ckpt", "expert.pt"),
        final=os.path.join(job_dir, "ckpt", "final.pt"),
        counters=os.path.join(job_dir, "counters.json"),
        progress=os.path.join(job_dir, "progress.jsonl"),
    )
    assert os.path.exists(P["expert"]), "phase 1 must save ckpt/expert.pt"
    assert os.path.exists(P["final"]), "phase 2 must save ckpt/final.pt"
    counters = json.load(open(P["counters"]))
    expected = cfg["phase1_steps"] + cfg["phase2_steps"]
    assert counters == {"phase1_steps": cfg["phase1_steps"],
                        "phase2_steps": cfg["phase2_steps"],
                        "real_transitions": expected}, counters

    # ---- progress.jsonl sanity ----
    entries = _read_progress(job_dir)
    p1 = [e for e in entries if e.get("phase") == 1 and "loss" in e]
    p2 = [e for e in entries if e.get("phase") == 2 and "loss" in e]
    assert p1, "phase 1 loss entries missing"
    assert p2, "phase 2 adapter-loss entries missing"

    # ---- phase-2 regression actually learned: trained << fresh adapter ----
    mse_trained, mse_fresh = _probe_mse(job_dir, task, seed, cfg)
    assert mse_trained < 0.5 * mse_fresh, \
        f"adapter MSE should drop well below a fresh adapter: " \
        f"trained={mse_trained} fresh={mse_fresh}"

    # ---- idempotent re-entry ----
    n_lines = len(entries)
    rma.train(job_dir, task, seed, cfg)
    assert len(_read_progress(job_dir)) == n_lines, \
        "completed job must be a no-op on re-entry"

    # ---- frozen adapter through the unified evaluator ----
    adapter = rma.load_eval_adapter(job_dir, task, seed, cfg)
    conds = G.grid(task, with_nominal=True)
    conds = [conds[0], conds[-1]]           # one OOD grid cell + nominal
    seeds = (20000, 20001)
    records, eval_wall = evaluator.evaluate_checkpoint(
        adapter, task, conditions=conds, seeds=seeds)
    assert len(records) == len(conds) * len(seeds) == 4
    rets = [r["return"] for r in records]
    assert all(isinstance(r, float) for r in rets) and rets

    # ---- history-reset determinism: same adapter, same condition, twice ----
    r1 = evaluator.evaluate_condition(adapter, task, conds[0], seeds)
    r2 = evaluator.evaluate_condition(adapter, task, conds[0], seeds)
    assert r1 == r2, "evaluate_condition must be reproducible on the same " \
                     "adapter (history reset semantics)"

    sps1 = [e["sps"] for e in p1 if e.get("sps")]
    sps2 = [e["sps"] for e in p2 if e.get("sps")]
    p2_losses = [e["loss"] for e in p2]
    out = {
        "job_dir": job_dir,
        "counters": counters,
        "phase1_sps": sps1[-1] if sps1 else None,
        "phase2_sps": sps2[-1] if sps2 else None,
        "train_wall_s": round(wall, 1),
        "eval_wall_s": round(eval_wall, 1),
        "eval_returns": {c["condition_str"]: [
            r["return"] for r in records if r["condition_id"] == c["condition_id"]
        ] for c in conds},
        "adapter_mse_fresh": round(mse_fresh, 6),
        "adapter_mse_trained": round(mse_trained, 6),
        "adapter_mse_first_logged": p2_losses[0] if p2_losses else None,
        "adapter_mse_last_logged": p2_losses[-1] if p2_losses else None,
    }
    return out


def test_rma_smoke():
    out = run_smoke()
    print(json.dumps(out, indent=1))
    shutil.rmtree(out["job_dir"], ignore_errors=True)


if __name__ == "__main__":
    out = run_smoke(job_dir=os.path.join(ROOT, "runs", "smoke", "rma",
                                         "hopper", "seed0"))
    print(json.dumps(out, indent=1))
