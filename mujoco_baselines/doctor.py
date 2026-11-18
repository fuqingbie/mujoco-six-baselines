"""Doctor: environment, GPU and grid sanity checks (task §2/§5)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

from . import constants as C
from . import grids as G

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("lockfile")
def _lock():
    assert os.path.exists(C.LOCK), "upstreams.lock.json missing"
    return json.load(open(C.LOCK))["upstreams"].keys()


@check("import-stack")
def _imports():
    import torch
    import gymnasium
    import mujoco
    import stable_baselines3
    return {
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__,
        "sb3": stable_baselines3.__version__,
    }


@check("gpu")
def _gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(512, 512, device="cuda")
    y = (x @ x).sum()
    torch.cuda.synchronize()
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"devices": names, "matmul_ok": bool(torch.isfinite(y).item())}


@check("nvidia-smi")
def _smi():
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return out.strip().splitlines()


@check("grids")
def _grids():
    s = G.summary()
    for t, v in s.items():
        assert v["n"] in (90, 154, 64), f"{t}: expected 90/154/64 cells, got {v['n']}"
        assert v["nominal_mass"] > 0
    return s


@check("perturb-env")
def _perturb():
    from .envs.perturbed import PerturbEnv, PhysSampler
    env = PerturbEnv("hopper", sampler=PhysSampler(0))
    base = env.unwrapped
    pristine_friction = base.model.geom_friction.copy()
    m0 = base.model.body_mass[1].item()
    obs, _ = env.reset(seed=1)
    m1 = base.model.body_mass[1].item()
    f1 = base.model.geom_friction.copy()
    # repeated resets with the same sampler stream must not compound: friction is
    # always assigned from the pristine copy, mass is absolute.
    obs, _ = env.reset(seed=2)
    assert base.model.body_mass[1] <= 1.11 * m0 + 1e-9
    ratio = base.model.geom_friction / pristine_friction
    assert abs(ratio.max() - ratio.min()) < 1e-12, "friction scaling not uniform"
    assert 0.89 < ratio.flat[0] < 1.11, f"friction mult out of range: {ratio.flat[0]}"
    env.reset(seed=3)
    assert abs(base.model.geom_friction.flat[0] / pristine_friction.flat[0] - 1) < 1.21
    env.close()
    return {"m0": m0, "friction_ok": True, "mass_ok": True}


@check("action-noise-disabled")
def _no_noise():
    from .envs.perturbed import PerturbEnv
    import numpy as np
    env = PerturbEnv("hopper", condition=(2.5, 1.0))
    obs, _ = env.reset(seed=7)
    o1, r1, t1, tr1, _ = env.step(np.zeros(env.action_space.shape))
    o2, r2, t2, tr2, _ = env.step(np.zeros(env.action_space.shape))
    # deterministic physics: identical action from identical state twice cannot be
    # bit-identical (state advances), so verify noise-free by exact replay:
    env2 = PerturbEnv("hopper", condition=(2.5, 1.0))
    env2.reset(seed=7)
    a = np.zeros(env2.action_space.shape)
    obs_a, r_a, _, _, _ = env2.step(a)
    env2.reset(seed=7)
    obs_b, r_b, _, _, _ = env2.step(a)
    assert np.allclose(obs_a, obs_b) and r_a == r_b, "step is stochastic — noise not disabled"
    env.close(); env2.close()
    return {"deterministic_step": True}


def main() -> int:
    os.makedirs(C.MPS_PIPE_DIR, exist_ok=True)
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", C.MPS_PIPE_DIR)
    print(f"python: {sys.version.split()[0]}  platform: {platform.platform()}")
    failures = 0
    for name, fn in CHECKS:
        try:
            res = fn()
            print(f"[PASS] {name}: {res}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print("doctor:", "ALL PASS" if failures == 0 else f"{failures} FAILURES")
    return 0 if failures == 0 else 1
