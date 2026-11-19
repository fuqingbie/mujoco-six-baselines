"""Unified phase-B environments (task §4.1).

Construction recipe, frozen here:
  * base envs: vanilla gymnasium 0.29.2 MuJoCo v4 (same observation/action/reward/
    termination semantics as the robustness-gymnasium fork's Perturbed envs, whose
    step() differs only by U(-5e-3, 5e-3) action noise — disabled in phase B per
    task §4.1 "各环境禁用训练施力对手与额外动作噪声").
  * perturbations replicate the official implementation exactly
    (rmbpo-tmlr/embodied/robustness-gymnasium .../hopper_v4-perturbed.py reset()):
        model.geom_friction[:, :] = pristine_friction * (1 + friction_p)
        model.body_mass[1] = absolute torso mass
    Pristine copies are captured once at __init__, so repeated resets never
    compound (verified by tests/test_perturb_env.py).
  * per-episode parameters are drawn by the shared PhysSampler (m/m0 ~ U[0.9,1.1],
    friction multiplier ~ U[0.9,1.1]) for training, or fixed values for evaluation.
  * episode cap 1000 via the standard gymnasium TimeLimit (v4 spec).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from ..constants import EPISODE_CAP, MASS_MULT_RANGE, FRICTION_MULT_RANGE

PERTURBED_IDS = {
    "hopper": "Hopper-v4",
    "halfcheetah": "HalfCheetah-v4",
    "walker2d": "Walker2d-v4",
}


class PhysSampler:
    """The one shared per-episode parameter sampler for all six methods (task §4.1)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        mm = self.rng.uniform(*MASS_MULT_RANGE)
        fm = self.rng.uniform(*FRICTION_MULT_RANGE)
        return float(mm), float(fm)


class PerturbEnv(gym.Wrapper):
    """v4 env + reset-time torso-mass / all-geom-friction perturbation.

    Parameters hold for the whole episode. `sampler` (training) draws both
    multipliers per reset; `condition=(mass_kg, friction_multiplier)` (frozen
    evaluation) fixes them. `context_out=True` records the episode context on
    the unwrapped env as `current_context = (mass_mult, friction_mult)`.
    """

    def __init__(self, task: str, sampler: PhysSampler | None = None,
                 condition: tuple | None = None, context_out: bool = False):
        assert sampler is None or condition is None
        self.task = task
        env = gym.make(PERTURBED_IDS[task], disable_env_checker=True)  # TimeLimit(1000)
        super().__init__(env)
        base = env.unwrapped
        model = base.model
        self._model = model
        self._pristine_friction = np.copy(model.geom_friction)
        self._m0 = float(model.body_mass[1])  # nominal torso mass from raw env
        self._sampler = sampler
        self._condition = condition
        self._context_out = context_out
        self.current_context = None
        self.current_params = None  # (mass_kg, friction_multiplier)

    def _apply(self, mass_kg: float, fm: float):
        self._model.geom_friction[:, :] = self._pristine_friction * fm
        self._model.body_mass[1] = float(mass_kg)
        self.current_params = (float(mass_kg), float(fm))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self._condition is not None:
            mass_kg, fm = self._condition
        else:
            mm, fm = self._sampler.sample()
            mass_kg = self._m0 * mm
            if self._context_out:
                self.current_context = (mm, fm)
        self._apply(mass_kg, fm)
        return obs, info

    @property
    def nominal_torso_mass(self) -> float:
        return self._m0


def make_train_env(task: str, seed: int, context_out: bool = False) -> PerturbEnv:
    """Single training env with the shared sampler (seeded per job/env)."""
    return PerturbEnv(task, sampler=PhysSampler(seed), context_out=context_out)


def make_train_venv(task: str, seed: int, n_envs: int, context_out: bool = False):
    """Vectorized training envs, one PhysSampler each (seed + i per sub-env)."""
    return gym.vector.SyncVectorEnv(
        [lambda i=i: make_train_env(task, seed + i, context_out=context_out)
         for i in range(n_envs)]
    )


def make_eval_env(task: str, condition: tuple, seed: int) -> PerturbEnv:
    """Frozen-evaluation env: fixed condition, deterministic step, pre-seeded RNG.

    The generator is seeded without drawing, so the first real reset (seed=None)
    reproduces exactly the state stream of reset(seed=seed) — identical initial
    states across conditions for the same reset seed.
    """
    env = PerturbEnv(task, condition=condition)
    base = env.env.unwrapped
    base._np_random, base._np_random_seed = gym.utils.seeding.np_random(seed)
    return env


def make_eval_venv(task: str, condition: tuple, seeds: list[int]):
    return gym.vector.SyncVectorEnv(
        [lambda s=s: make_eval_env(task, condition, seed=s) for s in seeds]
    )
