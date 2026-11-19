"""Body-force adversary interface shared by RARL-PPO and QARL (task §4.1).

Vendored from robustrllib@ a665ddf baselines/rarl/game.py (BodyForce), semantics
kept identical: before every step the adversary action (unit box) is scaled by
`force` and written to ``data.xfrc_applied[body, axis]`` (world-frame axes),
all other entries zeroed. Phase B fixes force = 1.0 N on world x and z.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

AXES = {"x": 0, "y": 1, "z": 2}


class BodyForce:
    def __init__(self, env, bodies, axes, force):
        base = env.unwrapped
        self._sim = base
        self.ids = [int(self._sim.model.body(name).id) for name in bodies]
        self.axes = [AXES[a] for a in axes]
        self.force = float(force)
        self.dim = len(self.ids) * len(self.axes)
        self.space = gym.spaces.Box(-1.0, 1.0, (self.dim,), dtype=np.float32)

    def reset(self):
        self._sim.data.xfrc_applied[:] = 0.0

    def apply(self, adv):
        """Write the adversary force; returns nothing (env.step(ctrl) follows)."""
        u = self.force * np.clip(np.asarray(adv, dtype=np.float64).reshape(-1), -1.0, 1.0)
        xfrc = self._sim.data.xfrc_applied
        xfrc[:] = 0.0
        k = 0
        for body in self.ids:
            for axis in self.axes:
                xfrc[body, axis] = u[k]
                k += 1

    def clear(self):
        self._sim.data.xfrc_applied[:] = 0.0


class TwoPlayerForceEnv(gym.Wrapper):
    """Single-env two-player game view for QARL-style joint stepping.

    One simulation step per (protagonist action, adversary action) pair: both
    players act in the same transition, which therefore counts once (task §4.2).
    Observation/reward are the protagonist's. The adversary reward is the
    negation, available via ``info["adversary_reward"]``.
    """

    def __init__(self, env, bodies, axes, force):
        super().__init__(env)
        self.adversary = BodyForce(env, bodies, axes, force)

    def reset(self, **kwargs):
        self.adversary.reset()
        return self.env.reset(**kwargs)

    def step(self, action, adv_action):
        self.adversary.apply(adv_action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.adversary.clear()
        info = dict(info)
        info["adversary_reward"] = -float(reward)
        return obs, reward, terminated, truncated, info

    @property
    def adversary_action_space(self):
        return self.adversary.space
