"""Phase-A lane: DR-PPO on the AD-RRL v3 tasks (task §3) — RECONSTRUCTION.

No public DR-PPO implementation exists (the AD-RRL repo trains AD-RRL; the
paper's appendix E states: PPO from Stable-Baselines3 with mass and friction
uniformly sampled per the Table-4 uncertainty intervals). This module
reconstructs the baseline:

  * environments: AD-RRL's own XML assets (upstreams/AD-RRL/polygrad/
    environments/assets/) on gymnasium's MujocoEnv, replicating the AD-RRL
    FullObs env semantics verbatim (full qpos in the observation, per-task
    reward/termination/reset-noise as in their polygrad/environments/*.py):
      hopper:  fwd vel + alive 1.0 - 1e-3 sum(a^2), termination height>0.7,
               |ang|<0.2, |s[2:]|<100, isfinite; reset noise +-0.005; skip 4
      walker2d: fwd vel + alive 1.0 - 1e-3 sum(a^2), termination height in
               (0.8,2.0), |ang|<1.0; reset noise +-0.005; skip 4
      halfcheetah: fwd vel - 0.1 sum(a^2), no termination; reset noise
               +-0.1 qpos / 0.1 std qvel; skip 5
  * DR wrapper [REBUILD choices, paper states only "uniformly sampled from
    the uncertainty intervals"]: resample at every reset; mass interval as
    the ABSOLUTE torso mass (body 1); friction interval as the floor geom's
    sliding friction coefficient.
  * PPO: SB3 (2.3.2) with default hyperparameters, single env, no obs/reward
    normalisation [REBUILD: paper specifies only "SB3 PPO"]. 1.5M steps.
  * evaluation: 1.2x the nominal torso mass (read from the pristine model),
    nominal friction kept, 30 episodes with reset seeds [20000..20029],
    1,000-step episode cap; actual termination lengths recorded.

Driver: python -m mujoco_baselines.phase_a.dr_v3 <task> <seed>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

from ..constants import EXP_ROOT, RUNS_DIR

# AD-RRL's vendored XMLs use coordinate="global" (unsupported by MuJoCo >=2.3.3);
# the robustness-gymnasium assets are the official local-coordinate conversions of
# the same models (identical geoms, friction, actuators, gear, integrator).
ASSETS = os.path.join(EXP_ROOT, "upstreams/rmbpo-tmlr/embodied/robustness-gymnasium/gymnasium/envs/mujoco/assets")
MJB_PY = os.path.join(EXP_ROOT, "envs/mjb/bin/python")
STEPS = 1_500_000
EPISODES = 30
SEEDS = list(range(20000, 20030))

SPECS = {
    "hopper": dict(xml="hopper.xml", frame_skip=4, kind="hopper",
                   mass_range=(0.5, 6.5), friction_range=(0.1, 3.0)),
    "halfcheetah": dict(xml="half_cheetah.xml", frame_skip=5, kind="cheetah",
                        mass_range=(3.5, 9.5), friction_range=(0.2, 0.8)),
    "walker2d": dict(xml="walker2d.xml", frame_skip=4, kind="walker",
                     mass_range=(0.5, 6.5), friction_range=(0.5, 2.0)),
}


def build_env(task: str, rng=None, mass=None, friction=None):
    """AD-RRL FullObs env on gymnasium with optional per-episode DR."""
    import gymnasium as gym
    from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
    from gymnasium import spaces
    
    spec = SPECS[task]
    xml = os.path.join(ASSETS, spec["xml"])

    class FullObs(MujocoEnv):
        metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

        def __init__(self):
            obs_dim = self.model_nq_nv(xml)
            super().__init__(model_path=xml, frame_skip=spec["frame_skip"],
                             observation_space=spaces.Box(
                                 low=-np.inf, high=np.inf, shape=(obs_dim,),
                                 dtype=np.float64))
            loww = self.model.actuator_ctrlrange[:, 0].min()
            highh = self.model.actuator_ctrlrange[:, 1].max()
            self.action_space = spaces.Box(low=loww, high=highh,
                                           shape=(self.model.nu,), dtype=np.float32)

        @staticmethod
        def model_nq_nv(x):
            import mujoco
            m = mujoco.MjModel.from_xml_path(x)
            return m.nq + m.nv

        def step(self, action):
            x_before = self.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            x_after = self.data.qpos[0]
            obs = self._get_obs()
            if spec["kind"] == "cheetah":
                reward = (x_after - x_before) / self.dt - 0.1 * np.square(action).sum()
                terminated = False
            else:
                height, ang = self.data.qpos[1], self.data.qpos[2]
                reward = (x_after - x_before) / self.dt + 1.0 \
                    - 1e-3 * np.square(action).sum()
                if spec["kind"] == "hopper":
                    s = self.state_vector()[2:]
                    healthy = (np.isfinite(s).all() and (np.abs(s) < 100).all()
                               and height > 0.7 and abs(ang) < 0.2)
                else:
                    healthy = (0.8 < height < 2.0 and -1.0 < ang < 1.0)
                terminated = not healthy
            return obs, float(reward), bool(terminated), False, {}

        def _get_obs(self):
            if spec["kind"] == "cheetah":
                return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
            if spec["kind"] == "hopper":
                return np.concatenate([
                    self.data.qpos.flat,
                    np.clip(self.data.qvel.flat, -10, 10)])
            return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

        def reset_model(self):
            if spec["kind"] == "cheetah":
                qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
                qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
            else:
                qpos = self.init_qpos + self.np_random.uniform(-0.005, 0.005, self.model.nq)
                qvel = self.init_qvel + self.np_random.uniform(-0.005, 0.005, self.model.nv)
            self.set_state(qpos, qvel)
            return self._get_obs()

    env = FullObs()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    m0 = float(env.unwrapped.model.body_mass[1])
    floor = _floor_geom_id(env.unwrapped.model)
    base_friction = float(env.unwrapped.model.geom_friction[floor, 0])

    class DRWrapper(gym.Wrapper):
        """Resamples absolute torso mass + floor sliding friction per reset."""

        def reset(self, *, seed=None, options=None):
            obs, info = self.env.reset(seed=seed, options=options)
            model = self.env.unwrapped.model
            if mass is not None:
                model.body_mass[1] = float(mass)
            elif rng is not None:
                model.body_mass[1] = float(rng.uniform(*spec["mass_range"]))
            if friction is not None:
                model.geom_friction[floor, 0] = float(friction)
            elif rng is not None:
                model.geom_friction[floor, 0] = float(rng.uniform(*spec["friction_range"]))
            return obs, info

    return DRWrapper(env), m0, base_friction


def _floor_geom_id(model):
    for i in range(model.ngeom):
        if model.geom(i).name and "floor" in model.geom(i).name:
            return i
    # fall back to the largest plane
    return int(np.argmax(model.geom_size[:, 1] * model.geom_size[:, 0]))


def train(job_dir: str, task: str, seed: int) -> str:
    """SB3 PPO (defaults) on the DR env; returns the checkpoint path."""
    from stable_baselines3 import PPO
    rng = np.random.default_rng(seed)
    env, m0, f0 = build_env(task, rng=rng)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=STEPS, progress_bar=False)
    ck = os.path.join(job_dir, "ckpt")
    os.makedirs(ck, exist_ok=True)
    model.save(os.path.join(ck, "final"))
    env.close()
    json.dump({"real_transitions": int(model.num_timesteps),
               "wall_s": time.time() - t0,
               "nominal_mass": m0, "nominal_floor_friction": f0},
              open(os.path.join(job_dir, "counters.json"), "w"), indent=1)
    print(f"[dr-a] trained {model.num_timesteps} steps", flush=True)
    return os.path.join(ck, "final.zip")


def evaluate(job_dir: str, task: str, seed: int, ckpt: str):
    """1.2x nominal mass point, nominal friction, 30 episodes."""
    from stable_baselines3 import PPO
    env, m0, f0 = build_env(task, mass=1.2 * m0_nominal(job_dir), friction=None)
    model = PPO.load(ckpt, device="cpu")
    records = []
    for i, s in enumerate(SEEDS):
        obs, _ = env.reset(seed=s)
        done = False
        total, length, term = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)
            length += 1
            term = term or terminated
            done = terminated or truncated
        records.append({
            "task": task, "condition_id": 0, "condition_str": "mass_x1.2",
            "mass_kg": float(1.2 * m0_nominal(job_dir)),
            "friction_multiplier": 1.0, "is_ood": True, "is_nominal": False,
            "eval_seed": int(s), "return": float(total), "length": int(length),
            "terminated": bool(term), "truncated": bool((not term) and length >= 1000),
            "extra": json.dumps({"point": "1.2x nominal torso mass, nominal friction",
                                 "v3_default_cap": 1000}),
        })
    env.close()
    return records


def m0_nominal(job_dir):
    return json.load(open(os.path.join(job_dir, "counters.json")))["nominal_mass"]


def main(task: str, seed: int) -> int:
    job_dir = os.path.join(RUNS_DIR, "source", "dr_ppo", task, f"seed{seed}")
    os.makedirs(job_dir, exist_ok=True)
    status_p = os.path.join(job_dir, "status.json")
    if os.path.exists(status_p) and json.load(open(status_p)).get("state") == "DONE":
        print(f"[dr-a] already done {job_dir}")
        return 0
    json.dump({"state": "RUNNING", "pid": os.getpid(), "started": time.time()},
              open(status_p, "w"))
    t0 = time.time()
    try:
        ckpt = os.path.join(job_dir, "ckpt", "final.zip")
        if not os.path.exists(ckpt):
            ckpt = train(job_dir, task, seed)
        records = evaluate(job_dir, task, seed, ckpt)
        from ..eval.records import save_records
        save_records(records, "source", "dr_ppo", task, seed, "final",
                     time.time() - t0)
        import numpy as np
        rets = [r["return"] for r in records]
        summary = {
            "point_mean": float(np.mean(rets)),
            "point_std": float(np.std(rets)),
            "mean_length": float(np.mean([r["length"] for r in records])),
            "frac_terminated": float(np.mean([r["terminated"] for r in records])),
            "reference": "AD-RRL fig 2 interpolation ~1589/2199/2196",
            "n_episodes": len(records),
        }
        json.dump(summary, open(os.path.join(job_dir, "summary.json"), "w"), indent=1)
        manifest = {
            "protocol": "source", "algo": "dr_ppo", "task": task,
            "train_seed": seed, "state": "DONE", "real_transitions": STEPS,
            "wall_s": time.time() - t0, "command": " ".join(sys.argv),
            "checkpoint": "final", "n_eval_episodes": len(records),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(EXP_ROOT, "runs.jsonl"), "a") as f:
            f.write(json.dumps(manifest) + "\n")
        json.dump({"state": "DONE", "rc": 0, "finished": time.time(),
                   "summary": summary}, open(status_p, "w"), indent=1)
        print(f"[dr-a] DONE {task}/{seed}: {summary}")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        open(os.path.join(job_dir, "stderr_tail.txt"), "a").write(traceback.format_exc())
        json.dump({"state": "FAILED", "error": str(e),
                   "finished": time.time()}, open(status_p, "w"), indent=1)
        print(f"[dr-a] FAILED {task}/{seed}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m mujoco_baselines.phase_a.dr_v3 <task> <seed>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
