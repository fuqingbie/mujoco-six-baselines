"""Bypass regression test: RMBPO with the auxiliary model disabled must be
BIT-IDENTICAL to MBPO on the rollout path (same seeds / data / weights / RNG).

Contract being tested (task book, phase B):
  * MBPO and RMBPO share the SAC, the nominal ensemble, the rollout procedure,
    the seeds and the update cadence;
  * with ``bypass_aux=True`` RMBPO constructs no auxiliary model at all, trains
    the nominal model on the same data with the same RNG stream, and generates
    rollouts from the nominal model — so two rollout rounds must reproduce
    MBPO's model-buffer contents exactly (atol=0 / rtol=0);
  * with the auxiliary model ENABLED, the rollouts must differ (the aux path
    actually engages), evaluated under the same RNG state at rollout entry so
    that start states and policy actions coincide and only the prediction path
    differs.

Run:  PYTHONPATH=/root/mjbexp /root/mjbexp/envs/mjb/bin/python -m pytest
      mujoco_baselines/tests/test_rmbpo_bypass.py -v
(or directly:  python mujoco_baselines/tests/test_rmbpo_bypass.py)
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

# the exact-equality comparison runs on CPU; keep torch from spawning one
# thread per core (this box exposes 384) which thrashes on the tiny batches
torch.set_num_threads(int(os.environ.get("MJB_TEST_THREADS", "8")))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mujoco_baselines.algos.mbpo_core import MbpoTrainer, set_global_seed  # noqa: E402
from mujoco_baselines.algos.rmbpo import RmbpoTrainer  # noqa: E402
from mujoco_baselines.config import MODEL_BASED_COMMON  # noqa: E402

TASK = "hopper"
SEED = 0
N_SYNTH = 3000          # synthetic env-buffer transitions
ROLLOUT_ROUNDS = 2


def _tiny_cfg() -> dict:
    cfg = dict(MODEL_BASED_COMMON)
    cfg["rollout"] = dict(cfg["rollout"], batch=2000, mini_batch=1000)
    cfg["model"] = dict(cfg["model"])
    cfg["sac"] = dict(cfg["sac"])
    cfg["warmup"] = 0                  # rounds are driven manually here
    cfg["budget"] = 10                 # train() is never called
    cfg["device"] = "cpu"              # exact-equality device
    cfg["env_buffer"] = 4 * N_SYNTH
    cfg["model_buffer"] = 5000         # no wrapping during the test rollouts
    return cfg


def _fill_env_buffer(buffer, n=N_SYNTH):
    """Deterministic synthetic transitions (local RNG: global stream untouched)."""
    rng = np.random.default_rng(1234)
    obs = rng.normal(size=(n, 11)).astype(np.float64)
    act = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    next_obs = obs + 0.05 * np.sin(act).repeat(11 // 3 + 1, axis=1)[:, :11]
    reward = rng.normal(1.0, 0.1, size=n)
    done = (rng.uniform(size=n) < 0.05).astype(np.int8)
    buffer.add_traj(obs, act, next_obs, reward, done)
    assert buffer.max_sample_size == n


def _snapshot(buffer):
    """Only the filled region of the ring buffer matters, in write order.

    The last ``m = max_sample_size`` samples occupy ring slots
    ``(curr - m + j) % size`` — equal to ``(curr + j) % m`` — for both the
    wrapped and unwrapped case."""
    m = buffer.max_sample_size
    if m == 0:
        return {}
    idx = np.arange(buffer.curr, buffer.curr + m) % m
    return {k: np.array(v[idx], copy=True) for k, v in
            dict(obs=buffer.obs_buffer, action=buffer.action_buffer,
                 next_obs=buffer.next_obs_buffer, reward=buffer.reward_buffer,
                 done=buffer.done_buffer, truncated=buffer.truncated_buffer).items()}


def _model_weights_equal(a: MbpoTrainer, b: MbpoTrainer) -> bool:
    sa = a.transition_model.model.state_dict()
    sb = b.transition_model.model.state_dict()
    if set(sa.keys()) != set(sb.keys()):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def _agent_weights_equal(a: MbpoTrainer, b: MbpoTrainer) -> bool:
    sa, sb = a.agent.state_dict(), b.agent.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def _run_two_rounds(trainer: MbpoTrainer):
    """Two cadence rounds: train nominal -> (aux) -> rollout, twice."""
    out = []
    for _ in range(ROLLOUT_ROUNDS):
        logs, train_data, eval_data = trainer.train_model()
        trainer._after_model_training(train_data, eval_data)
        trainer.rollout_model()
        out.append((_snapshot(trainer.model_buffer), logs))
    return out


def test_bypass_rollout_bit_identical():
    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as d_mbpo, tempfile.TemporaryDirectory() as d_rmb:
        set_global_seed(SEED)
        mbpo = MbpoTrainer(TASK, SEED, cfg, d_mbpo, use_aux=False)
        set_global_seed(SEED)
        rmb = RmbpoTrainer(TASK, SEED, cfg, d_rmb, bypass_aux=True)

        # identical construction -> identical initial weights (no aux net built)
        assert _model_weights_equal(mbpo, rmb), "initial ensemble weights differ"
        assert _agent_weights_equal(mbpo, rmb), "initial SAC weights differ"
        assert not rmb.use_aux and rmb.aux_updates == 0

        _fill_env_buffer(mbpo.env_buffer)
        _fill_env_buffer(rmb.env_buffer)

        # identical RNG stream for both runs
        set_global_seed(777)
        mb_rounds = _run_two_rounds(mbpo)
        set_global_seed(777)
        rm_rounds = _run_two_rounds(rmb)

        for r, (mb, rm) in enumerate(zip(mb_rounds, rm_rounds)):
            assert _model_weights_equal(mbpo, rmb), \
                f"round {r}: nominal model weights diverged"
            for key in ("obs", "action", "next_obs", "reward", "done", "truncated"):
                a, b = mb[0][key], rm[0][key]
                assert a.shape == b.shape, f"round {r} {key}: shape {a.shape} vs {b.shape}"
                assert np.array_equal(a, b), \
                    f"round {r} {key}: bypass RMBPO rollout differs from MBPO " \
                    f"(max abs diff {np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))})"
        # counters must agree too
        assert mbpo.model_transitions == rmb.model_transitions
        assert mbpo.model_train_iterations == rmb.model_train_iterations
        assert rmb.aux_updates == 0
        print(f"[bypass] OK: {ROLLOUT_ROUNDS} rollout rounds bit-identical "
              f"({mbpo.model_transitions} model transitions each, "
              f"{mbpo.model_train_iterations} model train iters)")


def test_aux_engaged_rollout_differs():
    cfg = _tiny_cfg()
    # single rollout minibatch: start states and policy actions are drawn BEFORE
    # any model prediction, so they are shared by both paths and any difference
    # comes from the auxiliary prediction itself (with multiple minibatches the
    # per-sample RNG consumption differs and would shift later actions)
    cfg["rollout"] = dict(cfg["rollout"], batch=1000, mini_batch=1000)
    with tempfile.TemporaryDirectory() as d_mbpo, tempfile.TemporaryDirectory() as d_rmb:
        set_global_seed(SEED)
        mbpo = MbpoTrainer(TASK, SEED, cfg, d_mbpo, use_aux=False)
        set_global_seed(SEED)
        rmb = RmbpoTrainer(TASK, SEED, cfg, d_rmb, bypass_aux=False)
        assert rmb.use_aux

        _fill_env_buffer(mbpo.env_buffer)
        _fill_env_buffer(rmb.env_buffer)

        # MBPO: one model round, then rollout under a fixed RNG state
        set_global_seed(777)
        mb_logs, _, _ = mbpo.train_model()
        set_global_seed(31337)
        mbpo.rollout_model()
        mb_buf = _snapshot(mbpo.model_buffer)

        # RMBPO: model round, aux round, then rollout under the SAME RNG state
        set_global_seed(777)
        rm_logs, train_data, eval_data = rmb.train_model()
        aux_logs = rmb._after_model_training(train_data, eval_data)
        assert rmb.aux_updates > 0, "auxiliary model did not train"
        assert np.isfinite(aux_logs["aux/kl_loss"]), "aux KL not finite"
        assert np.isfinite(aux_logs["aux/robustness_loss"]), "aux robustness not finite"
        set_global_seed(31337)
        rmb.rollout_model()
        rm_buf = _snapshot(rmb.model_buffer)

        # same start states + same policy actions (same RNG entry state), so any
        # difference comes from the auxiliary prediction path itself
        assert np.array_equal(mb_buf["obs"], rm_buf["obs"]), \
            "start states diverged (RNG misalignment in the test)"
        assert np.array_equal(mb_buf["action"], rm_buf["action"]), \
            "rollout actions diverged (RNG misalignment in the test)"
        differs = (not np.allclose(mb_buf["next_obs"], rm_buf["next_obs"])) or \
                  (not np.allclose(mb_buf["reward"], rm_buf["reward"]))
        assert differs, "non-bypassed RMBPO produced MBPO-identical rollouts"
        n_diff = int(np.sum(np.abs(mb_buf["next_obs"] - rm_buf["next_obs"]) > 1e-9))
        print(f"[aux-engaged] OK: rollouts differ ({n_diff}/{mb_buf['next_obs'].size} "
              f"next_obs elements differ; aux iters={rmb.aux_updates}, "
              f"kl={aux_logs['aux/kl_loss']:.3g}, robustness={aux_logs['aux/robustness_loss']:.3g})")


if __name__ == "__main__":
    test_bypass_rollout_bit_identical()
    test_aux_engaged_rollout_differs()
    print("ALL BYPASS TESTS PASSED")
