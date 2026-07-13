"""Level 1 — API smoke check (NOT correctness).

Confirms ``solution.env.make_env`` exists with the right signature and that
calling it returns an object with the documented attributes / methods, and
that ``reset()`` / ``step(action)`` produce arrays of the correct shape and
dtype on a single step.

This file does NOT assert determinism, behavior, cross-sim consistency,
learning, or any value-level property. Those live in the interviewer's
private grader. See ``INTERVIEW.md`` § "What you're graded on".
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from solution.env import make_env


@pytest.mark.parametrize("sim", ["isaaclab", "mujoco"])
def test_make_env_returns_object_with_required_surface(sim: str, num_envs: int) -> None:
    env = make_env(sim, num_envs, seed=0)
    try:
        for attr in ("num_envs", "obs_dim", "act_dim"):
            assert hasattr(env, attr), f"unified env is missing attribute {attr!r}"
            assert isinstance(getattr(env, attr), int), f"{attr} must be int, got {type(getattr(env, attr)).__name__}"
        for method in ("reset", "step", "close"):
            assert hasattr(env, method) and callable(getattr(env, method)), (
                f"unified env is missing method {method}()"
            )
    finally:
        env.close()


@pytest.mark.parametrize("sim", ["isaaclab", "mujoco"])
def test_reset_returns_correct_array_shape(sim: str, num_envs: int) -> None:
    env = make_env(sim, num_envs, seed=0)
    try:
        obs = env.reset()
        assert isinstance(obs, np.ndarray), f"reset() must return np.ndarray, got {type(obs).__name__}"
        assert obs.shape == (num_envs, env.obs_dim), (
            f"reset() obs shape {obs.shape} != expected ({num_envs}, {env.obs_dim})"
        )
        assert obs.dtype == np.float32, f"reset() obs dtype {obs.dtype} != float32"
    finally:
        env.close()


@pytest.mark.parametrize("sim", ["isaaclab", "mujoco"])
def test_step_returns_correct_4tuple_shape(sim: str, num_envs: int) -> None:
    env = make_env(sim, num_envs, seed=0)
    try:
        env.reset()
        action = np.zeros((num_envs, env.act_dim), dtype=np.float32)
        out = env.step(action)
        assert len(out) == 4, f"step() must return a 4-tuple (obs, reward, done, info), got {len(out)} elements"
        obs, reward, done, info = out
        assert obs.shape == (num_envs, env.obs_dim) and obs.dtype == np.float32
        assert reward.shape == (num_envs,) and reward.dtype == np.float32
        assert done.shape == (num_envs,) and done.dtype == np.bool_
        assert isinstance(info, dict)
    finally:
        env.close()


def test_make_env_signature_accepts_seed_kwarg() -> None:
    sig = inspect.signature(make_env)
    assert "seed" in sig.parameters, "make_env must accept a `seed` keyword argument"


# ----------------------------------------------------------------- unified state object

REQUIRED_STATE_ATTRS = ("joint_pos", "joint_vel", "ee_pos", "goal", "episode_step", "last_action")


def _find_state(env):
    """Best-effort lookup of the candidate's per-step unified state object.

    Accepts any of: ``env.state`` attribute, ``env.snapshot()`` method,
    ``env.get_state()`` method. Skips with a clear message if none of
    those exist — candidates can update their adapter to match.
    """
    if hasattr(env, "state"):
        s = env.state
        if s is not None:
            return s
    for meth in ("snapshot", "get_state"):
        if hasattr(env, meth) and callable(getattr(env, meth)):
            return getattr(env, meth)()
    pytest.skip(
        "could not find a unified state object: expected `env.state` "
        "attribute, or `env.snapshot()` / `env.get_state()` method. "
        "See interview/level_1.md §1b — Unified state object."
    )


@pytest.mark.parametrize("sim", ["isaaclab", "mujoco"])
def test_state_object_exposes_required_attributes(sim: str, num_envs: int) -> None:
    env = make_env(sim, num_envs, seed=0)
    try:
        env.reset()
        env.step(np.zeros((num_envs, env.act_dim), dtype=np.float32))
        state = _find_state(env)
        for attr in REQUIRED_STATE_ATTRS:
            assert hasattr(state, attr), (
                f"state object missing attribute {attr!r}. "
                f"interview/level_1.md §1b lists the required attribute names."
            )
    finally:
        env.close()


@pytest.mark.parametrize("sim", ["isaaclab", "mujoco"])
def test_state_attribute_shapes_and_dtypes(sim: str, num_envs: int) -> None:
    env = make_env(sim, num_envs, seed=0)
    try:
        env.reset()
        env.step(np.zeros((num_envs, env.act_dim), dtype=np.float32))
        state = _find_state(env)
        # 2D arrays of shape (num_envs, 2) — per-joint or per-coordinate.
        for attr in ("joint_pos", "joint_vel", "ee_pos", "goal", "last_action"):
            arr = np.asarray(getattr(state, attr))
            assert arr.shape == (num_envs, 2), f"state.{attr} shape {arr.shape} != ({num_envs}, 2)"
            assert np.issubdtype(arr.dtype, np.floating), f"state.{attr} dtype {arr.dtype} not floating"
        # 1D integer episode_step.
        es = np.asarray(state.episode_step)
        assert es.shape == (num_envs,), f"state.episode_step shape {es.shape} != ({num_envs},)"
        assert np.issubdtype(es.dtype, np.integer), f"state.episode_step dtype {es.dtype} not integer"
    finally:
        env.close()
