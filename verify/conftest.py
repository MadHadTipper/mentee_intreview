"""Shared fixtures for the per-level verification suite.

Every test here imports from ``solution.*`` — i.e., the candidate's code.
Tests pass against the reference solution and fail with NotImplementedError
against the empty skeleton.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


@pytest.fixture
def num_envs() -> int:
    return 4


@pytest.fixture
def matched_state_inputs():
    """Joint state + goal that both sims can be set to deterministically.

    Returns a dict with arrays sized for ``num_envs=4`` (the default fixture).
    Tests that need a different batch size should set them themselves.
    """
    return {
        "q":    np.array([[0.30, -0.20], [-1.10, 0.40], [0.05, 1.20], [2.40, -0.70]], dtype=np.float32),
        "qd":   np.array([[0.10, -0.05], [0.00, 0.00], [-0.20, 0.15], [0.05, 0.05]], dtype=np.float32),
        "goal": np.array([[1.00, 0.50], [-0.80, -0.40], [0.20, 1.10], [1.50, -1.10]], dtype=np.float32),
    }


def set_isaaclab_state(adapter, q, qd, goal) -> None:
    """Reach into an IsaacLabAdapter (or any wrapper that holds an IsaacLabEnv
    accessible as ``._env``, ``.env``, or ``.sim``) and set joint state + goal."""
    raw = _unwrap_isaaclab(adapter)
    raw.set_joint_state(torch.from_numpy(q), torch.from_numpy(qd))
    raw.set_target(torch.from_numpy(goal))


def set_mujoco_state(adapter, q, qd, goal) -> None:
    raw = _unwrap_mujoco(adapter)
    raw.write_qpos(q)
    raw.write_qvel(qd)
    raw.write_target(goal)


def _unwrap_isaaclab(unified):
    """Find the underlying IsaacLabEnv inside a candidate's unified env."""
    from dummy_isaaclab import IsaacLabEnv
    return _walk_for(unified, IsaacLabEnv)


def _unwrap_mujoco(unified):
    from dummy_mujoco import MuJoCoSim
    return _walk_for(unified, MuJoCoSim)


def _walk_for(obj, target_type, max_depth: int = 6):
    """Best-effort: scan attributes for an instance of ``target_type``.

    Candidates store the underlying sim under various names (``._env``,
    ``._sim``, ``.adapter._env``, ``.unwrapped``, etc.). We recurse a few levels
    looking for it. If a candidate stores it somewhere truly unusual the test
    will fail loudly with a clear message — they can rename or expose
    ``unwrapped`` to make verification work.
    """
    if isinstance(obj, target_type):
        return obj
    if max_depth == 0:
        raise RuntimeError(
            f"could not find a {target_type.__name__} inside the unified env. "
            "Expose it as `.unwrapped`, `._env`/`._sim`, or store it in an "
            "attribute named obviously."
        )
    for name in ("unwrapped", "_env", "_sim", "env", "sim", "adapter", "_adapter", "inner"):
        if hasattr(obj, name):
            try:
                return _walk_for(getattr(obj, name), target_type, max_depth - 1)
            except RuntimeError:
                continue
    raise RuntimeError(
        f"could not find a {target_type.__name__} inside {obj!r}. "
        "Expose it as `.unwrapped`, `._env`/`._sim`, or another conventional attribute."
    )
