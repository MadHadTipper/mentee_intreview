"""Level 1 deliverable — unified env adapter + unified state.

Implement ``make_env(name, num_envs, *, seed=None)`` returning a unified
batched env that wraps ``dummy_isaaclab.IsaacLabEnv`` (when ``name="isaaclab"``)
and ``dummy_mujoco.MuJoCoSim`` (when ``name="mujoco"``) behind a single Python
interface.

The returned object MUST expose, regardless of which sim it wraps:

    .num_envs:   int
    .obs_dim:    int
    .act_dim:    int
    .reset() -> obs: np.ndarray (num_envs, obs_dim) float32
    .step(action: np.ndarray (num_envs, act_dim) float32)
        -> (obs, reward, done, info)
            obs:    np.ndarray (num_envs, obs_dim) float32
            reward: np.ndarray (num_envs,)         float32
            done:   np.ndarray (num_envs,)         bool
            info:   dict
    .close() -> None

AND a way to access the per-step **unified state object** — the same
single Python object regardless of which sim is underneath, exposing at
least these attributes (numpy arrays):

    .joint_pos      (num_envs, 2) float32 — joint angles, wrapped to [-π, π]
    .joint_vel      (num_envs, 2) float32 — joint angular velocities
    .ee_pos         (num_envs, 2) float32 — end-effector (x, y)
    .goal           (num_envs, 2) float32 — per-env goal (x, y)
    .episode_step   (num_envs,)   int64   — steps since each env's last reset
    .last_action    (num_envs, 2) float32 — last action passed to step()

How you expose this state object is your design choice — `env.state`
attribute, `env.snapshot()` method, returned in `info`, etc. The verify
suite accepts any of those common patterns.

Notes:
- The two sims have deliberately different APIs (different method names,
  different return-tuple orders, torch vs numpy, auto-reset vs caller-managed
  reset). Reconciling all of that is your job.
- Standardize on numpy arrays at this boundary so any RL library plugs in.
- Action range is ``[-1, 1]`` per joint for both sims; both interpret it as a
  normalized joint position target by default.

See ``interview/level_1.md`` for the full contract & verification.

Run ``pytest verify/level_1.py`` after implementing — it must pass.
"""
from __future__ import annotations

import numpy as np


def make_env(name: str, num_envs: int, *, seed: int | None = None):
    """Construct a unified batched env wrapping the named dummy sim.

    Args:
        name: ``"isaaclab"`` or ``"mujoco"``.
        num_envs: parallel batch size.
        seed: optional seed forwarded to the underlying sim.

    Returns:
        An object meeting the unified env contract documented above
        (including the unified state object — see module docstring).
    """
    raise NotImplementedError(
        "Level 1: implement make_env. See solution/env.py docstring and interview/level_1.md."
    )
