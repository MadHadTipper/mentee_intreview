"""Levels 2 / 3 / 5 deliverables — shared MDP definition.

Level 2: implement the concrete reward / obs / termination functions
specified in ``interview/level_2.md``. They must be **simulation-agnostic** —
the same Python code (no per-sim branching) works for both ``isaaclab`` and
``mujoco``. To make that possible, your Level-1 unified env should expose
a unified state object that all three functions consume.

Level 3: implement ``load_mdp(config)`` that composes ``reward_fn`` / ``obs_fn``
/ ``termination_fn`` from a YAML / dict config built out of named, parameterized
**terms**. Switching the task should be a config edit, not a code edit.

Level 5a / 5b: extend ``load_mdp`` so the config can carry ``delay`` and
``noise_std`` per observation term. Per-env independent ring buffer for the
delay; reproducible Gaussian noise from a top-level ``seed``.

See ``interview/level_2.md / level_3.md / level_5.md`` for full contracts & verification.

Run ``pytest verify/level_2.py``, ``pytest verify/level_3.py``, etc., after
implementing.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from solution.constants import (
    CONTROL_EFFORT_WEIGHT,
    MAX_STEPS,
    OBS_DIM_L2,
    SMOOTHNESS_WEIGHT,
    SUCCESS_BONUS_WEIGHT,
    SUCCESS_THRESHOLD,
    TRACKING_WEIGHT,
)

# --------------------------------------------------------------- reward terms


def tracking(state: Any, action: Any) -> np.ndarray:
    """-‖ee - goal‖."""
    return -np.linalg.norm(state.ee_pos - state.goal, axis=-1).astype(np.float32)


def smoothness(state: Any, action: Any) -> np.ndarray:
    """-‖joint_vel‖²."""
    return -np.sum(state.joint_vel**2, axis=-1).astype(np.float32)


def control_effort(state: Any, action: Any) -> np.ndarray:
    """-‖action‖²."""
    return -np.sum(np.asarray(action, dtype=np.float32) ** 2, axis=-1).astype(np.float32)


def success_bonus(state: Any, action: Any) -> np.ndarray:
    """1[‖ee - goal‖ < SUCCESS_THRESHOLD]."""
    dist = np.linalg.norm(state.ee_pos - state.goal, axis=-1)
    return (dist < SUCCESS_THRESHOLD).astype(np.float32)


# ----------------------------------------------------------------- obs terms


def sin_cos_joint_pos(state: Any) -> np.ndarray:
    """concat[sin(q), cos(q)] -> (N, 4); avoids the +pi -> -pi discontinuity."""
    return np.concatenate([np.sin(state.joint_pos), np.cos(state.joint_pos)], axis=-1).astype(
        np.float32
    )


def joint_vel_obs(state: Any) -> np.ndarray:
    """Pass-through joint_vel -> (N, 2); named _obs to avoid shadowing state.joint_vel."""
    return state.joint_vel.astype(np.float32)


def ee_minus_goal(state: Any) -> np.ndarray:
    """ee_pos - goal -> (N, 2)."""
    return (state.ee_pos - state.goal).astype(np.float32)


# ----------------------------------------------------------- termination terms


def reached_goal(state: Any) -> np.ndarray:
    """‖ee - goal‖ < SUCCESS_THRESHOLD."""
    return np.linalg.norm(state.ee_pos - state.goal, axis=-1) < SUCCESS_THRESHOLD


def timeout(state: Any) -> np.ndarray:
    """episode_step >= MAX_STEPS."""
    return np.asarray(state.episode_step) >= MAX_STEPS


# ------------------------------------------------------------------ combinators


def _weighted_sum(terms, state: Any, action: Any) -> np.ndarray:
    """Sum weight * term(state, action) over all reward terms."""
    return sum(w * fn(state, action) for fn, w in terms).astype(np.float32)


def _concat(terms, state: Any) -> np.ndarray:
    """Concatenate obs term outputs along the last axis, in list order."""
    return np.concatenate([fn(state) for fn in terms], axis=-1).astype(np.float32)


def _logical_or(terms, state: Any) -> np.ndarray:
    """Elementwise OR of termination term outputs."""
    return np.logical_or.reduce([fn(state) for fn in terms])


_REWARD_TERMS = [
    (tracking, TRACKING_WEIGHT),
    (smoothness, SMOOTHNESS_WEIGHT),
    (control_effort, CONTROL_EFFORT_WEIGHT),
    (success_bonus, SUCCESS_BONUS_WEIGHT),
]
_OBS_TERMS = [sin_cos_joint_pos, joint_vel_obs, ee_minus_goal]
_TERMINATION_TERMS = [reached_goal, timeout]


def reward_fn(state: Any, action: Any) -> Any:
    """Level 2 reward — see ``interview/level_2.md`` for the full spec.

    Implements (per env in the batch):
        reward = − ‖ee − goal‖
               − 0.001 · ‖joint_vel‖²
               − 0.001 · ‖action‖²
               + 50 · 1[‖ee − goal‖ < 0.05]

    Returns: ``(num_envs,)`` numpy float32.
    """
    return _weighted_sum(_REWARD_TERMS, state, action)


def obs_fn(state: Any) -> Any:
    """Level 2 observation — see ``interview/level_2.md`` for the full spec.

    Implements (per env in the batch):
        obs = concat[ sin(q), cos(q), joint_vel, ee − goal ]    # 8 dims

    Returns: ``(num_envs, 8)`` numpy float32.
    """
    return _concat(_OBS_TERMS, state)


def termination_fn(state: Any) -> Any:
    """Level 2 termination — see ``interview/level_2.md`` for the full spec.

    Implements (per env in the batch):
        done = (‖ee − goal‖ < 0.05)  OR  (episode_step ≥ 200)

    Returns: ``(num_envs,)`` numpy bool.
    """
    return _logical_or(_TERMINATION_TERMS, state)


# ------------------------------------------------------------------------ wiring


class MdpEnv:
    """Wraps a UnifiedEnv, substituting obs/reward with the composed MDP fns.

    ``done`` is reused from the wrapped env's own ``step()`` rather than
    recomputed via ``termination_fn`` on ``env.state``, since Level 1's
    ``UnifiedEnv`` auto-resets done envs in-place: by the time ``step()``
    returns, ``env.state`` already reflects the fresh post-reset episode for
    any env that just finished, so it can't reproduce the terminal-step
    ``termination_fn`` value. The two are formula-identical (same
    ``SUCCESS_THRESHOLD``/``MAX_STEPS``) on every non-boundary step.
    """

    def __init__(
        self,
        env: Any,
        reward_fn=reward_fn,
        obs_fn=obs_fn,
        termination_fn=termination_fn,
        obs_dim: int = OBS_DIM_L2,
    ) -> None:
        self.env = env
        self._reward_fn = reward_fn
        self._obs_fn = obs_fn
        self._termination_fn = termination_fn
        self.num_envs = env.num_envs
        self.obs_dim = obs_dim
        self.act_dim = env.act_dim

    def reset(self) -> np.ndarray:
        """Reset every env; returns obs (num_envs, obs_dim) float32."""
        self.env.reset()
        return self._obs_fn(self.env.state)

    def step(self, action: np.ndarray):
        """One RL step; returns (obs, reward, done, info)."""
        _, _, done, info = self.env.step(action)
        state = self.env.state
        obs = self._obs_fn(state)
        reward = self._reward_fn(state, action)
        return obs, reward, done, info

    def close(self) -> None:
        """Tear down the wrapped env."""
        self.env.close()


def load_mdp(config: str | dict):
    """Level 3 (+ Level 5): build (reward_fn, obs_fn, termination_fn, obs_dim, act_dim)
    from a YAML file path or a dict.

    Returns: tuple ``(reward_fn, obs_fn, termination_fn, obs_dim: int, act_dim: int)``.
    """
    raise NotImplementedError(
        "Level 3: implement load_mdp. See solution/mdp.py docstring and interview/level_3.md."
    )
