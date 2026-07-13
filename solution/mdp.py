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


def reward_fn(state: Any, action: Any) -> Any:
    """Level 2 reward — see ``interview/level_2.md`` for the full spec.

    Implements (per env in the batch):
        reward = − ‖ee − goal‖
               − 0.001 · ‖joint_vel‖²
               − 0.001 · ‖action‖²
               + 50 · 1[‖ee − goal‖ < 0.05]

    Returns: ``(num_envs,)`` numpy float32.
    """
    raise NotImplementedError(
        "Level 2: implement reward_fn. See solution/mdp.py docstring and interview/level_2.md."
    )


def obs_fn(state: Any) -> Any:
    """Level 2 observation — see ``interview/level_2.md`` for the full spec.

    Implements (per env in the batch):
        obs = concat[ sin(q), cos(q), joint_vel, ee − goal ]    # 8 dims

    Returns: ``(num_envs, 8)`` numpy float32.
    """
    raise NotImplementedError(
        "Level 2: implement obs_fn. See solution/mdp.py docstring and interview/level_2.md."
    )


def termination_fn(state: Any) -> Any:
    """Level 2 termination — see ``interview/level_2.md`` for the full spec.

    Implements (per env in the batch):
        done = (‖ee − goal‖ < 0.05)  OR  (episode_step ≥ 200)

    Returns: ``(num_envs,)`` numpy bool.
    """
    raise NotImplementedError(
        "Level 2: implement termination_fn. See solution/mdp.py docstring and interview/level_2.md."
    )


def load_mdp(config: str | dict):
    """Level 3 (+ Level 5): build (reward_fn, obs_fn, termination_fn, obs_dim, act_dim)
    from a YAML file path or a dict.

    Returns: tuple ``(reward_fn, obs_fn, termination_fn, obs_dim: int, act_dim: int)``.
    """
    raise NotImplementedError(
        "Level 3: implement load_mdp. See solution/mdp.py docstring and interview/level_3.md."
    )
