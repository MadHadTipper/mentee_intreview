"""Level 2 — API smoke check (NOT correctness).

Confirms ``solution.mdp.{reward_fn, obs_fn, termination_fn}`` exist as
module-level callables and that, when wired into your unified env via Level
1, ``env.step(...)`` returns arrays of the right shape and dtype.

This file does NOT verify cross-sim equality on matched states (that's the
interviewer's grader). See ``INTERVIEW.md`` § "What you're graded on".
"""
from __future__ import annotations

import numpy as np
import pytest

from solution.env import make_env
from solution.mdp import obs_fn, reward_fn, termination_fn


def test_module_level_fns_are_callable() -> None:
    """``solution.mdp`` must expose three module-level callables."""
    assert callable(reward_fn), "solution.mdp.reward_fn must be a callable"
    assert callable(obs_fn), "solution.mdp.obs_fn must be a callable"
    assert callable(termination_fn), "solution.mdp.termination_fn must be a callable"


@pytest.mark.parametrize("sim", ["isaaclab", "mujoco"])
def test_step_outputs_have_consistent_shape(sim: str, num_envs: int) -> None:
    """A single step through your unified env must produce arrays of consistent
    shape with what reward_fn / obs_fn / termination_fn return. Doesn't assert
    *what* the values are — just that the wiring is plausible."""
    env = make_env(sim, num_envs, seed=0)
    try:
        env.reset()
        action = np.zeros((num_envs, env.act_dim), dtype=np.float32)
        obs, reward, done, _ = env.step(action)
        assert obs.shape[0] == num_envs and obs.ndim == 2
        assert reward.shape == (num_envs,)
        assert done.shape == (num_envs,)
    finally:
        env.close()
