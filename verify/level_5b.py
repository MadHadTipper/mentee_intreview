"""Level 5b — API smoke check (NOT correctness).

Confirms that ``solution.mdp.load_mdp`` accepts a YAML/dict config that
includes a ``noise_std: σ`` field on an observation term, and that calling
the returned ``obs_fn`` doesn't crash. Does NOT verify the noise's
distribution, reproducibility under seed, or zero-noise identity — those
live in the interviewer's grader.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from solution.mdp import load_mdp


CFG_WITH_NOISE = {
    "observations": [{"name": "joint_pos", "noise_std": 0.05}],
    "rewards": [{"name": "tracking", "weight": 1.0}],
    "terminations": {"mode": "any", "terms": [{"name": "timeout", "params": {"max_steps": 200}}]},
    "seed": 0,
}


def _state(num_envs: int = 2) -> SimpleNamespace:
    z2 = np.zeros((num_envs, 2), dtype=np.float32)
    return SimpleNamespace(
        joint_pos=z2.copy(),
        joint_vel=z2.copy(),
        ee_pos=z2.copy(),
        goal=z2.copy(),
        last_action=z2.copy(),
        applied_torque=z2.copy(),
        contact_force=np.zeros(num_envs, dtype=np.float32),
        episode_step=np.zeros(num_envs, dtype=np.int64),
    )


def test_load_mdp_accepts_noise_std_field_without_crashing() -> None:
    _, obs_fn, _, obs_dim, _ = load_mdp(CFG_WITH_NOISE)
    out = obs_fn(_state())
    assert out.shape[-1] == obs_dim, f"obs_fn output width {out.shape[-1]} != obs_dim {obs_dim}"
