"""Level 3 — API smoke check (NOT correctness).

Confirms ``solution.mdp.load_mdp`` exists and returns a 5-tuple of
``(reward_fn, obs_fn, termination_fn, obs_dim, act_dim)`` for both YAML-path
and dict inputs. Does NOT verify that two distinct configs produce
distinguishable functions, that term composition is mathematically correct,
or that unknown terms raise — those live in the interviewer's grader.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from solution.mdp import load_mdp


MIN_CFG = {
    "observations": [{"name": "joint_pos"}, {"name": "joint_vel"}],
    "rewards": [{"name": "tracking", "weight": 1.0}],
    "terminations": {"mode": "any", "terms": [{"name": "timeout", "params": {"max_steps": 200}}]},
}


def _write_yaml(d: dict) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    return Path(f.name)


def test_load_mdp_returns_5_tuple_from_path() -> None:
    out = load_mdp(str(_write_yaml(MIN_CFG)))
    assert len(out) == 5, "load_mdp must return (reward_fn, obs_fn, termination_fn, obs_dim, act_dim)"
    reward_fn, obs_fn, term_fn, obs_dim, act_dim = out
    assert callable(reward_fn) and callable(obs_fn) and callable(term_fn)
    assert isinstance(obs_dim, int) and obs_dim > 0
    assert isinstance(act_dim, int) and act_dim > 0


def test_load_mdp_accepts_dict() -> None:
    out = load_mdp(MIN_CFG)
    assert len(out) == 5
