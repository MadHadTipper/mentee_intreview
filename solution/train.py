"""Level 4 deliverable — train an RL policy on either sim.

Implement a CLI that:

    python -m solution.train --sim {isaaclab,mujoco} \\
        --config PATH_TO_YAML \\
        --steps N \\
        --out DIR \\
        --seed K

writes ``{DIR}/progress.csv`` with columns ``step,mean_episode_reward`` (one
row per logging interval, plus a final row), trained mean episode reward
beating a random-policy baseline.

Use any RL library you prefer — stable-baselines3, RLlib, CleanRL, tianshou,
etc. The package already declares ``stable-baselines3`` and ``gymnasium`` as
optional deps via the ``[rl]`` extra.

See ``interview/level_4.md`` for the full contract & verification.

Run ``pytest verify/level_4.py -m slow`` after implementing — it must pass.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a policy on one of the dummy sims.")
    parser.add_argument("--sim", choices=["isaaclab", "mujoco"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    raise NotImplementedError(
        "Level 4: implement train(). See solution/train.py docstring and interview/level_4.md. "
        f"Got args: {args!r}"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
