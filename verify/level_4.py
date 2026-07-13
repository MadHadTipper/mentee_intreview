"""Level 4 — API smoke check (NOT correctness).

Confirms that ``python -m solution.train`` accepts the documented CLI flags,
runs to completion (briefly), and writes a ``progress.csv`` with the right
header. Does NOT verify that the trained policy actually beats a random
baseline — that's the interviewer's grader, which uses a longer training
budget and a learning threshold.

Marked ``slow`` because the brief training run (~2k env-steps) still takes
a few seconds to spin up SB3.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


CONFIGS_DIR = Path(__file__).parent.parent / "solution" / "configs"


def _pick_config() -> Path:
    """Find a usable MDP config under ``solution/configs/``.

    Returns the first ``*.yaml`` file there that parses as a complete MDP
    spec (has ``rewards``, ``observations``, ``terminations``). Skips with
    a clear message if none is found — this is normal until you fill in
    your config at Level 3.
    """
    import yaml
    if not CONFIGS_DIR.exists():
        pytest.skip("no `solution/configs/` directory yet — fill in a config in Level 3.")
    candidates = sorted(CONFIGS_DIR.glob("*.yaml")) + sorted(CONFIGS_DIR.glob("*.yml"))
    for candidate in candidates:
        try:
            cfg = yaml.safe_load(candidate.read_text()) or {}
            if "rewards" in cfg and "observations" in cfg and "terminations" in cfg:
                return candidate
        except Exception:
            continue
    pytest.skip(
        "no valid MDP config found in solution/configs/. Write one with "
        "`rewards:`, `observations:`, `terminations:` keys (see interview/level_3.md)."
    )


@pytest.mark.slow
@pytest.mark.parametrize("sim", ["isaaclab", "mujoco"])
def test_train_runs_briefly_and_writes_progress_csv(sim: str) -> None:
    cfg = _pick_config()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / sim
        out_dir.mkdir()
        cmd = [
            sys.executable,
            "-m", "solution.train",
            "--sim", sim,
            "--config", str(cfg),
            "--steps", "2000",
            "--out", str(out_dir),
            "--seed", "0",
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (
            f"`solution.train --steps 2000` failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        progress = out_dir / "progress.csv"
        assert progress.exists(), f"expected {progress} to exist after training"
        with open(progress) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert "step" in reader.fieldnames, f"progress.csv missing `step` column: {reader.fieldnames}"
            assert "mean_episode_reward" in reader.fieldnames, (
                f"progress.csv missing `mean_episode_reward` column: {reader.fieldnames}"
            )
            rows = list(reader)
            assert rows, "progress.csv contains no data rows"
