# `ai_dummy_sim` — interview exercise

Two deliberately divergent batched simulators (Isaac-Lab-flavored, MuJoCo-flavored)
plus the scaffolding for an AI engineer interview exercise: build a unified RL
training layer over both.

> ⚠️ Working on the exercise? Open **`INTERVIEW.md`**. This file is just the
> repo overview / install.

## Quickstart

The recommended install path uses [`uv`](https://github.com/astral-sh/uv) — it's
fast, deterministic across OSes, and avoids most pip-with-venv friction.

### 1. Install `uv` (one-time, machine-wide)

```sh
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. From the repo root

```sh
uv sync --extra rl
```

This creates `.venv/` and installs everything. ~30–60 seconds the first time
on a laptop. Re-running it is near-instant. The committed `uv.lock` pins
exact versions per platform; CPU torch is pulled from PyTorch's CPU-only
index automatically — no GPU needed, no CUDA wheel downloaded by accident.

### 3. Verify the install

```sh
uv run pytest tests/
uv run python examples/random_policy_isaaclab.py
uv run python examples/random_policy_mujoco.py
```

`pytest tests/` should finish in well under a second with everything green.
The examples should each print reward stats and exit cleanly.

### 4. Read the task

Open `INTERVIEW.md` for the leveled task description and the per-level
verification commands.

## Fallback install (no `uv`)

```sh
python -m venv .venv
# POSIX:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
pip install -e ".[rl]"
pytest tests/
```

Slower; no cross-OS lockfile guarantee; works.

## Repo layout

```
ai_dummy_sim/
├── README.md                  ← you are here
├── INTERVIEW.md               ← orientation hub: pacing, world, vocab, eval, level index
├── interview/                 ← one file per level (level_1.md … level_5.md)
├── pyproject.toml             ← deps (uv-first; pip-compatible)
├── uv.lock                    ← committed, deterministic install
│
├── dummy_isaaclab/            ← torch-backed batched arm sim + per-package README
├── dummy_mujoco/              ← numpy-backed batched arm sim + per-package README
├── examples/                  ← random-policy driver per sim — read these first
├── tests/                     ← contract tests for each sim (also living docs)
│
├── solution/                  ← YOU FILL THIS IN — empty stubs for each level
└── verify/                    ← per-level executable checks (your self-grade)
```

## Supported platforms

Tested on Linux (Ubuntu 22.04), macOS 14 (Apple Silicon + Intel), Windows 11.
Python 3.10–3.12. CPU-only — no GPU required.

## Troubleshooting

- **`uv: command not found`** after install → restart your shell (or run
  `source ~/.bashrc` / equivalent) so the new PATH entry is picked up.
- **`pytest tests/` collected 0 items** → make sure you ran the previous step
  inside the repo root (where `pyproject.toml` lives), and that `uv sync`
  finished without errors.
- **Torch wheel huge / slow** → `uv` should pull a ~200MB CPU build. If you
  see CUDA wheels (`+cu121`), open an issue — `pyproject.toml` is misconfigured.
- **`No module named dummy_isaaclab`** → run via `uv run python …` (not bare
  `python`); `uv run` activates the project venv.
