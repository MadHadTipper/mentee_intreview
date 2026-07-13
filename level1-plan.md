# Level 1 — Unified env adapter + unified state

## Context

Interview exercise: wrap two API-divergent batched sims (`dummy_isaaclab`,
torch-backed, gym-style 5-tuple, auto-reset; `dummy_mujoco`, numpy-backed,
4-tuple reward-first, caller-managed reset) behind one Python interface, plus
expose a per-step unified state object with fixed attribute names that Level
2's sim-agnostic reward/obs/termination terms will read from. Entry point is
`solution/env.py::make_env`, verified by `verify/level_1.py` (API smoke only;
the hidden grader checks cross-sim numeric equality on matched states — which
`verify/` deliberately does not check, so we need our own self-check for it).

Design decisions settled during planning discussion:
- **Optional state fields** (`applied_torque`, `contact_force`) are **out of
  scope for now** — add later only if a level actually needs them.
- **Level 1's own obs** is a **unified 6-dim projection**
  `[joint_pos(2), joint_vel(2), goal-ee_pos(2)]`, computed identically from
  `UnifiedState` for both backends (`obs_dim == 6` regardless of sim) — this
  doubles as the obs term Level 2 will default to.
- **`make_env`'s signature is copied verbatim from the `solution/env.py`
  stub** — `def make_env(name: str, num_envs: int, *, seed: int | None =
  None):`, no signature edits, no added `Literal`/return-type annotations.
  This applies to every entry-point signature the `solution/` stubs
  already declare, not just this one — implementation goes in function
  bodies, signatures never move. Validation happens *inside* the function
  body: `SimName(name)` (a `str, Enum`, docstring-marked as an internal
  validation-only helper, not part of the contract) both validates against
  the hard-coded closed set (raises `ValueError` natively for anything
  else) and dispatches to the right adapter class, replacing an open-ended
  dict/string-branching lookup.
- **Internal (non-public-contract) return types are small named objects,
  not positional tuples — and pydantic, not stdlib `dataclasses.dataclass`,
  same reasoning as `UnifiedState`.** `SimAdapter.step()` returns a
  `StepResult` pydantic dataclass (`state`, `reward`, `terminated`,
  `truncated`) instead of a bare 4-tuple — it's an adapter-internal detail
  we own, so `result.state` beats positional unpacking. This does **not**
  apply to `UnifiedEnv.step()` / `.reset()` — those keep returning the
  exact `(obs, reward, done, info)` tuple the Level 1 contract specifies,
  since that's a supplied contract, not our internal plumbing.
- **Workflow**: edits will be applied one at a time for manual review (no
  batch auto-accept). A self-check test script gets written and run against
  each adapter as it's built, before moving to the next piece — not all
  code first, tests after.
- **`UnifiedState` is a pydantic dataclass, not a stdlib `@dataclass`.**
  This is the one class whose whole job is "guarantee it holds the right
  objects" — required attribute names/shapes/dtypes are exactly what the
  Level 1 contract and the hidden grader check — so it gets field
  validators that raise immediately on a wrong shape/dtype instead of
  silently producing a malformed state Level 2 would choke on later.
  `SimAdapter`/`IsaacLabAdapter`/`MuJoCoAdapter`/`UnifiedEnv` stay plain
  classes — they wrap mutable engine/sim objects and behavior, not
  validated data, so pydantic doesn't fit there. Requires adding
  `pydantic>=2` to `pyproject.toml` (not currently a dependency).

## File layout

Everything in **`solution/env.py`** (single file, ~180-220 lines):

```
UnifiedState          — dataclass, the per-step state object
SimAdapter             — ABC: reset()/step()/close(), owns raw sim as self._sim
IsaacLabAdapter(SimAdapter)
MuJoCoAdapter(SimAdapter)
StepResult              — plain dataclass: state, reward, terminated, truncated (adapter-internal, not the public contract)
_build_obs(state)       — module-level helper, the unified 6-dim projection
UnifiedEnv              — the object make_env() returns; wraps adapter as self.adapter
SimName(str, Enum)     — internal validation-only: {ISAACLAB="isaaclab", MUJOCO="mujoco"}, .adapter_cls dispatches via match
make_env(name, num_envs, *, seed=None)
```

Naming `self.adapter` / `self._sim` is deliberate: `verify/conftest.py`'s
`_walk_for` (used by matched-state test helpers) recursively searches for the
raw `IsaacLabEnv`/`MuJoCoSim` under attribute names `unwrapped, _env, _sim,
env, sim, adapter, _adapter, inner` — `UnifiedEnv.adapter._sim` satisfies
that walk directly, so `verify/conftest.py`'s `set_isaaclab_state` /
`set_mujoco_state` helpers (and our own self-check script, same technique)
can reach into either backend without special-casing.

## Self-check tests (write + run these *before* trusting each adapter)

`verify/level_1.py` only checks shapes/dtypes/attribute presence on one
sim at a time — it will never catch "IsaacLabAdapter and MuJoCoAdapter
disagree on the resulting `ee_pos` given the same joint state and action."
That's exactly what the hidden grader is expected to probe, so we write our
own script for it first and run it after each adapter lands.

**Location**: `$CLAUDE_JOB_DIR/tmp/level1_selfcheck.py` — a scratch pytest
file, not part of the graded `solution/` deliverable, not touching
`tests/`/`verify/`. Run with `uv run pytest <path> -q` from the repo root.

Planned cases:

1. **`test_shapes_and_dtypes[sim]`** (parametrized isaaclab/mujoco) — reset +
   a few random-action steps, assert `obs.shape/dtype`,
   `state.<each required attr>.shape/dtype` match the contract. (Belt-and-
   suspenders on top of `verify/level_1.py`, cheap to have standalone.)

2. **`test_reset_all_zero_episode_step`** — after `reset()`,
   `state.episode_step` is all-zero for both sims.

3. **`test_cross_sim_matched_state_one_step`** — the important one. Using
   `verify/conftest.py`'s own `matched_state_inputs` fixture values (q, qd,
   goal for 4 envs), reach into `env.adapter._sim` for both backends via the
   same `set_isaaclab_state`/`set_mujoco_state` pattern conftest uses, step
   both envs with an **identical zero action** (isolates physics/PD
   translation from action-scaling bugs), then assert
   `np.allclose(state_a.joint_pos, state_b.joint_pos, atol=1e-4)` and same
   for `joint_vel`, `ee_pos`. This is the test that would have caught, e.g.,
   a PD-gain or `_wrap_to_pi` translation mistake in one adapter but not the
   other.

4. **`test_cross_sim_matched_state_nonzero_action`** — same as #3 but with a
   fixed non-trivial action (e.g. `[[0.3, -0.2], ...]` per env) to also
   exercise the action-scaling path identically on both backends.

5. **`test_auto_reset_semantics[sim]`** — drive an env with an action chosen
   to reach the goal quickly (or just run `MAX_STEPS` steps with a zero
   action to force truncation), assert that on the step where `done[i]`
   first goes `True`, the *same* `step()` call's returned `state.episode_step[i]
   == 0` (fresh post-reset state in the same call, matching IsaacLab's
   auto-reset contract) — this is the "reset is the trap" behavior and the
   easiest thing to get subtly wrong on the MuJoCo side (forgetting to
   `reset_idx` before building the returned snapshot).

6. **`test_seed_determinism`** — two `make_env(sim, N, seed=0)` instances,
   same action sequence, assert identical `obs` trajectories — cheap check
   that seeding is actually wired through.

I'll write this file first (it will fail against the `NotImplementedError`
stub, which is expected — same TDD shape as `verify/`), then implement
`solution/env.py` piece by piece, running it after each adapter.

## Detailed Python plan

```python
# solution/env.py
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from dummy_isaaclab import IsaacLabEnv
from dummy_mujoco import MuJoCoSim

_NP_CONFIG = ConfigDict(arbitrary_types_allowed=True)


@pydantic_dataclass(config=_NP_CONFIG)
class UnifiedState:
    """Per-step state snapshot, identical shape/attrs/dtypes regardless of backend."""
    joint_pos: np.ndarray     # (N, 2) float32
    joint_vel: np.ndarray     # (N, 2) float32
    ee_pos: np.ndarray        # (N, 2) float32
    goal: np.ndarray          # (N, 2) float32
    episode_step: np.ndarray  # (N,)   int64
    last_action: np.ndarray   # (N, 2) float32

    @field_validator("joint_pos", "joint_vel", "ee_pos", "goal", "last_action")
    @classmethod
    def _check_coord_field(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"expected shape (N, 2), got {v.shape}")
        if not np.issubdtype(v.dtype, np.floating):
            raise ValueError(f"expected floating dtype, got {v.dtype}")
        return v

    @field_validator("episode_step")
    @classmethod
    def _check_episode_step(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            raise ValueError(f"expected shape (N,), got {v.shape}")
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError(f"expected integer dtype, got {v.dtype}")
        return v


@pydantic_dataclass(config=_NP_CONFIG)
class StepResult:
    """One adapter step's outputs. Adapter-internal — not the public env contract,
    so it's a plain object rather than a positional tuple to unpack."""

    state: UnifiedState
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class SimAdapter(ABC):
    """Internal contract both backend adapters implement."""

    num_envs: int
    obs_dim: int = 6
    act_dim: int = 2

    @abstractmethod
    def reset(self) -> UnifiedState: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """done is OR'd from `.terminated`/`.truncated` by UnifiedEnv."""
        ...

    @abstractmethod
    def close(self) -> None: ...


class IsaacLabAdapter(SimAdapter):
    """Adapts IsaacLabEnv (torch, 5-tuple, auto-reset) to SimAdapter."""

    def __init__(self, num_envs: int, seed: int | None) -> None:
        self.num_envs = num_envs
        self._sim = IsaacLabEnv(num_envs, seed=seed)

    def _snapshot(self) -> UnifiedState:
        s = self._sim
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float32, copy=True)
        return UnifiedState(
            joint_pos=to_np(s.joint_pos),
            joint_vel=to_np(s.joint_vel),
            ee_pos=to_np(s.ee_pos),
            goal=to_np(s.goal_pos),
            episode_step=s.episode_step.detach().cpu().numpy().astype(np.int64, copy=True),
            last_action=to_np(s.last_action),
        )

    def reset(self) -> UnifiedState:
        self._sim.reset()
        return self._snapshot()

    def step(self, action: np.ndarray) -> StepResult:
        a = torch.from_numpy(np.asarray(action, dtype=np.float32))
        _, reward, terminated, truncated, _ = self._sim.step(a)
        return StepResult(
            state=self._snapshot(),
            reward=reward.detach().cpu().numpy().astype(np.float32),
            terminated=terminated.detach().cpu().numpy(),
            truncated=truncated.detach().cpu().numpy(),
        )

    def close(self) -> None:
        self._sim.close()


class MuJoCoAdapter(SimAdapter):
    """Adapts MuJoCoSim (numpy, 4-tuple reward-first, caller-managed reset)."""

    def __init__(self, num_envs: int, seed: int | None) -> None:
        self.num_envs = num_envs
        self._sim = MuJoCoSim(n_parallel=num_envs, seed=seed)

    def _snapshot(self) -> UnifiedState:
        s = self._sim
        return UnifiedState(
            joint_pos=s.read_qpos(),
            joint_vel=s.read_qvel(),
            ee_pos=s.compute_site_pos("ee"),
            goal=s.target_xy(),
            episode_step=s.t_steps.astype(np.int64, copy=True),
            last_action=s.last_ctrl.astype(np.float32, copy=True),
        )

    def reset(self) -> UnifiedState:
        self._sim.reset_all()
        return self._snapshot()

    def step(self, action: np.ndarray) -> StepResult:
        ctrl = np.asarray(action, dtype=np.float32)
        reward, _, done, _ = self._sim.advance(ctrl)
        if done.any():
            self._sim.reset_idx(np.flatnonzero(done))  # THE reset trap
        return StepResult(
            state=self._snapshot(),  # reflects post-reset for done envs, matching IsaacLab
            reward=reward.astype(np.float32),
            terminated=done,
            truncated=np.zeros_like(done),
        )

    def close(self) -> None:
        self._sim.shutdown()


def _build_obs(state: UnifiedState) -> np.ndarray:
    """Unified obs = [joint_pos, joint_vel, goal - ee_pos], same for both backends."""
    return np.concatenate(
        [state.joint_pos, state.joint_vel, state.goal - state.ee_pos], axis=-1
    ).astype(np.float32)


class UnifiedEnv:
    """Batched env wrapping either backend behind one reset/step/close API."""

    def __init__(self, adapter: SimAdapter) -> None:
        self.adapter = adapter
        self.num_envs = adapter.num_envs
        self.obs_dim = adapter.obs_dim
        self.act_dim = adapter.act_dim
        self.state: UnifiedState | None = None

    def reset(self) -> np.ndarray:
        self.state = self.adapter.reset()
        return _build_obs(self.state)

    def step(self, action: np.ndarray):
        result = self.adapter.step(action)
        self.state = result.state
        done = result.terminated | result.truncated
        return _build_obs(self.state), result.reward, done, {}

    def close(self) -> None:
        self.adapter.close()


class SimName(str, Enum):
    """Internal validation-only helper — not part of the public contract.
    Closed set of valid `make_env` backends; subclasses `str` so it compares
    equal to the plain strings the contract/verify suite passes in."""

    ISAACLAB = "isaaclab"
    MUJOCO = "mujoco"

    @property
    def adapter_cls(self) -> type[SimAdapter]:
        match self:
            case SimName.ISAACLAB:
                return IsaacLabAdapter
            case SimName.MUJOCO:
                return MuJoCoAdapter


def make_env(name: str, num_envs: int, *, seed: int | None = None):
    """Construct a unified batched env wrapping the named dummy sim.

    Args:
        name: ``"isaaclab"`` or ``"mujoco"``.
        num_envs: parallel batch size.
        seed: optional seed forwarded to the underlying sim.
    """
    sim_name = SimName(name)  # raises ValueError for anything outside the closed set
    return UnifiedEnv(sim_name.adapter_cls(num_envs, seed))
```

Signature is character-for-character what `solution/env.py`'s stub already
declares — every `solution/`-supplied entry-point signature stays untouched;
only bodies change. Validation (`SimName(name)`) lives inside the function,
and `SimName` itself is marked internal/validation-only via its docstring
(no leading underscore — kept as a plain name per feedback).

Open items to double check once code is running (not blocking, just flagged):
- Confirm `IsaacLabEnv.episode_step` / `terminated` / `truncated` tensors
  don't require an explicit `.long()`/bool cast quirk when calling `.numpy()`
  directly on CPU int64/bool tensors (should be fine, but verify in the
  self-check run).
- `MuJoCoSim.advance` requires a real `np.ndarray` — `np.asarray(action,
  dtype=np.float32)` covers both a caller passing a list and a non-contiguous
  array.

## Implementation order

1. Add `pydantic>=2` to `pyproject.toml` `[project].dependencies`, `uv sync`.
2. Write `$CLAUDE_JOB_DIR/tmp/level1_selfcheck.py` (fails against the
   `NotImplementedError` stub — expected).
3. Implement `UnifiedState` (pydantic dataclass + validators), `SimAdapter`,
   `IsaacLabAdapter` in `solution/env.py`. Run self-check cases 1, 2, 5, 6
   filtered to `sim=isaaclab` only — the pydantic validators should trip
   immediately if a shape/dtype/cast is wrong, before the assertion-based
   checks even run.
4. Implement `MuJoCoAdapter`. Run the same cases filtered to `sim=mujoco`.
5. Implement `_build_obs`, `UnifiedEnv`, `_ADAPTERS`, `make_env`. Run the
   full self-check file, including the cross-sim cases (3, 4).
6. `uv run pytest verify/level_1.py -x -q`, then `uv run pytest verify/ -q`.
7. Save this finalized plan to `level1-plan.md` at the repo root.

## Verification

1. Self-check script (above) — the only place cross-sim numeric agreement
   gets exercised before the hidden grader does.
2. `uv run pytest verify/level_1.py -x -q` — must pass.
3. `uv run pytest verify/ -q` — full suite sanity (other levels still
   correctly raise `NotImplementedError`, no import breakage).
