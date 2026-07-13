"""Level 1 deliverable — unified env adapter + unified state.

See ``interview/level_1.md`` for the full contract & verification.
Run ``pytest verify/level_1.py`` after implementing — it must pass.
"""
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
    """Per-step state snapshot; identical attribute names/shapes/dtypes regardless of backend."""

    joint_pos: np.ndarray  # (N, 2) float32
    joint_vel: np.ndarray  # (N, 2) float32
    ee_pos: np.ndarray  # (N, 2) float32
    goal: np.ndarray  # (N, 2) float32
    episode_step: np.ndarray  # (N,) int64
    last_action: np.ndarray  # (N, 2) float32

    @field_validator("joint_pos", "joint_vel", "ee_pos", "goal", "last_action")
    @classmethod
    def _check_coord_field(cls, v: np.ndarray) -> np.ndarray:
        """Enforce shape (N, 2) float on every per-coordinate state field."""
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"expected shape (N, 2), got {v.shape}")
        if not np.issubdtype(v.dtype, np.floating):
            raise ValueError(f"expected floating dtype, got {v.dtype}")
        return v

    @field_validator("episode_step")
    @classmethod
    def _check_episode_step(cls, v: np.ndarray) -> np.ndarray:
        """Enforce shape (N,) integer on episode_step."""
        if v.ndim != 1:
            raise ValueError(f"expected shape (N,), got {v.shape}")
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError(f"expected integer dtype, got {v.dtype}")
        return v


@pydantic_dataclass(config=_NP_CONFIG)
class StepResult:
    """One adapter step's outputs; adapter-internal, not the public env contract."""

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
    def reset(self) -> UnifiedState:
        """Reset every env and return the fresh unified state."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Advance one RL step; ``done`` is OR'd from terminated/truncated by UnifiedEnv."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Tear down the underlying sim."""
        ...


class IsaacLabAdapter(SimAdapter):
    """Adapts IsaacLabEnv (torch, 5-tuple step, auto-reset) to SimAdapter."""

    def __init__(self, num_envs: int, seed: int | None) -> None:
        self.num_envs = num_envs
        self._sim = IsaacLabEnv(num_envs, seed=seed)

    def _snapshot(self) -> UnifiedState:
        """Pull the sim's live torch views into a copied, numpy UnifiedState."""
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
    """Adapts MuJoCoSim (numpy, 4-tuple reward-first advance, caller-managed reset) to SimAdapter."""

    def __init__(self, num_envs: int, seed: int | None) -> None:
        self.num_envs = num_envs
        self._sim = MuJoCoSim(n_parallel=num_envs, seed=seed)

    def _snapshot(self) -> UnifiedState:
        """Read the sim's already-copied numpy state into a UnifiedState."""
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
    """Unified obs = [joint_pos, joint_vel, goal - ee_pos], identical for both backends."""
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
        """Reset every env; returns obs (num_envs, obs_dim) float32."""
        self.state = self.adapter.reset()
        return _build_obs(self.state)

    def step(self, action: np.ndarray):
        """One RL step; returns (obs, reward, done, info)."""
        result = self.adapter.step(action)
        self.state = result.state
        done = result.terminated | result.truncated
        return _build_obs(self.state), result.reward, done, {}

    def close(self) -> None:
        """Tear down the underlying sim."""
        self.adapter.close()


class SimName(str, Enum):
    """Internal validation-only helper — not part of the public contract.

    Closed set of valid ``make_env`` backends; subclasses ``str`` so it
    compares equal to the plain strings the contract/verify suite passes in.
    """

    ISAACLAB = "isaaclab"
    MUJOCO = "mujoco"

    @property
    def adapter_cls(self) -> type[SimAdapter]:
        """Dispatch to the adapter class for this backend."""
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

    Returns:
        An object meeting the unified env contract documented above
        (including the unified state object — see module docstring).
    """
    sim_name = SimName(name)  # raises ValueError for anything outside the closed set
    return UnifiedEnv(sim_name.adapter_cls(num_envs, seed))
