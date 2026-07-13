"""Dummy Isaac-Lab-flavored batched simulator: 2-link planar arm reaching a target.

API conventions deliberately mirror Isaac Lab:
- torch tensors on a configurable device
- gym-style 5-tuple step() return: (obs, reward, terminated, truncated, info)
- auto-resets done envs inside step()
- joint_pos / joint_vel / ee_pos / goal_pos exposed as properties
- step() takes a *normalized joint position target* in [-1, 1] and runs
  ``decimation`` internal physics ticks driven by an internal PD controller
  (configurable via ``kp`` / ``kd`` attributes)
- low-level physics_step(torque) is exposed separately for direct torque control

Two distinct kinds of "target":
- ``goal_pos`` — the workspace xy point the arm is trying to reach (RL goal)
- ``joint_pos_target`` — the per-step joint position setpoint fed to the PD
  controller (the *motor* target). These are independent.
"""
from __future__ import annotations

import math

import torch

L1: float = 1.0
L2: float = 1.0
DT: float = 0.02              # physics tick
DAMPING: float = 0.98          # joint-velocity damping per physics tick
MAX_STEPS: int = 200           # max RL steps per episode (truncation)
SUCCESS_THRESHOLD: float = 0.05
DEFAULT_DECIMATION: int = 4    # physics ticks per step()
DEFAULT_KP: float = 18.0
DEFAULT_KD: float = 1.5
DEFAULT_MAX_TORQUE: float = 4.0
WORKSPACE_RADIUS: float = L1 + L2  # 2.0


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


class IsaacLabEnv:
    """Batched 2-link planar arm reaching a goal in xy.

    Args:
        num_envs: parallel batch size.
        device: torch device. Defaults to ``"cpu"``.
        seed: optional seed for the env's torch.Generator.
        decimation: physics ticks per ``step()``. Default 4. Constructor-fixed
            (cannot be changed after construction — Isaac-Lab style).
        control_mode: ``"position"`` (default) or ``"torque"``. In position mode
            actions are normalized joint position targets; in torque mode they
            are normalized direct joint torques.

    Tunables exposed as plain attributes (settable at runtime):
        kp, kd:      PD gains used by ``step`` in position mode.
        max_torque:  Clip applied to torque before ``physics_step``.

    Coordinate / unit notes:
        - Action range is always ``[-1, 1]`` per joint (regardless of mode).
        - In position mode an action ``a`` is unscaled internally to a target
          angle ``a * pi``.
        - In torque mode an action ``a`` is unscaled to a torque ``a * max_torque``.

    Episode terminates when ``‖ee − goal‖ < SUCCESS_THRESHOLD`` (terminated)
    or when ``episode_step >= MAX_STEPS`` (truncated). Done envs auto-reset
    in-place at the end of ``step()``; the returned obs reflects the post-reset
    state for those envs.
    """

    obs_dim: int = 6
    act_dim: int = 2

    def __init__(
        self,
        num_envs: int,
        device: str = "cpu",
        seed: int | None = None,
        decimation: int = DEFAULT_DECIMATION,
        control_mode: str = "position",
        auto_reset: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if decimation <= 0:
            raise ValueError(f"decimation must be positive, got {decimation}")
        if control_mode not in ("position", "torque"):
            raise ValueError(f"control_mode must be 'position' or 'torque', got {control_mode!r}")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.decimation = int(decimation)
        self.control_mode = control_mode
        self.auto_reset = bool(auto_reset)

        # PD gains and torque clip — runtime-mutable.
        self.kp: float = DEFAULT_KP
        self.kd: float = DEFAULT_KD
        self.max_torque: float = DEFAULT_MAX_TORQUE

        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        z2 = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._joint_pos = z2.clone()
        self._joint_vel = z2.clone()
        self._goal = z2.clone()
        self._joint_pos_target = z2.clone()
        self._last_action = z2.clone()
        self._applied_torque = z2.clone()
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        self._action_low = torch.tensor([-1.0, -1.0], dtype=torch.float32, device=self.device)
        self._action_high = torch.tensor([1.0, 1.0], dtype=torch.float32, device=self.device)

        self._closed = False
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------ properties

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def ee_pos(self) -> torch.Tensor:
        return self._forward_kinematics(self._joint_pos)

    @property
    def goal_pos(self) -> torch.Tensor:
        return self._goal

    @property
    def joint_pos_target(self) -> torch.Tensor:
        """Last joint position target driven by the PD controller (radians)."""
        return self._joint_pos_target

    @property
    def applied_torque(self) -> torch.Tensor:
        """Torque applied during the most recent ``physics_step`` (post-clip)."""
        return self._applied_torque

    @property
    def last_action(self) -> torch.Tensor:
        return self._last_action

    @property
    def episode_step(self) -> torch.Tensor:
        return self._episode_step

    @property
    def action_space_low(self) -> torch.Tensor:
        return self._action_low

    @property
    def action_space_high(self) -> torch.Tensor:
        return self._action_high

    @property
    def workspace_violation(self) -> torch.Tensor:
        """Bool tensor (num_envs,): True if ee at the workspace boundary.

        Pseudo-contact signal — the arm has finite reach (L1 + L2) so the ee
        sits on the boundary when both joints are stretched. We flag envs where
        ``‖ee‖ > 0.99 * (L1 + L2)``.
        """
        ee = self.ee_pos
        return torch.linalg.norm(ee, dim=-1) > (0.99 * WORKSPACE_RADIUS)

    # ------------------------------------------------------------------ low-level

    def physics_step(self, torque: torch.Tensor) -> None:
        """One physics tick driven by raw joint torque.

        Updates ``joint_pos`` / ``joint_vel`` / ``applied_torque`` in place. Does
        NOT advance ``episode_step``, compute reward, or auto-reset. Use this
        for custom integration loops; ``step()`` calls this internally
        ``decimation`` times.

        Args:
            torque: shape ``(num_envs, 2)``. Clipped to ``±max_torque``.
        """
        if torque.shape != (self.num_envs, 2):
            raise ValueError(
                f"torque must have shape ({self.num_envs}, 2), got {tuple(torque.shape)}"
            )
        tau = torque.to(device=self.device, dtype=torch.float32)
        tau = torch.clamp(tau, -self.max_torque, self.max_torque)
        self._applied_torque = tau
        self._joint_vel = DAMPING * self._joint_vel + tau * DT
        self._joint_pos = _wrap_to_pi(self._joint_pos + self._joint_vel * DT)

    def set_actuator_targets(self, targets: torch.Tensor) -> None:
        """Set PD position targets (radians) directly.

        These are the motor setpoints used by ``step()`` in position mode.
        Useful if you want to drive the arm with a manual loop of
        ``set_actuator_targets`` + ``physics_step(self._compute_pd_torque())``.
        """
        if targets.shape != (self.num_envs, 2):
            raise ValueError(
                f"targets must have shape ({self.num_envs}, 2), got {tuple(targets.shape)}"
            )
        self._joint_pos_target = _wrap_to_pi(
            targets.to(device=self.device, dtype=torch.float32).clone()
        )

    def _compute_pd_torque(self) -> torch.Tensor:
        err = _wrap_to_pi(self._joint_pos_target - self._joint_pos)
        return self.kp * err - self.kd * self._joint_vel

    def jacobian(self) -> torch.Tensor:
        """Jacobian of ee position w.r.t. joint angles. Shape (num_envs, 2, 2).

        Columns are ∂ee/∂q1, ∂ee/∂q2.
        """
        q1, q2 = self._joint_pos[:, 0], self._joint_pos[:, 1]
        s1, c1 = torch.sin(q1), torch.cos(q1)
        s12, c12 = torch.sin(q1 + q2), torch.cos(q1 + q2)
        # de/dq1 = [-L1*s1 - L2*s12, L1*c1 + L2*c12]
        # de/dq2 = [-L2*s12, L2*c12]
        col1 = torch.stack([-L1 * s1 - L2 * s12, L1 * c1 + L2 * c12], dim=-1)
        col2 = torch.stack([-L2 * s12, L2 * c12], dim=-1)
        return torch.stack([col1, col2], dim=-1)  # (N, 2, 2) — last dim indexes joint

    # ------------------------------------------------------------------ core API

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._build_obs(), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """One RL step (= ``decimation`` physics ticks).

        Args:
            actions: shape ``(num_envs, 2)``, range ``[-1, 1]`` per joint. Clipped
                if out of range. In ``"position"`` mode ``actions * pi`` is the
                target joint angle; in ``"torque"`` mode ``actions * max_torque``
                is the applied torque.

        Returns gym 5-tuple: (obs, reward, terminated, truncated, info).
        """
        if self._closed:
            raise RuntimeError("step() called on a closed IsaacLabEnv")
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"actions must be torch.Tensor, got {type(actions).__name__}")
        if actions.shape != (self.num_envs, self.act_dim):
            raise ValueError(
                f"actions must have shape ({self.num_envs}, {self.act_dim}), got {tuple(actions.shape)}"
            )

        a = actions.to(device=self.device, dtype=torch.float32)
        a = torch.clamp(a, self._action_low, self._action_high)
        self._last_action = a

        if self.control_mode == "position":
            self._joint_pos_target = _wrap_to_pi(a * math.pi)
            for _ in range(self.decimation):
                self.physics_step(self._compute_pd_torque())
        else:  # "torque"
            tau = a * self.max_torque
            for _ in range(self.decimation):
                self.physics_step(tau)

        self._episode_step = self._episode_step + 1

        ee = self._forward_kinematics(self._joint_pos)
        dist = torch.linalg.norm(ee - self._goal, dim=-1)
        reward = -dist

        terminated = dist < SUCCESS_THRESHOLD
        truncated = self._episode_step >= MAX_STEPS
        done = terminated | truncated

        if self.auto_reset:
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if done_ids.numel() > 0:
                self._reset_idx(done_ids)

        return self._build_obs(), reward, terminated, truncated, {}

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset only the specified envs.

        Useful when ``auto_reset=False`` and you (or your wrapper) are managing
        episode boundaries manually.
        """
        if env_ids.dtype != torch.int64:
            env_ids = env_ids.to(dtype=torch.int64)
        self._reset_idx(env_ids.to(device=self.device))

    def close(self) -> None:
        self._closed = True

    # ------------------------------------------------------------------ state I/O

    def get_state(self) -> torch.Tensor:
        return torch.cat([self._joint_pos, self._joint_vel], dim=-1)

    def set_state(self, state: torch.Tensor) -> None:
        if state.shape != (self.num_envs, 4):
            raise ValueError(
                f"state must have shape ({self.num_envs}, 4), got {tuple(state.shape)}"
            )
        s = state.to(device=self.device, dtype=torch.float32)
        self._joint_pos = _wrap_to_pi(s[:, :2].clone())
        self._joint_vel = s[:, 2:].clone()

    def set_joint_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        if q.shape != (self.num_envs, 2):
            raise ValueError(f"q must have shape ({self.num_envs}, 2), got {tuple(q.shape)}")
        if qd.shape != (self.num_envs, 2):
            raise ValueError(f"qd must have shape ({self.num_envs}, 2), got {tuple(qd.shape)}")
        self._joint_pos = _wrap_to_pi(q.to(device=self.device, dtype=torch.float32).clone())
        self._joint_vel = qd.to(device=self.device, dtype=torch.float32).clone()

    def set_target(self, target: torch.Tensor) -> None:
        """Set the workspace goal xy. Shape ``(num_envs, 2)``.

        Note: this is the *RL goal*, distinct from the per-step joint actuator
        target driven by the PD controller (see ``set_actuator_targets``).
        """
        if target.shape != (self.num_envs, 2):
            raise ValueError(
                f"target must have shape ({self.num_envs}, 2), got {tuple(target.shape)}"
            )
        self._goal = target.to(device=self.device, dtype=torch.float32).clone()

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _forward_kinematics(q: torch.Tensor) -> torch.Tensor:
        q1, q2 = q[..., 0], q[..., 1]
        x = L1 * torch.cos(q1) + L2 * torch.cos(q1 + q2)
        y = L1 * torch.sin(q1) + L2 * torch.sin(q1 + q2)
        return torch.stack([x, y], dim=-1)

    def _sample_reachable_goal(self, n: int) -> torch.Tensor:
        rand_q = (
            torch.rand((n, 2), generator=self._gen, device=self.device, dtype=torch.float32)
            * (2 * math.pi)
            - math.pi
        )
        return self._forward_kinematics(rand_q)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        if n == 0:
            return
        new_q = (
            torch.rand((n, 2), generator=self._gen, device=self.device, dtype=torch.float32)
            * (2 * math.pi)
            - math.pi
        )
        self._joint_pos[env_ids] = new_q
        self._joint_vel[env_ids] = 0.0
        self._goal[env_ids] = self._sample_reachable_goal(n)
        self._joint_pos_target[env_ids] = new_q.clone()  # so PD doesn't slam to 0 on first step
        self._last_action[env_ids] = 0.0
        self._applied_torque[env_ids] = 0.0
        self._episode_step[env_ids] = 0

    def _build_obs(self) -> torch.Tensor:
        ee = self._forward_kinematics(self._joint_pos)
        return torch.cat([self._joint_pos, self._joint_vel, self._goal - ee], dim=-1)
