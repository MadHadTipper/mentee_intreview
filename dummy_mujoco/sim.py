"""Dummy MuJoCo-flavored batched simulator: 2-link planar arm reaching a target.

API conventions deliberately mirror MuJoCo's bare-bones style:
- numpy arrays (float32 / int32)
- 4-tuple advance() return: (reward, observation, done, info) — reward FIRST
- caller is responsible for resetting done envs via reset_idx(env_ids)
- physics state exposed as qpos / qvel; sites accessed by name
- ``mj_step(tau)`` is the low-level physics tick taking raw joint torque
- ``advance(ctrl)`` is the high-level RL step interpreting ``ctrl`` as a
  normalized joint *position target* (PD control via the configured
  ``actuator_gainprm`` / ``actuator_biasprm`` arrays, MuJoCo-style)
- Decimation (``physics ticks per advance``) is a SETTABLE attribute on the
  sim, not a constructor arg — divergent with IsaacLabEnv on purpose.

Two distinct kinds of "target":
- ``target_xy()`` — workspace goal the arm tries to reach (RL goal)
- ``set_actuator_targets(qtarget)`` — joint position setpoint consumed by the
  PD controller during the next ``advance()`` / ``mj_step``. Independent.
"""
from __future__ import annotations

import math

import numpy as np

L1: float = 1.0
L2: float = 1.0
DT: float = 0.02
DAMPING: float = 0.98
MAX_STEPS: int = 200             # max RL steps (= advance() calls) per episode
SUCCESS_THRESHOLD: float = 0.05
DEFAULT_DECIMATION: int = 4
WORKSPACE_RADIUS: float = L1 + L2

# MuJoCo's general actuator: force = gainprm[0] * ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel
# We map our PD (kp, kd) to that representation, with ctrl interpreted as a
# *normalized* target in [-1, 1] which scales to [-pi, pi] before being fed to
# the actuator equation.
DEFAULT_KP: float = 18.0
DEFAULT_KD: float = 1.5
DEFAULT_MAX_TORQUE: float = 4.0


def _wrap_to_pi(q: np.ndarray) -> np.ndarray:
    return ((q + math.pi) % (2 * math.pi) - math.pi).astype(np.float32)


class MuJoCoSim:
    """Batched 2-link planar arm reaching a goal in xy.

    Args:
        n_parallel: number of parallel envs.
        seed: optional seed for ``numpy.random.default_rng``.

    Settable attributes (runtime-mutable, no constructor args — divergent with
    IsaacLabEnv where these are constructor-fixed):
        decimation:        physics ticks per ``advance()`` call (default 4).
        actuator_gainprm:  shape (2,) float32. PD-style position gain (kp).
        actuator_biasprm:  shape (3,) float32. ``[0, -kp, -kd]`` so the actuator
            equation ``force = gainprm[0]*scaled_ctrl + biasprm[0] +
            biasprm[1]*qpos + biasprm[2]*qvel`` reduces to PD with kp / kd.
            Set kp / kd by mutating these arrays.
        max_torque:        post-actuator torque clip.

    Coordinate / unit notes:
        - ``ctrl`` to ``advance`` is a normalized joint position target in
          ``[-1, 1]`` (scaled internally to ``[-pi, pi]``).
        - ``mj_step(tau)`` takes raw torque (NOT normalized).

    Episode terminates when ``‖ee − target_xy‖ < SUCCESS_THRESHOLD`` or when
    ``t_steps >= MAX_STEPS``. Done envs stay terminal — caller must invoke
    ``reset_idx(env_ids)`` to reset them.
    """

    obs_dim: int = 4
    act_dim: int = 2

    def __init__(
        self,
        n_parallel: int,
        seed: int | None = None,
        *,
        freeze_on_terminal: bool = True,
    ) -> None:
        if n_parallel <= 0:
            raise ValueError(f"n_parallel must be positive, got {n_parallel}")

        self.n_parallel = int(n_parallel)
        self._rng = np.random.default_rng(seed)

        # Whether terminated envs are frozen until ``reset_idx`` is called.
        # Default True (the documented MuJoCo-style semantics). Set False to
        # let an outer wrapper own termination/reset entirely.
        self.freeze_on_terminal = bool(freeze_on_terminal)

        # Settable attributes — runtime knobs.
        self.decimation: int = DEFAULT_DECIMATION
        self.actuator_gainprm: np.ndarray = np.array(
            [DEFAULT_KP, DEFAULT_KP], dtype=np.float32
        )
        # biasprm encodes [0, -kp, -kd]; we replicate per joint (2 joints, 3 prms).
        self.actuator_biasprm: np.ndarray = np.array(
            [
                [0.0, -DEFAULT_KP, -DEFAULT_KD],
                [0.0, -DEFAULT_KP, -DEFAULT_KD],
            ],
            dtype=np.float32,
        )
        self.max_torque: float = DEFAULT_MAX_TORQUE

        self._qpos = np.zeros((self.n_parallel, 2), dtype=np.float32)
        self._qvel = np.zeros((self.n_parallel, 2), dtype=np.float32)
        self._target = np.zeros((self.n_parallel, 2), dtype=np.float32)
        self._actuator_target = np.zeros((self.n_parallel, 2), dtype=np.float32)
        self.last_ctrl = np.zeros((self.n_parallel, 2), dtype=np.float32)
        self._qfrc_actuator = np.zeros((self.n_parallel, 2), dtype=np.float32)
        self.t_steps = np.zeros(self.n_parallel, dtype=np.int32)
        self._terminal = np.zeros(self.n_parallel, dtype=bool)

        self._ctrl_low = np.array([-1.0, -1.0], dtype=np.float32)
        self._ctrl_high = np.array([1.0, 1.0], dtype=np.float32)

        self._closed = False
        self.reset_all()

    # ------------------------------------------------------------------ readers

    def read_qpos(self) -> np.ndarray:
        return self._qpos.copy()

    def read_qvel(self) -> np.ndarray:
        return self._qvel.copy()

    def compute_site_pos(self, name: str) -> np.ndarray:
        if name != "ee":
            raise KeyError(f"unknown site {name!r}; only 'ee' is defined")
        return self._forward_kinematics(self._qpos)

    def target_xy(self) -> np.ndarray:
        return self._target.copy()

    def qfrc_actuator(self) -> np.ndarray:
        """Generalized actuator force applied during the most recent physics tick.

        Shape ``(n_parallel, 2)``, float32. MuJoCo lingo for "the per-joint
        torque produced by the actuator after gain/bias and clipping".
        """
        return self._qfrc_actuator.copy()

    def compute_jacp(self, site_name: str) -> np.ndarray:
        """Positional Jacobian of the named site w.r.t. joint angles.

        Shape ``(n_parallel, 2, 2)``: rows are xy, cols are joints. MuJoCo's
        ``mj_jac`` returns this kind of dense Jacobian for a chosen site.
        """
        if site_name != "ee":
            raise KeyError(f"unknown site {site_name!r}; only 'ee' is defined")
        q1, q2 = self._qpos[:, 0], self._qpos[:, 1]
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        col1 = np.stack([-L1 * s1 - L2 * s12, L1 * c1 + L2 * c12], axis=-1)
        col2 = np.stack([-L2 * s12, L2 * c12], axis=-1)
        return np.stack([col1, col2], axis=-1).astype(np.float32)

    def contact_force(self) -> np.ndarray:
        """Pseudo-contact normal-force signal at the workspace boundary.

        Shape ``(n_parallel,)``, float32. The arm has finite reach
        ``L1 + L2``; we report a soft-constraint force ``max(0, ‖ee‖ − 0.99 R)``
        so the candidate has a contact-like signal to put in their reward.
        """
        ee = self._forward_kinematics(self._qpos)
        r = np.linalg.norm(ee, axis=-1)
        return np.maximum(0.0, r - 0.99 * WORKSPACE_RADIUS).astype(np.float32)

    @property
    def ctrl_range(self) -> tuple[np.ndarray, np.ndarray]:
        return self._ctrl_low, self._ctrl_high

    # ------------------------------------------------------------------ writers

    def write_qpos(self, qpos: np.ndarray) -> None:
        if qpos.shape != (self.n_parallel, 2):
            raise ValueError(f"qpos must have shape ({self.n_parallel}, 2), got {qpos.shape}")
        self._qpos = _wrap_to_pi(np.asarray(qpos, dtype=np.float32).copy())

    def write_qvel(self, qvel: np.ndarray) -> None:
        if qvel.shape != (self.n_parallel, 2):
            raise ValueError(f"qvel must have shape ({self.n_parallel}, 2), got {qvel.shape}")
        self._qvel = np.asarray(qvel, dtype=np.float32).copy()

    def write_target(self, target: np.ndarray) -> None:
        """Set the workspace goal (per-env xy)."""
        if target.shape != (self.n_parallel, 2):
            raise ValueError(f"target must have shape ({self.n_parallel}, 2), got {target.shape}")
        self._target = np.asarray(target, dtype=np.float32).copy()

    def set_actuator_targets(self, qtarget: np.ndarray) -> None:
        """Set the joint position target consumed by the PD actuator.

        Distinct from ``write_target``: this is the *motor* setpoint, in radians,
        which the next ``mj_step`` (called via ``advance``) will use as the
        position reference.
        """
        if qtarget.shape != (self.n_parallel, 2):
            raise ValueError(
                f"qtarget must have shape ({self.n_parallel}, 2), got {qtarget.shape}"
            )
        self._actuator_target = _wrap_to_pi(np.asarray(qtarget, dtype=np.float32).copy())

    # ------------------------------------------------------------------ low-level

    def mj_step(self, tau: np.ndarray | None = None) -> None:
        """One physics tick.

        Args:
            tau: optional ``(n_parallel, 2)`` raw torque override. If ``None``
                (default), the current actuator targets are evaluated through
                the gainprm/biasprm equation:

                    qfrc = gainprm[0] * scaled_target + biasprm[0]
                           + biasprm[1] * qpos + biasprm[2] * qvel

                where ``scaled_target = actuator_target / pi`` (so ``ctrl`` is
                effectively the position target normalized into the gain). With
                the default biasprm ``[0, -kp, -kd]`` and gainprm ``[kp]`` this
                reduces to PD: ``kp*(target - qpos*pi/pi) - kd*qvel``.

        Does NOT advance ``t_steps`` or compute reward — those are
        ``advance()``'s job. Frozen terminal envs are skipped.
        """
        active = ~self._terminal
        if not active.any():
            return

        if tau is None:
            scaled_target = self._actuator_target  # already in radians
            # Per-joint PD via actuator equation:
            #   qfrc = gainprm * (target / pi) + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel
            # We want target to be in radians and biasprm[1] = -kp so the result is
            # kp*target - kp*qpos - kd*qvel. So we scale gainprm contribution as
            # gainprm * target (treating gainprm as kp directly).
            qfrc = (
                self.actuator_gainprm * scaled_target
                + self.actuator_biasprm[:, 0]
                + self.actuator_biasprm[:, 1] * self._qpos
                + self.actuator_biasprm[:, 2] * self._qvel
            )
        else:
            if tau.shape != (self.n_parallel, 2):
                raise ValueError(f"tau must have shape ({self.n_parallel}, 2), got {tau.shape}")
            qfrc = np.asarray(tau, dtype=np.float32)

        qfrc = np.clip(qfrc, -self.max_torque, self.max_torque).astype(np.float32)
        self._qfrc_actuator = qfrc

        # Integrate only active envs.
        self._qvel[active] = DAMPING * self._qvel[active] + qfrc[active] * DT
        self._qpos[active] = _wrap_to_pi(self._qpos[active] + self._qvel[active] * DT)

    # ------------------------------------------------------------------ core API

    def reset_all(self) -> np.ndarray:
        self.reset_idx(np.arange(self.n_parallel, dtype=np.int64))
        return self._build_obs()

    def reset_idx(self, env_ids: np.ndarray) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return
        n = env_ids.size
        new_q = self._rng.uniform(-math.pi, math.pi, size=(n, 2)).astype(np.float32)
        self._qpos[env_ids] = new_q
        self._qvel[env_ids] = 0.0
        rand_q = self._rng.uniform(-math.pi, math.pi, size=(n, 2)).astype(np.float32)
        self._target[env_ids] = self._forward_kinematics(rand_q)
        self._actuator_target[env_ids] = new_q.copy()  # PD doesn't slam at reset
        self.last_ctrl[env_ids] = 0.0
        self._qfrc_actuator[env_ids] = 0.0
        self.t_steps[env_ids] = 0
        self._terminal[env_ids] = False

    def advance(
        self, ctrl: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """One RL step (= ``decimation`` physics ticks).

        Args:
            ctrl: shape ``(n_parallel, 2)`` numpy array. Interpreted as a
                normalized joint position target in ``[-1, 1]`` (scaled to
                ``[-pi, pi]``). Clipped if out of range.

        Returns 4-tuple ``(reward, observation, done, info)`` — reward FIRST.
        """
        if self._closed:
            raise RuntimeError("advance() called on a shutdown MuJoCoSim")
        if not isinstance(ctrl, np.ndarray):
            raise TypeError(f"ctrl must be np.ndarray, got {type(ctrl).__name__}")
        if ctrl.shape != (self.n_parallel, self.act_dim):
            raise ValueError(
                f"ctrl must have shape ({self.n_parallel}, {self.act_dim}), got {ctrl.shape}"
            )

        ctrl = np.clip(np.asarray(ctrl, dtype=np.float32), self._ctrl_low, self._ctrl_high)
        self.last_ctrl = ctrl.copy()
        # Scale [-1, 1] → [-pi, pi] joint position targets.
        self.set_actuator_targets(ctrl * math.pi)

        if self.freeze_on_terminal:
            active = ~self._terminal
        else:
            active = np.ones(self.n_parallel, dtype=bool)

        if active.any():
            for _ in range(self.decimation):
                self.mj_step()  # uses actuator_target via the gainprm/biasprm equation
            self.t_steps[active] += 1

        ee = self._forward_kinematics(self._qpos)
        dist = np.linalg.norm(ee - self._target, axis=-1)
        if self.freeze_on_terminal:
            reward = np.where(self._terminal, 0.0, -dist).astype(np.float32)
        else:
            reward = (-dist).astype(np.float32)

        success = dist < SUCCESS_THRESHOLD
        timeout = self.t_steps >= MAX_STEPS
        if self.freeze_on_terminal:
            done_now = (success | timeout) & ~self._terminal
            self._terminal = self._terminal | done_now
            done = self._terminal.copy()
        else:
            done = (success | timeout)

        return reward, self._build_obs(), done, {}

    def shutdown(self) -> None:
        self._closed = True

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _forward_kinematics(q: np.ndarray) -> np.ndarray:
        q1, q2 = q[..., 0], q[..., 1]
        x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
        y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
        return np.stack([x, y], axis=-1).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        return np.concatenate([self._qpos, self._qvel], axis=-1).astype(np.float32)
