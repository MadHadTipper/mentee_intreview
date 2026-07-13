# `dummy_mujoco` — numpy-backed batched arm sim

A toy batched simulator that mimics MuJoCo's bare-bones API style: numpy
arrays, `qpos`/`qvel` accessors, named sites, **`mj_step`** as the low-level
physics tick, **`advance(ctrl)`** as the high-level RL step (4-tuple,
`reward, observation, done, info` — reward FIRST), **caller-managed reset of
done envs**, **MuJoCo-style actuator** (`gainprm`/`biasprm` arrays) doing PD
position control internally, and **decimation as a settable attribute**.

The world: a planar 2-link arm trying to reach a workspace goal. Same physics
as `dummy_isaaclab`, intentionally different surface.

```python
from dummy_mujoco import MuJoCoSim
import numpy as np

sim = MuJoCoSim(n_parallel=8, seed=0)
sim.decimation = 4                                        # runtime-settable
obs = sim.reset_all()                                     # obs: (8, 4) float32
ctrl = np.zeros((8, 2), dtype=np.float32)                 # joint position targets, normalized
reward, obs, done, info = sim.advance(ctrl)
if done.any():
    sim.reset_idx(np.flatnonzero(done))
sim.shutdown()
```

## Two distinct kinds of "target"

- **Goal target** — workspace `(x, y)` the arm should reach. `sim.target_xy()`
  / `sim.write_target(...)`. Used for reward.
- **Actuator target** — joint position setpoint consumed by the actuator
  during `mj_step` / `advance`. `sim.set_actuator_targets(...)`. Used to drive
  the PD motor.

## Constructor

```python
MuJoCoSim(n_parallel: int, seed: int | None = None)
```

| Arg          | Meaning                                       |
|--------------|-----------------------------------------------|
| `n_parallel` | Number of parallel envs.                      |
| `seed`       | Optional seed for `numpy.random.default_rng`. |

(Yes, the parameter is `n_parallel`, not `num_envs`. Yes, `decimation` is NOT
a constructor arg — it is a runtime-settable attribute. Divergence is the
point of the exercise.)

## Settable attributes (mutate at runtime)

| Attribute            | Type                  | Default                      | Purpose                                                                 |
|----------------------|-----------------------|------------------------------|-------------------------------------------------------------------------|
| `decimation`         | `int`                 | `4`                          | Physics ticks per `advance()`.                                          |
| `actuator_gainprm`   | `np.ndarray (2,) f32` | `[18.0, 18.0]`               | Per-joint position gain (kp).                                           |
| `actuator_biasprm`   | `np.ndarray (2, 3) f32` | `[[0,-18,-1.5],[0,-18,-1.5]]` | Per-joint `[const, qpos-coeff, qvel-coeff]`. Default encodes `-kp`/`-kd` so the actuator equation reduces to PD. |
| `max_torque`         | `float`               | `4.0`                        | Post-actuator torque clip.                                              |
| `last_ctrl`          | `np.ndarray (n,2) f32`| zeros                        | Last `ctrl` passed to `advance` (post-clip). Read-only by convention.   |
| `t_steps`            | `np.ndarray (n,) i32` | zeros                        | RL steps since each env's last reset.                                   |

The actuator equation evaluated by `mj_step` (when no explicit `tau` is
passed):

```
qfrc = gainprm * actuator_target
       + biasprm[:, 0]
       + biasprm[:, 1] * qpos
       + biasprm[:, 2] * qvel
```

then clipped to `±max_torque`. With the default biasprm `[0, -kp, -kd]` and
gainprm `kp` this is plain PD: `kp * (target − qpos) − kd * qvel`.

## State queries (methods return COPIES; mutating them does not affect the sim)

| Method / attribute             | Returns                | Shape           | Notes                                       |
|--------------------------------|------------------------|-----------------|---------------------------------------------|
| `read_qpos()`                  | `np.ndarray` float32   | `(n_parallel, 2)` | Joint angles, wrapped to `[-π, π]`.       |
| `read_qvel()`                  | `np.ndarray` float32   | `(n_parallel, 2)` | Joint velocities.                          |
| `compute_site_pos("ee")`       | `np.ndarray` float32   | `(n_parallel, 2)` | End-effector xy (forward kinematics).      |
| `compute_jacp("ee")`           | `np.ndarray` float32   | `(n_parallel, 2, 2)` | Positional Jacobian of the named site.   |
| `target_xy()`                  | `np.ndarray` float32   | `(n_parallel, 2)` | Workspace **goal** (RL target).            |
| `qfrc_actuator()`              | `np.ndarray` float32   | `(n_parallel, 2)` | Per-joint actuator force from the most recent physics tick. |
| `contact_force()`              | `np.ndarray` float32   | `(n_parallel,)`   | Pseudo-contact: `max(0, ‖ee‖ − 0.99 R)`.   |
| `last_ctrl` *(attribute)*      | `np.ndarray` float32   | `(n_parallel, 2)` | Last `ctrl` passed to `advance`, post-clip.|
| `t_steps` *(attribute)*        | `np.ndarray` int32     | `(n_parallel,)`   | RL steps since each env's last reset.      |
| `ctrl_range` *(property)*      | `tuple[ndarray, ndarray]` | each `(2,)` f32 | `(low, high)` per-control-dim bounds.    |

Other site names than `"ee"` raise `KeyError` from `compute_site_pos` /
`compute_jacp`.

## Setters

| Method                         | Shape             | Notes                                                  |
|--------------------------------|-------------------|--------------------------------------------------------|
| `write_qpos(qpos)`             | `(n_parallel, 2)` | Wrapped to `[-π, π]` on write.                         |
| `write_qvel(qvel)`             | `(n_parallel, 2)` |                                                        |
| `write_target(target)`         | `(n_parallel, 2)` | Workspace **goal** xy.                                 |
| `set_actuator_targets(qtarget)`| `(n_parallel, 2)` | Joint **motor** position setpoint, in radians, wrapped to `[-π, π]`. |

## Core API

### `reset_all() -> obs`
Reset every env. Returns observation `(n_parallel, 4)` float32.

### `reset_idx(env_ids: ndarray) -> None`
Reset just the listed envs (e.g. `np.flatnonzero(done)`). **Required** after
observing `done` from `advance` — see "Gotchas".

### `mj_step(tau: ndarray | None = None) -> None`
One physics tick.

- `tau is None`: actuator targets are evaluated through the gainprm/biasprm
  equation (PD by default). This is the path that `advance()` uses.
- `tau` provided: shape `(n_parallel, 2)`, raw torque override (still
  clipped to `±max_torque`).

Does NOT advance `t_steps` or compute reward. Frozen terminal envs are
skipped.

### `advance(ctrl: ndarray) -> (reward, observation, done, info)`
One RL step (= `decimation` physics ticks).

- `ctrl`: `(n_parallel, 2)`, must be a numpy array. Interpreted as a
  normalized joint position target in `[-1, 1]` (scaled to `[-π, π]`,
  fed to `set_actuator_targets`). Clipped if out of range.

Returns the **4-tuple, reward FIRST**:
- `reward`: `(n_parallel,)` float32. `−‖ee − target_xy‖` for active envs;
  `0` for envs in terminal state (haven't been reset yet).
- `observation`: `(n_parallel, 4)` float32 — `[q1, q2, q̇1, q̇2]`. Note: target
  is **not** in the observation.
- `done`: `(n_parallel,)` bool. `True` for envs that have terminated and not
  yet been reset.
- `info`: dict.

### `shutdown() -> None`
Marks the sim shut down; further `advance()` / `mj_step()` calls raise.

## Constants (importable from `dummy_mujoco.sim`)

| Name                | Value | Meaning                                      |
|---------------------|-------|----------------------------------------------|
| `L1`, `L2`          | `1.0` | Link lengths.                                |
| `DT`                | `0.02`| Physics tick.                                |
| `DAMPING`           | `0.98`| Joint-velocity damping per physics tick.     |
| `MAX_STEPS`         | `200` | Episode timeout.                             |
| `SUCCESS_THRESHOLD` | `0.05`| Success criterion on ee-to-goal distance.    |
| `DEFAULT_DECIMATION`| `4`   | Initial value of `sim.decimation`.           |
| `WORKSPACE_RADIUS`  | `2.0` | Pseudo-contact threshold base.               |

## Gotchas

- **No auto-reset.** Done envs stay terminal — `t_steps` does not advance,
  rewards are zero, qpos/qvel do not change — until you call
  `reset_idx(env_ids)`. If you ignore `done`, you train on a degenerate stream.
- **`mj_step` ≠ `advance`.** `mj_step` is one physics tick; `advance` is one
  RL step that internally calls `mj_step` `decimation` times. `t_steps`
  counts RL steps, not physics ticks.
- **`ctrl` semantics depend on actuator config.** With the default gainprm /
  biasprm, `ctrl` is a normalized joint position target. If you mutate
  `actuator_gainprm` or `biasprm`, the meaning changes accordingly.
- **`ctrl` must be a numpy array.** Passing a list or torch tensor will raise.
- **Read-methods return copies.** Mutating `sim.read_qpos()` is a no-op on the
  sim. Use `write_qpos` to set state.
- **`compute_site_pos` / `compute_jacp` only know `"ee"`.**
- **`decimation` is a runtime attribute**, not a constructor arg. (Divergent
  with IsaacLabEnv on purpose.)

## Contract test

`tests/test_mujoco_api.py` is the executable contract for this package.
