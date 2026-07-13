# `dummy_isaaclab` — torch-backed batched arm sim

A toy batched simulator that mimics the Isaac Lab API style: torch tensors,
configurable device, gym-style 5-tuple `step()` returns, **automatic reset of
done envs inside `step()`**, **decimation** as a constructor arg, an internal
**PD position controller** for the high-level `step()`, plus a low-level
`physics_step(torque)` for direct torque control.

The world: a planar 2-link arm trying to reach a workspace **goal**. Each env
in the batch runs an independent episode.

```python
from dummy_isaaclab import IsaacLabEnv
import torch

env = IsaacLabEnv(num_envs=8, decimation=4, seed=0)
obs, info = env.reset()                     # obs: (8, 6) float32
action = torch.zeros((8, 2))                # joint position targets, normalized
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Two distinct kinds of "target"

Don't confuse them:

- **Goal target** — the workspace `(x, y)` point the arm is trying to reach.
  Read with `env.goal_pos`, set with `env.set_target(...)`. Used for reward.
- **Joint position target** — the per-step setpoint for the internal PD
  motor controller (radians). Read with `env.joint_pos_target`, set with
  `env.set_actuator_targets(...)` or implicitly by `step(action)` in
  position mode. Used to drive the actuator.

## Constructor

```python
IsaacLabEnv(
    num_envs:    int,
    device:      str = "cpu",
    seed:        int | None = None,
    decimation:  int = 4,                # constructor-fixed
    control_mode: str = "position",      # or "torque"
)
```

| Arg            | Meaning                                                                                  |
|----------------|------------------------------------------------------------------------------------------|
| `num_envs`     | Number of parallel envs in the batch.                                                    |
| `device`       | Torch device string. CPU works on any laptop.                                            |
| `seed`         | Seeds the env's `torch.Generator`.                                                       |
| `decimation`   | Physics ticks per `step()`. **Cannot be changed after construction** (Isaac-Lab style).  |
| `control_mode` | `"position"` (default) — actions are normalized joint position targets fed to internal PD. `"torque"` — actions are normalized direct joint torques. |

## Runtime-mutable attributes

These are plain attributes; mutate them directly between calls:

| Attribute    | Type    | Default | Purpose                                                 |
|--------------|---------|---------|---------------------------------------------------------|
| `kp`         | `float` | `18.0`  | PD position gain.                                       |
| `kd`         | `float` | `1.5`   | PD damping gain.                                        |
| `max_torque` | `float` | `4.0`   | Clip applied to torque before `physics_step` integrates.|

## State queries (properties — read-only views into live state)

| Property              | Type                       | Shape           | Notes                                              |
|-----------------------|----------------------------|-----------------|----------------------------------------------------|
| `joint_pos`           | `torch.Tensor` float32     | `(num_envs, 2)` | Joint angles `[q1, q2]`, wrapped to `[-π, π]`.    |
| `joint_vel`           | `torch.Tensor` float32     | `(num_envs, 2)` | Joint velocities `[q̇1, q̇2]`.                     |
| `ee_pos`              | `torch.Tensor` float32     | `(num_envs, 2)` | End-effector xy via forward kinematics.            |
| `goal_pos`            | `torch.Tensor` float32     | `(num_envs, 2)` | Workspace **goal** (RL target).                    |
| `joint_pos_target`    | `torch.Tensor` float32     | `(num_envs, 2)` | Last PD **motor** setpoint (radians).              |
| `applied_torque`      | `torch.Tensor` float32     | `(num_envs, 2)` | Torque applied during the most recent physics tick (post-clip). |
| `last_action`         | `torch.Tensor` float32     | `(num_envs, 2)` | Last action passed to `step()`, post-clip to `[-1, 1]`. |
| `episode_step`        | `torch.Tensor` int64       | `(num_envs,)`   | RL steps since this env's last reset (NOT physics ticks). |
| `workspace_violation` | `torch.Tensor` bool        | `(num_envs,)`   | `True` when `‖ee‖ > 0.99 * (L1 + L2)` — pseudo-contact at the boundary. |
| `action_space_low`    | `torch.Tensor` float32     | `(2,)`          | Per-action-dim lower bound, `[-1, -1]`.            |
| `action_space_high`   | `torch.Tensor` float32     | `(2,)`          | Per-action-dim upper bound, `[1, 1]`.              |

All tensors live on `env.device`.

## Methods

### `reset() -> (obs, info)`
Resets every env. Returns observation `(num_envs, 6)` float32 and an info dict.

### `step(actions: Tensor) -> (obs, reward, terminated, truncated, info)`
One RL step. Internally runs `decimation` physics ticks; in position mode each
tick is driven by the PD controller toward the action-derived target.

Inputs:
- `actions`: `(num_envs, 2)`, range `[-1, 1]`. Clipped if out of range.
  - Position mode: target angle = `action * π`.
  - Torque mode: torque = `action * max_torque`.

Returns the **gym 5-tuple**:
- `obs`: `(num_envs, 6)` float32 — `[q1, q2, q̇1, q̇2, goal_x − ee_x, goal_y − ee_y]`
- `reward`: `(num_envs,)` float32 — `−‖ee − goal‖`
- `terminated`: `(num_envs,)` bool — success (ee within `SUCCESS_THRESHOLD`).
- `truncated`: `(num_envs,)` bool — `episode_step >= MAX_STEPS = 200`.
- `info`: dict.

**Auto-reset.** Done envs are reset in-place at the end of `step()`.

### `physics_step(torque: Tensor) -> None`
Low-level: one physics tick driven by raw joint torque (clipped to ±`max_torque`).
Updates `joint_pos`, `joint_vel`, `applied_torque`. Does NOT advance
`episode_step`, compute reward, or auto-reset. Use this for custom integration
loops; `step()` calls it `decimation` times.

### `set_actuator_targets(targets: Tensor) -> None`
Set the joint position setpoint (radians) consumed by the PD controller. Useful
for manual control loops:

```python
env.set_actuator_targets(my_target_q)
for _ in range(env.decimation):
    env.physics_step(env._compute_pd_torque())
```

### `jacobian() -> Tensor`
Positional Jacobian of the ee w.r.t. joint angles. Shape `(num_envs, 2, 2)`;
last dim indexes the joint.

### `set_joint_state(q, qd) -> None`
Set joint positions and velocities for all envs. Shapes `(num_envs, 2)`.

### `set_target(target: Tensor) -> None`
Set the workspace goal xy. Shape `(num_envs, 2)`. Distinct from
`set_actuator_targets`.

### `get_state() / set_state(state)`
Round-trip the concatenated `[q1, q2, q̇1, q̇2]` state, shape `(num_envs, 4)`.

### `close() -> None`
Marks the env closed; further `step()` / `physics_step()` calls raise.

## Constants (importable from `dummy_isaaclab.env`)

| Name                | Value | Meaning                                       |
|---------------------|-------|-----------------------------------------------|
| `L1`, `L2`          | `1.0` | Link lengths (workspace radius `L1 + L2 = 2`).|
| `DT`                | `0.02`| Physics tick.                                 |
| `DAMPING`           | `0.98`| Joint-velocity damping per physics tick.      |
| `MAX_STEPS`         | `200` | Episode timeout (truncation).                 |
| `SUCCESS_THRESHOLD` | `0.05`| ee-to-goal distance for `terminated=True`.    |
| `DEFAULT_DECIMATION`| `4`   | Default value of the `decimation` ctor arg.   |
| `WORKSPACE_RADIUS`  | `2.0` | Pseudo-contact threshold base.                |

## Gotchas

- **Auto-reset.** A `done` env returns a fresh-episode `obs` from `step()` in
  the same call where `terminated`/`truncated` is `True`. Do not call `reset()`
  after observing termination — you'd start a *third* episode.
- **`step` ≠ `physics_step`.** `step` is the RL action; `physics_step` is one
  Δt of dynamics. `episode_step` counts `step` calls, not physics ticks.
- **Goal target vs joint target.** Reward uses `goal_pos`. The PD uses
  `joint_pos_target`. They are unrelated.
- **Action clipping is implicit.** Actions outside `[-1, 1]` are silently
  clipped. `last_action` returns the clipped value.
- **`decimation` is locked at construction.** If you need to change it,
  rebuild the env. (Mirrors Isaac Lab's config-driven decimation.)
- **No torque coupling.** Each joint is an independent PD-controlled actuator.

## Contract test

`tests/test_isaaclab_api.py` is the executable contract — if it passes, every
behavior documented above holds.
