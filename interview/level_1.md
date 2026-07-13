# Level 1 — Unified env adapter + unified state

← back to [INTERVIEW.md](../INTERVIEW.md)

**Goal.** Wrap both sims behind one Python interface so a single driver loop
runs against either, AND expose a per-step **unified state object** that
later levels' MDP code will read from.

**Time budget.** 35–45 min.

## 1a. Unified env API

**Entry point** (`solution/env.py`):

```python
def make_env(name: str, num_envs: int, *, seed: int | None = None):
    """Returns a unified env wrapping the named sim. ``name`` ∈ {"isaaclab", "mujoco"}."""
```

The returned object MUST expose:

- `num_envs: int`, `obs_dim: int`, `act_dim: int`
- `reset() -> obs: np.ndarray (num_envs, obs_dim) float32`
- `step(action: np.ndarray (num_envs, act_dim) float32) -> (obs, reward, done, info)`
  - `obs:    np.ndarray (num_envs, obs_dim)` float32
  - `reward: np.ndarray (num_envs,)`         float32
  - `done:   np.ndarray (num_envs,)`         bool
  - `info:   dict`
- `close() -> None`

Standardize on **numpy** at this boundary so any RL library plugs in.

## 1b. Unified state object

In addition to the API above, your env must expose the **current per-step
state** as a single Python object — the same shape regardless of which
sim is underneath. Level 2's reward / observation / termination functions
will read attributes off this object.

How you expose it is your design choice (any of these is fine):

- An attribute, e.g. `env.state` — refreshed on each `step()`.
- A method, e.g. `env.snapshot() -> State` — called when needed.
- Returned alongside obs in `info`, etc.

What matters is that **you can document where to find it** so the
interviewer (and your own Level 2 code) can locate it.

The container's *type* is also your choice — `dataclass`,
`SimpleNamespace`, custom class, anything that supports attribute access.
What's NOT your choice is the **attribute names**, because the graders
will read them off your state object directly:

- `joint_pos` — `np.ndarray (num_envs, 2)` float32. Joint angles, in radians, wrapped to `[−π, π]`.
- `joint_vel` — `np.ndarray (num_envs, 2)` float32. Joint angular velocities.
- `ee_pos` — `np.ndarray (num_envs, 2)` float32. End-effector `(x, y)` in workspace.
- `goal` — `np.ndarray (num_envs, 2)` float32. Per-env goal `(x, y)`.
- `episode_step` — `np.ndarray (num_envs,)` int64. Steps since each env's last reset (0 at reset, +1 per `step`).
- `last_action` — `np.ndarray (num_envs, 2)` float32. Last action passed to `step` (post-clip).

Optional attributes (Level 5 / advanced reward terms): `applied_torque`,
`contact_force`. Add them if your terms need them.

## Why this matters

Level 2's `reward_fn` / `obs_fn` / `termination_fn` will be sim-agnostic
only because they read from this state object — which both adapters
populate identically from each sim's divergent accessors. This is the
foundation of the whole exercise.

## What's hard

torch ↔ numpy conversion. The 5-tuple-from-IsaacLab vs 4-tuple-reward-first-from-MuJoCo. Auto-reset semantics (IsaacLab) vs caller-managed reset
(MuJoCo). The two sims even use different *names* for the same concept
(`goal_pos` vs `target_xy()`, etc.). Read each sim's "Gotchas" section
before writing the adapter.

## Self-verify

```sh
pytest verify/level_1.py
```

**API smoke only**: env contract (shapes/dtypes), state surface
(attribute names exist with correct types). Cross-sim equality on
matched-state attributes is the hidden grader's main probe for this level.

## Walkthrough — what to be ready to present

In the 30-min walkthrough we'll spend time on:

- **The two adapters side by side.** Walk through how each sim's
  divergent API (`step` vs `advance`, 5-tuple vs 4-tuple-reward-first,
  auto-reset vs manual reset, torch vs numpy) is reconciled. Point at
  the lines where each divergence is handled.
- **Where the unified state object is built and how.** The single
  function or method (one per adapter) that pulls from the sim's
  native accessors and lands the values in the unified container.
  Be ready to name a sim-specific accessor you had to translate
  (e.g., IsaacLab `goal_pos` ↔ MuJoCo `target_xy()`).
- **How you expose the state to Level 2.** Attribute, method, returned
  in `info`, etc. — your choice; just be able to point at it.
- **One design decision you'd revisit.** E.g., container type
  (dataclass vs SimpleNamespace), whether to copy or reference numpy
  arrays, how you batch the per-env episode counter.

→ next: [Level 2 — sim-agnostic MDP composition + wiring](level_2.md)
