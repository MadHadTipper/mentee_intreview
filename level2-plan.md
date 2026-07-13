# Level 2 — Sim-agnostic MDP composition + wiring

## Decisions (locked)

- **No registry in Level 2.** Plain function lists; Level 3 adds a `dict[str, Callable]` on top, zero rework.
- **Wiring via a small `MdpEnv` wrapper** in `mdp.py`. `env.py` stays untouched.
- **One file** (`mdp.py`) — sections: term functions → combinators → public entry points → `MdpEnv`.
- **`solution/constants.py`** holds the exercise's shared numeric constants; `mdp.py` and `env.py` import from it instead of hardcoding.
- **Known limitation (flag in walkthrough, don't fix now):** `UnifiedEnv.step()` auto-resets done envs in-place (Level 1's design), so `env.state` right after `step()` is already the *post-reset* state for any env that just finished. `reward_fn`/`termination_fn` computed on that post-reset state won't reproduce the terminal-step reward/done exactly — the wrapper below returns the correct `reward`/`done` for that step (via env's own values, mathematically identical for termination since constants match), but an independent call to `termination_fn(env.state)` right after won't match on that exact boundary step. Real fix: caller-managed reset (`auto_reset=False` + manual `reset_envs`/`reset_idx` at the wrapper layer) so `env.state` always reflects the true pre-reset transition — didn't do it now for time.

## `solution/constants.py` (new file)

```python
"""Shared numeric constants used across solution/ levels."""
from dummy_isaaclab.env import SUCCESS_THRESHOLD, MAX_STEPS

ACT_DIM = 2
OBS_DIM_L2 = 8

TRACKING_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 0.001
CONTROL_EFFORT_WEIGHT = 0.001
SUCCESS_BONUS_WEIGHT = 50.0
```

## `solution/env.py` (small touch-up)

Replace the hardcoded `act_dim: int = 2` in `SimAdapter` with an import:
```python
from solution.constants import ACT_DIM
...
class SimAdapter(ABC):
    act_dim: int = ACT_DIM
```

## `solution/mdp.py` additions

```python
import numpy as np

from solution.constants import (
    SUCCESS_THRESHOLD, MAX_STEPS, OBS_DIM_L2,
    TRACKING_WEIGHT, SMOOTHNESS_WEIGHT, CONTROL_EFFORT_WEIGHT, SUCCESS_BONUS_WEIGHT,
)

# --------------------------------------------------------------- reward terms

def tracking(state, action) -> np.ndarray:
    """-‖ee-goal‖."""
    return -np.linalg.norm(state.ee_pos - state.goal, axis=-1).astype(np.float32)

def smoothness(state, action) -> np.ndarray:
    """-‖joint_vel‖²."""
    return -np.sum(state.joint_vel ** 2, axis=-1).astype(np.float32)

def control_effort(state, action) -> np.ndarray:
    """-‖action‖²."""
    return -np.sum(np.asarray(action, dtype=np.float32) ** 2, axis=-1).astype(np.float32)

def success_bonus(state, action) -> np.ndarray:
    """1[‖ee-goal‖ < SUCCESS_THRESHOLD]."""
    dist = np.linalg.norm(state.ee_pos - state.goal, axis=-1)
    return (dist < SUCCESS_THRESHOLD).astype(np.float32)

# ----------------------------------------------------------------- obs terms

def sin_cos_joint_pos(state) -> np.ndarray:
    """concat[sin(q), cos(q)] -> (N, 4)."""
    return np.concatenate([np.sin(state.joint_pos), np.cos(state.joint_pos)], axis=-1).astype(np.float32)

def joint_vel_obs(state) -> np.ndarray:
    """Pass-through joint_vel -> (N, 2). Named _obs to avoid shadowing state.joint_vel."""
    return state.joint_vel.astype(np.float32)

def ee_minus_goal(state) -> np.ndarray:
    """ee_pos - goal -> (N, 2)."""
    return (state.ee_pos - state.goal).astype(np.float32)

# ----------------------------------------------------------- termination terms

def reached_goal(state) -> np.ndarray:
    """‖ee-goal‖ < SUCCESS_THRESHOLD."""
    return np.linalg.norm(state.ee_pos - state.goal, axis=-1) < SUCCESS_THRESHOLD

def timeout(state) -> np.ndarray:
    """episode_step >= MAX_STEPS."""
    return state.episode_step >= MAX_STEPS

# ------------------------------------------------------------------ combinators

def _weighted_sum(terms, state, action) -> np.ndarray:
    """Sum weight * term(state, action) over all reward terms."""
    return sum(w * fn(state, action) for fn, w in terms).astype(np.float32)

def _concat(terms, state) -> np.ndarray:
    """Concatenate obs term outputs along the last axis, in list order."""
    return np.concatenate([fn(state) for fn in terms], axis=-1).astype(np.float32)

def _logical_or(terms, state) -> np.ndarray:
    """Elementwise OR of termination term outputs."""
    return np.logical_or.reduce([fn(state) for fn in terms])

_REWARD_TERMS = [
    (tracking, TRACKING_WEIGHT),
    (smoothness, SMOOTHNESS_WEIGHT),
    (control_effort, CONTROL_EFFORT_WEIGHT),
    (success_bonus, SUCCESS_BONUS_WEIGHT),
]
_OBS_TERMS = [sin_cos_joint_pos, joint_vel_obs, ee_minus_goal]
_TERMINATION_TERMS = [reached_goal, timeout]

# --------------------------------------------------------------- entry points

def reward_fn(state, action) -> np.ndarray:
    return _weighted_sum(_REWARD_TERMS, state, action)

def obs_fn(state) -> np.ndarray:
    return _concat(_OBS_TERMS, state)

def termination_fn(state) -> np.ndarray:
    return _logical_or(_TERMINATION_TERMS, state)

# ------------------------------------------------------------------- wiring

class MdpEnv:
    """Wraps a UnifiedEnv, substituting obs/reward/done with the composed MDP fns."""

    def __init__(self, env, reward_fn=reward_fn, obs_fn=obs_fn,
                 termination_fn=termination_fn, obs_dim: int = OBS_DIM_L2) -> None:
        self.env = env
        self._reward_fn = reward_fn
        self._obs_fn = obs_fn
        self._termination_fn = termination_fn
        self.num_envs = env.num_envs
        self.obs_dim = obs_dim
        self.act_dim = env.act_dim

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self._obs_fn(self.env.state)

    def step(self, action):
        _, _, done, info = self.env.step(action)
        state = self.env.state
        obs = self._obs_fn(state)
        reward = self._reward_fn(state, action)
        return obs, reward, done, info  # `done` reused from env.step (see limitation note above)

    def close(self) -> None:
        self.env.close()
```

Note: `MdpEnv.step()` deliberately reuses `env.step()`'s own `done` rather than recomputing `termination_fn(state)` — they're formula-identical (same `SUCCESS_THRESHOLD`/`MAX_STEPS`), so this is correct for the returned tuple; it's only the *independent* `termination_fn(env.state)` call after a reset boundary that can disagree, per the limitation note.

## Verification

```sh
uv run pytest verify/level_2.py -x -q
uv run pytest verify/level_1.py -x -q   # confirm the ACT_DIM refactor didn't regress Level 1
```
