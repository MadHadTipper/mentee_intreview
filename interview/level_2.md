# Level 2 — Sim-agnostic MDP composition + wiring

← back to [INTERVIEW.md](../INTERVIEW.md) · prev: [Level 1](level_1.md)

**Time budget.** 30–40 min.

## What to build

1. A set of **term functions** (small pure functions reading from the
   Level-1 state object — never from a sim directly):
   - **Reward terms**: signature `(state, action) -> (num_envs,) float32`.
   - **Observation terms**: signature `(state) -> (num_envs, kᵢ) float32`.
   - **Termination terms**: signature `(state) -> (num_envs,) bool`.
2. A way to **combine** them into the public `reward_fn` / `obs_fn` /
   `termination_fn`:
   - rewards = **weighted sum** of reward terms,
   - obs = **concatenation** of obs terms along the last axis,
   - terminations = **OR** of termination predicates.
3. **Wire** the combined functions into your env: `env.step()`'s
   `(obs, reward, done)` must equal what `obs_fn`, `reward_fn`,
   `termination_fn` would compute on the current state. *How* you wire
   this is your design choice — document the call site.

**Hard rule.** Every term function reads only from the unified state
object — no `if sim == "isaaclab": …`, no reaching into the underlying
sim. This is the property that makes the MDP sim-agnostic.

## The MDP you ship at this level

The full reward / obs / termination spec — your combined functions must
produce these values.

**Reward** (weighted sum of four terms):

```
reward = − ‖ee − goal‖                  (weight 1.0    — "tracking")
       − 0.001 · ‖joint_vel‖²           (weight 0.001  — "smoothness")
       − 0.001 · ‖action‖²              (weight 0.001  — "control_effort")
       + 50    · 1[‖ee − goal‖ < 0.05]  (weight 50.0   — "success_bonus")
```

**Observation** — `concat[sin(q), cos(q), joint_vel, ee − goal]`, 8 dims
total (sin/cos avoids the `+π → −π` discontinuity).

**Termination** — `(‖ee − goal‖ < 0.05) OR (episode_step ≥ 200)`.

## One worked example per category

To anchor what a "term function" looks like, here's one from each
category. The other terms in the spec follow the same pattern:

```python
def tracking(state, action):                       # reward term
    return -np.linalg.norm(state.ee_pos - state.goal, axis=-1)

def ee_minus_goal(state):                          # obs term
    return (state.ee_pos - state.goal).astype(np.float32)

def reached_goal(state):                           # termination term
    return np.linalg.norm(state.ee_pos - state.goal, axis=-1) < 0.05
```

You write the rest of the terms (smoothness, control_effort,
success_bonus, sin_cos_joint_pos, joint_vel obs, timeout), then build
your combiner that turns `[(term, weight), …]` into `reward_fn`, etc.
*How* the combiner is structured is up to you — Level 3 will generalize
it.

## Entry points (`solution/mdp.py`)

```python
def reward_fn(state, action) -> np.ndarray: ...        # (num_envs,)   float32
def obs_fn(state) -> np.ndarray: ...                   # (num_envs, 8) float32
def termination_fn(state) -> np.ndarray: ...           # (num_envs,)   bool
```

## Self-verify

```sh
pytest verify/level_2.py
```

**API smoke only**: the three module-level fns are callable and your
unified `env.step` returns arrays of consistent shape. Cross-sim
**equality on matched states** is the actual grading criterion and is
checked by the hidden grader.

## Walkthrough — what to be ready to present

- **One reward term, line by line.** Pick any of yours (e.g.,
  `tracking`) and explain: which fields of `state` does it read?
  Could it run on the MuJoCo sim if I unplugged the IsaacLab adapter?
  (The answer should be "yes — and here's why".)
- **The aggregation code.** Show the lines where rewards get summed
  by weight, observations get concatenated, and termination predicates
  get OR'd. Three separate functions or one generic combiner — explain
  your choice.
- **The wiring decision.** Point at the call site where `env.step()`
  produces its `(reward, obs, done)` from your `reward_fn` /
  `obs_fn` / `termination_fn`. There is no single right answer —
  but you should know yours and be able to defend it.
- **Sim-agnostic property in your own words.** Why does the same
  reward_fn work on both sims? Walk back from a term function to the
  state attribute it reads to the adapter line that populated it.

→ next: [Level 3 — config-driven MDP composition](level_3.md)
