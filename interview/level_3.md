# Level 3 — Config-driven MDP composition

← back to [INTERVIEW.md](../INTERVIEW.md) · prev: [Level 2](level_2.md)

**Time budget.** 30–40 min.

## Goal — the mechanism you're building

Build a system that **combines multiple term functions according to
fixed combination rules**, driven by a YAML config:

- **Rewards** are combined by **weighted sum**.
  Each entry under `rewards:` names a term function (signature
  `(state, action) -> (num_envs,)`) and a weight; the total reward is
  `Σᵢ weightᵢ × termᵢ(state, action)`.

- **Observations** are combined by **concatenation along the last axis**.
  Each entry under `observations:` names a term function (signature
  `(state) -> (num_envs, kᵢ)`); the total obs is `np.concatenate([…], axis=-1)`,
  shape `(num_envs, Σᵢ kᵢ)`.

- **Terminations** are combined by **OR** (or **AND** when `mode: all`).
  Each entry under `terminations.terms:` names a predicate function
  (signature `(state) -> (num_envs,) bool`); the result is their elementwise
  OR (any) or AND (all).

The acid test: **someone else can add a new reward term, observation
term, or termination predicate by (a) writing a Python function and
(b) referencing it in the YAML — without touching your `load_mdp`, your
combiner, your trainer, or any other plumbing.**

If you implemented Level 2 as separate term functions called from a
weighted sum / concat / OR, Level 3 is mostly wrapping that in a
registry and a config loader.

## Entry point (`solution/mdp.py`)

```python
def load_mdp(config: str | dict) -> tuple[reward_fn, obs_fn, termination_fn, obs_dim: int, act_dim: int]:
    """``config`` may be a path to a YAML file or an already-parsed dict."""
```

The returned `reward_fn` / `obs_fn` / `termination_fn` are constructed
from the config via the three combination rules above.

## Suggested config schema (you may diverge)

```yaml
observations:
  - { name: sin_cos_joint_pos }
  - { name: joint_vel }
  - { name: ee_minus_goal }

rewards:
  - { name: tracking,         weight: 1.0 }
  - { name: smoothness,       weight: 0.001 }
  - { name: control_effort,   weight: 0.001 }
  - { name: success_bonus,    weight: 50.0, params: { threshold: 0.05 } }

terminations:
  mode: any
  terms:
    - { name: ee_close_to_goal, params: { threshold: 0.05 } }
    - { name: timeout,          params: { max_steps: 200 } }

seed: 0
```

## What "easily add a new term" means concretely

To add a new reward term `path_smoothness` (penalizing change in joint
velocity step-to-step):

1. Write the function (registering it however your design demands):
   ```python
   def path_smoothness(state, action):
       return -np.sum(np.diff(state.joint_vel) ** 2, axis=-1)
   ```
2. Add an entry to the YAML:
   ```yaml
   rewards:
     - { name: path_smoothness, weight: 0.01 }
   ```
3. **No other code changes.** Your `load_mdp` finds the new term by
   name; the trainer (Level 4) just loads the new YAML; the verifier
   still runs.

Same pattern for new observation terms (write a function returning
`(num_envs, k)`, register it, add to `observations:`) and termination
predicates (write a function returning `(num_envs,)` bool, register it,
add to `terminations.terms:`).

The walkthrough at the end of the interview will probe this. Be ready
to demo "if I gave you 5 minutes, could you add a new reward term right
now?" — the answer should be "yes, and here's how".

## Self-verify

```sh
pytest verify/level_3.py
```

**API smoke only**: `load_mdp` exists and returns a 5-tuple from both
YAML paths and dicts. Whether two distinct configs *actually* produce
distinguishable functions, whether weights/params take effect, whether
unknown term names raise a clear error, etc., are graded by the hidden
grader.

## Walkthrough — what to be ready to present

- **The registry (or whatever maps term names to functions).** Show
  it. How are new terms added? What happens when the config names a
  term that isn't registered?
- **`load_mdp` from YAML to public fns.** Trace one term from the YAML
  entry, through your registry lookup, into the assembled
  `reward_fn` / `obs_fn` / `termination_fn`. Where does the weight
  apply? Where do the params land?
- **Live demo: "Add a new reward term right now."** Suppose we want a
  new `path_smoothness` term that penalizes squared change in joint
  velocity step-to-step. Show the code change and the YAML edit. The
  goal is **<2 minutes**, no edits to `load_mdp` or `train.py`.
- **How `mode: any` vs `mode: all` is honored** in your termination
  combiner.

→ next: [Level 4 — train with an outside RL library](level_4.md)
