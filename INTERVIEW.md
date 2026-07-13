# Interview exercise — unified batched-sim RL training layer

## Format and pacing

You have **3 hours total, on-site**:

- **0:00 – 0:15** — handoff and setup with the interviewer.
- **0:15 – 2:30** — you work alone (**2h 15m**). This is the coding window.
- **2:30 – 3:00** — you walk the interviewer through what you built (**30 min**).

The exercise has more levels than fit in 2h 15m on purpose — we don't expect
anyone to finish all of them. Aim for Levels 1–3 cleanly first; Level 4 if
you have time; Level 5 only if you finish 4 with time to spare. **Quality of
the code you submit and being able to explain what was done matter more than how many levels you reach.**

A realistic split for someone hitting most of the budget:

| Phase                       | Target    |
|-----------------------------|-----------|
| Onboarding (Level 0)              | ~10 min   |
| Level 1 (unified env + state)     | 35–45 min |
| Level 2 (MDP composition + wire)  | 30–40 min |
| Level 3 (config-driven MDP)       | 30–40 min |
| Level 4 (training)                | 30–40 min |
| Buffer / Level 5                  | remaining |

If you find yourself rabbit-holing on Level 1 backend conversion past the
40-min mark, ship something that just works (even ugly) and keep going —
you can come back if there's time. 

## Background — RL and environments in 60 seconds

If you've never written RL code before, here's the vocabulary you need:

- An **environment** is a thing you can `reset()` (gives a starting
  observation) and `step(action)` (advances one tick and returns the next
  observation, a scalar reward, and a `done` flag).
- An **episode** is a sequence of steps from `reset()` until `done=True`. The
  agent gets cumulative reward over the episode and we want that to be high.
- A **policy** is a function `obs → action` (typically a small neural net).
  RL **trains** the policy: it runs many episodes, observes which actions led
  to high reward, and adjusts the policy's weights toward those.
- A **batched** (or **vectorized**) environment runs N independent envs in
  parallel and returns batched arrays — `obs.shape == (N, obs_dim)`,
  `reward.shape == (N,)`, etc. Both sims here are batched. Parallel rollouts
  give RL libraries more samples per wall-clock second.
- An **RL library** (Stable-Baselines3 / CleanRL / RLlib / tianshou / …)
  contains the policy training loop — you give it an env conforming to its
  expected interface and it trains a policy on it. **You do not need to
  implement PPO yourself** — pick a library, plug your unified env in.

The typical interaction loop, in pseudocode:

```python
env = make_env(...)
obs = env.reset()                          # (num_envs, obs_dim)
for _ in range(many_steps):
    action = policy(obs)                   # (num_envs, act_dim)
    obs, reward, done, info = env.step(action)
    # RL library updates the policy from collected (obs, action, reward) data.
    # When `done` is True for an env, that env auto-resets to a fresh episode.
```

That's it. The exercise is about plumbing — making this same loop work over
two structurally different sims, with rewards / observations / terminations
defined once and reused.

## What you're building

You have two batched simulators in this repo: **`dummy_isaaclab`** (Isaac-Lab-flavored,
torch backend) and **`dummy_mujoco`** (MuJoCo-flavored, numpy backend). Both
simulate the same toy task — a 2-link planar arm reaching a target — but with
**deliberately divergent APIs**: different method names, different tensor
backends, different return-tuple shapes, different reset semantics, different
actuator surfaces.

Your job: build the infrastructure that lets you train an RL policy on
either sim. In practice that means you'll write **multiple small reward,
observation, and termination "term" functions** — each one sim-agnostic
(it reads from a unified state object, not from a specific sim) — and a
**mechanism that combines them**: rewards summed by weight, observations
concatenated, terminations OR'd. The whole MDP is **defined by a YAML
config** (which terms, with which weights / params), and a **training
script** loads the config, builds the env on top of either sim, and
hands it to the RL library of your choice.

The result: switching sims, swapping reward shaping, or adding a new
sensor signal is a config edit (and possibly one new term function) —
never a refactor.

## The world: a 2-link arm in 2D

You don't need a robotics background — this is a stick-figure problem. Both
sims simulate the same setup:

- Two rigid links of equal length 1.0, joined end-to-end (think: shoulder +
  elbow, but flat on a table — purely 2D motion).
- Link 1 is anchored at the origin `(0, 0)` and rotates around its joint there.
- Link 2 attaches to the far end of link 1 and rotates around its own joint.
- The far tip of link 2 is the **end-effector** — the point we want to
  position. We'll use the shorthand **`ee`** for it.
- Each episode samples a random **goal** — a 2D point `(x, y)` somewhere in
  the arm's reachable area. The arm tries to put its end-effector on the goal.

```
                                  ●  ← end-effector (the arm's tip; "ee")
                                 ╱
                                ╱  link 2
                               ╱
                          ●───╱      ← joint 2 (rotates by angle q2)
                         ╱
                        ╱  link 1
                       ╱
                      ●          ← joint 1 (anchored at origin (0,0); rotates by q1)

                                            ✕  ← goal (random per episode)
```

### Vocabulary you'll see in code and below

- **joint angle** — also `q`, `joint_pos`. Rotation angle (radians) of a
  joint. Two joints → 2 numbers per env: `q = [q1, q2]`. Wrapped to `[−π, π]`.
- **joint velocity** — also `qdot`, `joint_vel`. Rate of change of each
  joint angle. 2 numbers per env.
- **end-effector** — also `ee`, `ee_pos`. The arm's tip in `(x, y)`,
  computed from the two joint angles by trigonometry (forward kinematics).
- **goal** — also `target`. A 2D point `(x, y)` to reach. Sampled at
  episode start.
- **action** — also `control`. The command sent to each joint, per step.
  A 2-element vector in `[-1, 1]`. Drives the joints toward target angles.
- **episode** — a sequence of steps from `reset()` until `done=True`.
  Max 200 steps before timeout.
- **episode_step** — counter of steps within the current episode (0 at
  reset, +1 per step).
- **`‖v‖`** ("norm") — length of a 2D vector: `‖v‖ = sqrt(v[0]² + v[1]²)`.
  In numpy: `np.linalg.norm(v)`.
- **`‖v‖²`** — squared length: `v[0]² + v[1]²`. Cheaper to compute, used
  as a regularizer.

The task: minimize the distance from the end-effector to the goal.
**Reward goes up when the arm's tip gets closer to the goal**.

## How the levels build on each other

The exercise is **stratified, not orthogonal**. Each level depends on
choices you made in earlier ones. Skim this whole section before
opening Level 1 — a clean Level 1 design makes Levels 2–4 short; a
sloppy one forces refactors.

- **Level 1 — unified env + unified state.** Wrap both sims behind a
  single Python interface (`reset`, `step`, `close`) AND expose a
  per-step **unified state object** with the attribute names later
  levels will read.
- **Level 2 — sim-agnostic MDP composition.** Write reward / observation
  / termination *term functions* that read only from the unified state.
  Combine them via fixed rules (weighted sum / concat / OR). Wire the
  combined functions into your env so `env.step` returns their values.
- **Level 3 — config-driven terms.** Generalize Level 2's hardcoded
  term list into a registry plus a YAML config. Adding a term becomes
  a function + a config line.
- **Level 4 — train.** A training CLI that loads a config, builds the
  env, hands it to an RL library. The trainer never names specific
  terms.
- **Level 5 — sensor extensions.** Per-obs-term `delay` and `noise_std`
  in the config.

Reading order matters: choices about your **state object** (Level 1)
shape the term function signatures (Level 2). Choices about your
**term-aggregation mechanism** (Level 2) shape the registry (Level 3).
Choices about how you **wire the MDP into the env** (Level 2) determine
how `train.py` ends up loading the config (Level 4). Plan for the
chain.

## Read first

1. **`README.md`** for install (use `uv`).
2. **`dummy_isaaclab/README.md`** and **`dummy_mujoco/README.md`** — full API
   references for each sim. They cover every method/property, every divergence,
   and the common 2-link arm task. You should not need to read the sim source.
3. **`examples/random_policy_*.py`** — ~30 lines each, exercises every sim's
   bare API. Run them.
4. **`tests/test_*_api.py`** — provided contract tests. They are the
   executable spec for each sim. They must pass on a fresh install
   (`pytest tests/`).

## How you're evaluated

> ⚠️ **Read this carefully**

There are **three** layers of evaluation. Make sure you understand which
artifact each one looks at.

### 1. `pytest verify/` — API-compile check (you can run this)

The `verify/` suite shipped in this repo only checks that **the right
symbols exist with the right signatures and that your code runs end-to-end
without crashing**. It does **not** verify:
- whether your reward is computed correctly,
- whether your two adapters produce identical outputs on matched states,
- whether your policy actually trains,
- whether your sensor-delay buffer has the right semantics,
- almost anything else interesting.

Passing `pytest verify/` is **necessary** (your code has to compile and
fit the contract) but **not sufficient** to score well.

Use it as a self-check that you've wired things correctly. If `pytest
verify/level_3.py` fails with `NotImplementedError`, you haven't done
Level 3. If it passes, your `solution.mdp.load_mdp` exists with the right
shape — but whether it does the right thing is graded separately.

### 2. Hidden grader (we run, you don't see)

We run a private test suite against your submission that probes the
behaviors the verify suite skips: cross-sim equality on matched states,
training reaching a learning threshold, ring-buffer correctness on edge
cases, term-composition with novel configs, etc. **This is where the
correctness portion of your grade comes from.**

The grader is built on top of the same sim APIs and entry-point contracts
documented here — there's nothing secret about the *contracts*; we just
test them harder than `verify/` does. If your code is principled and
follows the contracts, you'll do fine on the grader without ever seeing it.

### 3. Walkthrough (30 min, at the end)

In the last 30 minutes you walk the interviewer through your code.
The interviewer asks follow-up questions while you present. **This is
where most of the "design quality" and "engineering judgment" signal
comes from — be ready to defend each design choice in your own words.**

**Format.** No slides; reading off your own code is fine. The
interviewer will be reading along with you, so use the 2h 15m of
coding time to make sure your code is presentable — clean imports,
sensible names, a one-line docstring per function.

**Plan to cover, roughly in this order:**

1. **Quick overview (≤ 5 min)** — which levels you completed, what's
   incomplete and why, anything you wish you'd had more time for.
2. **Per-level deep-dive** for each level you reached. The level-specific
   things you'll be asked to show / explain are listed at the bottom of
   each level file, under **"Walkthrough — what to be ready to
   present"**. Read those sections during the coding window so nothing
   surprises you:
   - [Level 1 — adapter divergence + state object](interview/level_1.md#walkthrough--what-to-be-ready-to-present)
   - [Level 2 — term function + aggregation + wiring](interview/level_2.md#walkthrough--what-to-be-ready-to-present)
   - [Level 3 — registry / config / live "add a term" demo](interview/level_3.md#walkthrough--what-to-be-ready-to-present)
   - [Level 4 — RL-lib bridge + learning curve](interview/level_4.md#walkthrough--what-to-be-ready-to-present)
   - [Level 5 — sensor wrapper, per-env reset, reproducibility](interview/level_5.md#walkthrough--what-to-be-ready-to-present) (if reached)
3. **Tradeoffs you made.** What did you do quickly that you'd revisit
   with more time? What's your weakest piece of code and why?

## What you're graded on

Roughly weighted:

- **Correctness** *(hidden grader + verify)* — the levels you attempted
  work end-to-end on the hidden grader. We don't expect every level
  finished.
- **Design quality** *(walkthrough + review)* — clean abstractions;
  sim-agnostic reward/obs/termination; you can explain *why* you chose
  each piece.
- **Code clarity** *(review)* — readable, idiomatic Python; sensible
  module boundaries; your naming + structure mirror the design.
- **Engineering judgment** *(walkthrough)* — you know your code's
  limits; you can articulate tradeoffs and what you'd do differently
  with more time.

You do NOT need to:
- match a specific architecture or library choice,
- chase the highest possible reward,
- write tests beyond what's needed to convince yourself things work,
- use any particular RL algorithm,
- finish every level (most candidates won't).

You SHOULD:
- read each sim's README before writing the adapter,
- be ready to explain every design choice in the 30-min walkthrough,
- pass `pytest verify/` (necessary floor) for the levels you completed,
- ship working code on the levels you attempted, even if not maximally clean.

Each level has a fixed **entry-point contract**: an importable symbol with a
specified signature. The contract is reprinted under each level below. Stick
to it — both `verify/` and the hidden grader assume it.

## Where to put your code

Inside the repo's `solution/` package — there are stub files for each level
with `NotImplementedError` placeholders. Your work should fit naturally into
those stubs (or new modules you import from them); don't put anything
elsewhere.

You may freely create your own modules under `solution/` (e.g. `solution/mdp_internals/`)
to hold helpers; just keep the public entry points where the contracts ask
for them.

You **must NOT** modify `dummy_isaaclab/`, `dummy_mujoco/`, `tests/`, or
`verify/`. (You may read them as much as you like.)

---

## The levels — open one at a time

Each level lives in its own file under `interview/`. Read the level you're
about to attempt; you don't need to read ahead.

- [**Level 1 — Unified env adapter + unified state**](interview/level_1.md) — wrap both sims behind one API and expose a per-step unified state object.
- [**Level 2 — Sim-agnostic MDP composition + wiring**](interview/level_2.md) — write reward / obs / termination terms over the unified state, combine them, wire them into the env.
- [**Level 3 — Config-driven MDP composition**](interview/level_3.md) — make the MDP definable from a YAML config; new terms = new function + config edit.
- [**Level 4 — Train with an outside RL library**](interview/level_4.md) — CLI + progress.csv; trainer is term-agnostic.
- [**Level 5 — Extensions**](interview/level_5.md) — observation delay, observation noise, action delay, per-term logging (stretch).

## Tips, gotchas, ground rules

- **The two sims diverge on purpose.** Read each sim's README before you
  write the adapter — every divergence is documented there.
- **Don't use the sims' built-in reward** in your training. The sims expose
  enough state (joint pos/vel, ee pos, goal, applied torque, contact-like
  signals, episode step) to build any reward you want via your own
  reward_fn. The sims' built-in reward is for diagnostics only.
- **Reset is the trap.** IsaacLab auto-resets done envs inside `step()`;
  MuJoCo freezes terminal envs and waits for `reset_idx`. Your unified env
  needs to handle both consistently. (Hint: the sims expose flags / methods
  for opting out of their auto-reset behavior — check their READMEs.)
- **Determinism.** Both sims accept `seed: int | None = None` and seed all
  RNG state. The verify suite passes `seed=0` everywhere.
- **`pytest verify/` is the source of truth.** If a verify test passes, that
  level is done. If it fails, fix what it complains about.
- **Out of scope.** You don't need GPU, distributed training, hyperparameter
  search, or beating any specific reward number beyond the threshold.

Good luck.
