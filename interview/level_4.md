# Level 4 — Train with an outside RL library

← back to [INTERVIEW.md](../INTERVIEW.md) · prev: [Level 3](level_3.md)

**Goal.** Train PPO (or SAC, A2C, etc. — your choice) on each sim using
your unified env + an MDP loaded from your Level-3 config system. Write
a CSV showing reward improving over a random-policy baseline.

`solution.train` is responsible for taking the `--config PATH` flag,
loading the MDP from it (via Level 3's `load_mdp`), and threading the
resulting `reward_fn` / `obs_fn` / `termination_fn` into the env that
the RL library actually trains on. **How** you thread them in —
constructor argument to `make_env`, module-level state, builder
function, decorator — is your design choice (Level 1 left this open
on purpose). Whatever you pick, the trainer should **not know anything
about specific reward / obs / termination terms**: swapping in a new
task or a new reward term must be a config edit (and possibly a new
term function + register call) — **no changes to `solution/train.py`**.

**Time budget.** 30–40 min. (Most of this is iterating on the wrapper +
a single short training run; don't try to tune hyperparameters.)

## Entry point (CLI)

```sh
python -m solution.train --sim {isaaclab,mujoco} --config PATH --steps N --out DIR --seed K
```

Must write `{DIR}/progress.csv` with header `step,mean_episode_reward`.
The final row's mean episode reward must clear the random baseline by
a clear margin.

The repo declares `stable-baselines3` and `gymnasium` as `[rl]` extras
already. You can use them, swap to `cleanrl`, `tianshou`, `RLlib` —
whatever. Either install your library of choice or work within `[rl]`.

## Property the walkthrough will check

In the 30-min walkthrough we may ask: "Add a new reward term and re-run
training without touching `train.py`." If your Level 3 used a registry
+ config, the answer is: write a 3-line function with
`@register_reward(...)`, add one line to the YAML, re-run the same
`python -m solution.train` command. **No edits to `train.py` or
`load_mdp` should be needed.**

## Self-verify

```sh
pytest verify/level_4.py -m slow
```

**API smoke only**: your CLI accepts the documented flags, runs to
completion (briefly, 2k steps), and writes a `progress.csv` with the
right header. Whether the trained policy *actually beats a random
baseline* is graded by the hidden grader, which uses a longer training
budget.

## Walkthrough — what to be ready to present

- **The line where the RL library gets your env.** Show the wrapper /
  adapter that bridges your unified env to whatever interface your
  library expects (`gymnasium.VectorEnv`, etc.). Briefly note any
  quirks (e.g., how `done=True` interacts with auto-reset).
- **How `--config PATH` flows from the CLI into the env.** Walk
  `args.config` → `load_mdp` → whatever mechanism injects the
  resulting `reward_fn` / `obs_fn` / `termination_fn` into
  `make_env`'s output. We're checking that `train.py` is
  term-agnostic.
- **The learning curve.** Open `progress.csv` from one of your runs.
  Compare the final mean episode reward to a rough mental random
  baseline. If they're close, talk about why training stalled.
- **Hyperparameter / library choice — *briefly*.** One sentence on
  why SB3 / CleanRL / RLlib / etc. We're not grading
  hyperparameter tuning.

→ next: [Level 5 — extensions](level_5.md)
