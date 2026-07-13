# Level 5 — Extensions

← back to [INTERVIEW.md](../INTERVIEW.md) · prev: [Level 4](level_4.md)

These are optional extensions. 

## 5a. Configurable observation delay

Each obs term may carry `delay: K` in the config. With `delay: K`, the
policy sees the obs term's value from K steps ago — per-env-independent
ring buffer, properly cleared on episode reset (no leakage across
episodes).

**Self-verify**:

```sh
pytest verify/level_5a.py
```

**API smoke only**: `load_mdp` accepts a `delay: K` field on an obs
term without crashing. The buffer's correctness (lag-by-K, per-env
reset behavior, no leakage) is graded by the hidden grader.

## 5b. Configurable observation noise

Each obs term may carry `noise_std: σ`. With non-zero σ, additive
Gaussian noise. Reproducible under a top-level `seed:`.

**Self-verify**:

```sh
pytest verify/level_5b.py
```

**API smoke only**: `load_mdp` accepts `noise_std: σ` without crashing.
Reproducibility under seed, zero-noise identity, and actual perturbation
are graded by the hidden grader.

## 5c. Action delay or domain randomization (stretch)

Either a configurable action-delay buffer OR per-episode randomization
of dynamics constants exposed via your unified env. Write your own
small verification test (`verify/level_5c.py`) demonstrating it works.

## 5d. Per-term reward logging (stretch)

During training, log each reward term's contribution separately (mean
per log interval). Should drop into your CSV alongside
`mean_episode_reward`.

## Walkthrough — what to be ready to present

If you reached any of the extensions, be ready to:

- **Show the wrapper code** for whichever extension you did (delay
  ring buffer, noise injection, action delay, per-term logging).
  Specifically: where does it sit in your obs / action / reward
  pipeline?
- **Explain the per-env reset behavior.** What happens to the buffer
  / noise state when one env in the batch resets while others keep
  going? Does old-episode data leak into the next episode?
- **Reproducibility.** For noise: how does the top-level `seed`
  thread through to your noise generator? If I run the same config
  twice, do I get identical observations?
- **The config schema you settled on.** Why these field names? How
  does it compose with the Level-3 term-list shape?

---

That's the whole exercise. Back to the [INTERVIEW.md](../INTERVIEW.md)
top page when you're ready to finalize.
