"""Drive `dummy_mujoco.MuJoCoSim` with a random policy for 100 steps.

Demonstrates the bare API surface in isolation, including the caller-managed
reset of done envs.
"""
import numpy as np

from dummy_mujoco import MuJoCoSim


def main() -> None:
    n_parallel = 8
    sim = MuJoCoSim(n_parallel=n_parallel, seed=0)
    obs = sim.reset_all()
    print(f"reset: obs.shape={obs.shape} obs.dtype={obs.dtype}")

    rng = np.random.default_rng(0)
    rewards = []
    n_resets = 0
    for t in range(100):
        ctrl = rng.uniform(-1.0, 1.0, size=(n_parallel, 2)).astype(np.float32)
        reward, obs, done, _ = sim.advance(ctrl)
        rewards.append(reward.mean())
        if done.any():
            ids = np.flatnonzero(done)
            sim.reset_idx(ids)
            n_resets += int(ids.size)

    print(f"100 steps done. mean reward over batch: {float(np.mean(rewards)):.3f}")
    print(f"#env-resets triggered: {n_resets}")
    print(f"final qpos[0]={sim.read_qpos()[0].tolist()}")
    print(f"final ee[0]={sim.compute_site_pos('ee')[0].tolist()}  target[0]={sim.target_xy()[0].tolist()}")
    sim.shutdown()


if __name__ == "__main__":
    main()
