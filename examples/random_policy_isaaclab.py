"""Drive `dummy_isaaclab.IsaacLabEnv` with a random policy for 100 steps.

Demonstrates the bare API surface in isolation. No unification, no shared MDP.
"""
import torch

from dummy_isaaclab import IsaacLabEnv


def main() -> None:
    num_envs = 8
    env = IsaacLabEnv(num_envs=num_envs, device="cpu", seed=0)
    obs, info = env.reset()
    print(f"reset: obs.shape={tuple(obs.shape)} obs.dtype={obs.dtype}")

    rewards = []
    terminations = 0
    truncations = 0
    for t in range(100):
        action = torch.empty((num_envs, 2)).uniform_(-1.0, 1.0)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward.mean().item())
        terminations += int(terminated.sum().item())
        truncations += int(truncated.sum().item())

    print(f"100 steps done. mean reward over batch: {sum(rewards) / len(rewards):.3f}")
    print(f"#terminated={terminations}  #truncated={truncations}")
    print(f"final joint_pos[0]={env.joint_pos[0].tolist()}")
    print(f"final ee_pos[0]={env.ee_pos[0].tolist()}  goal[0]={env.goal_pos[0].tolist()}")
    env.close()


if __name__ == "__main__":
    main()
