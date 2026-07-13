"""Contract tests for `dummy_isaaclab.IsaacLabEnv`.

These tests are also living documentation — read them for canonical examples.
"""
import math

import pytest
import torch

from dummy_isaaclab import IsaacLabEnv
from dummy_isaaclab.env import (
    DEFAULT_DECIMATION,
    DEFAULT_KP,
    DEFAULT_MAX_TORQUE,
    L1,
    L2,
    MAX_STEPS,
    SUCCESS_THRESHOLD,
    WORKSPACE_RADIUS,
)


N = 8


def make_env(seed: int = 0, **kw) -> IsaacLabEnv:
    return IsaacLabEnv(num_envs=N, device="cpu", seed=seed, **kw)


# ---------------------------------------------------------------- shapes / dtypes

def test_reset_shapes_and_dtypes():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape == (N, 6) and obs.dtype == torch.float32
    assert isinstance(info, dict)
    assert env.joint_pos.shape == (N, 2) and env.joint_pos.dtype == torch.float32
    assert env.joint_vel.shape == (N, 2)
    assert env.ee_pos.shape == (N, 2)
    assert env.goal_pos.shape == (N, 2)
    assert env.last_action.shape == (N, 2)
    assert env.applied_torque.shape == (N, 2)
    assert env.joint_pos_target.shape == (N, 2)
    assert env.episode_step.shape == (N,) and env.episode_step.dtype == torch.int64
    assert env.action_space_low.shape == (2,)
    assert env.action_space_high.shape == (2,)
    assert env.workspace_violation.shape == (N,) and env.workspace_violation.dtype == torch.bool


def test_step_returns_5_tuple_with_correct_shapes():
    env = make_env()
    env.reset()
    a = torch.zeros((N, 2))
    out = env.step(a)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == (N, 6) and obs.dtype == torch.float32
    assert reward.shape == (N,) and reward.dtype == torch.float32
    assert terminated.shape == (N,) and terminated.dtype == torch.bool
    assert truncated.shape == (N,) and truncated.dtype == torch.bool
    assert isinstance(info, dict)


# ---------------------------------------------------------------- decimation / step / physics_step

def test_decimation_default_and_constructor_arg():
    env = make_env()
    assert env.decimation == DEFAULT_DECIMATION
    env2 = make_env(decimation=2)
    assert env2.decimation == 2


def test_step_advances_episode_step_by_one_per_call_regardless_of_decimation():
    env = make_env(decimation=8)
    env.reset()
    a = torch.zeros((N, 2))
    env.step(a)
    assert int(env.episode_step[0].item()) == 1
    env.step(a)
    assert int(env.episode_step[0].item()) == 2


def test_physics_step_advances_state_but_not_episode_step():
    env = make_env()
    env.reset()
    es0 = env.episode_step.clone()
    qp0 = env.joint_pos.clone()
    env.physics_step(torch.full((N, 2), 1.0))
    assert torch.equal(env.episode_step, es0)
    assert not torch.allclose(env.joint_pos, qp0)


def test_physics_step_clips_torque_to_max_torque():
    env = make_env()
    env.reset()
    env.physics_step(torch.full((N, 2), 100.0))
    assert torch.all(env.applied_torque <= env.max_torque + 1e-6)
    assert torch.all(env.applied_torque >= -env.max_torque - 1e-6)
    assert env.max_torque == DEFAULT_MAX_TORQUE


def test_position_mode_drives_joints_toward_target():
    env = make_env(decimation=4)
    env.reset()
    env.set_joint_state(torch.zeros((N, 2)), torch.zeros((N, 2)))
    target_action = torch.full((N, 2), 0.5)  # target = 0.5 * pi
    for _ in range(60):
        env.step(target_action)
    expected = 0.5 * math.pi
    err = (env.joint_pos - expected).abs()
    assert err.max().item() < 0.2, f"PD did not converge; err.max={err.max().item():.3f}"


def test_torque_mode_action_scales_to_max_torque():
    env = make_env(control_mode="torque")
    env.reset()
    env.set_joint_state(torch.zeros((N, 2)), torch.zeros((N, 2)))
    env.step(torch.full((N, 2), 0.5))  # 0.5 * max_torque
    expected = 0.5 * env.max_torque
    assert torch.allclose(env.applied_torque, torch.full((N, 2), expected), atol=1e-5)


def test_set_actuator_targets_persists_until_overwritten():
    env = make_env()
    env.reset()
    target = torch.full((N, 2), 0.3)
    env.set_actuator_targets(target)
    assert torch.allclose(env.joint_pos_target, target, atol=1e-5)


# ---------------------------------------------------------------- semantics

def test_termination_when_ee_close_to_goal():
    env = make_env()
    env.reset()
    env.set_target(env.ee_pos.clone())
    obs, reward, terminated, truncated, _ = env.step(torch.zeros((N, 2)))
    assert terminated.all()
    # Auto-reset clears episode counters.
    assert (env.episode_step == 0).all()


def test_truncation_after_max_steps():
    env = make_env()
    env.reset()
    env.set_target(torch.full((N, 2), 100.0))  # unreachable, no early termination
    a = torch.zeros((N, 2))
    last_truncated = None
    for _ in range(MAX_STEPS):
        _, _, _, last_truncated, _ = env.step(a)
    assert last_truncated.all()
    assert (env.episode_step == 0).all()


def test_action_clipping_and_last_action():
    env = make_env()
    env.reset()
    env.step(torch.full((N, 2), 5.0))
    assert torch.all(env.last_action <= 1.0 + 1e-6)
    assert torch.all(env.last_action >= -1.0 - 1e-6)


def test_get_set_state_round_trip():
    env = make_env()
    env.reset()
    s = env.get_state().clone()
    env.set_state(torch.zeros_like(s))
    assert torch.allclose(env.joint_pos, torch.zeros((N, 2)))
    env.set_state(s)
    assert torch.allclose(env.get_state(), s, atol=1e-5)


def test_seed_determinism():
    env_a = make_env(seed=42)
    env_b = make_env(seed=42)
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    assert torch.allclose(obs_a, obs_b)
    a = torch.full((N, 2), 0.3)
    out_a = env_a.step(a)
    out_b = env_b.step(a)
    for x, y in zip(out_a[:4], out_b[:4]):
        assert torch.allclose(x, y), "seeded runs should produce identical outputs"


def test_step_after_close_raises():
    env = make_env()
    env.reset()
    env.close()
    with pytest.raises(RuntimeError):
        env.step(torch.zeros((N, 2)))


def test_invalid_action_shape_raises():
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(torch.zeros((N + 1, 2)))


def test_invalid_decimation_or_control_mode_raises():
    with pytest.raises(ValueError):
        IsaacLabEnv(num_envs=2, decimation=0)
    with pytest.raises(ValueError):
        IsaacLabEnv(num_envs=2, control_mode="impulsive")


def test_reward_is_negative_distance_to_goal():
    env = make_env()
    env.reset()
    obs, reward, _, _, _ = env.step(torch.zeros((N, 2)))
    assert (reward <= 0).all()
    assert reward.abs().max().item() < (2 * WORKSPACE_RADIUS)


def test_jacobian_shape_and_basic_kinematics():
    env = make_env()
    env.reset()
    env.set_joint_state(torch.zeros((N, 2)), torch.zeros((N, 2)))
    J = env.jacobian()
    assert J.shape == (N, 2, 2)
    # At q=(0,0): de/dq1 = (0, L1+L2); de/dq2 = (0, L2)
    expected = torch.tensor([[0.0, 0.0], [L1 + L2, L2]])
    for i in range(N):
        assert torch.allclose(J[i], expected, atol=1e-5)


def test_workspace_violation_when_arm_stretched():
    env = make_env()
    env.reset()
    # q = (0, 0) → ee = (L1+L2, 0) → ‖ee‖ = 2 > 0.99*2
    env.set_joint_state(torch.zeros((N, 2)), torch.zeros((N, 2)))
    assert env.workspace_violation.all()
    # q = (pi/2, -pi/2) → ee at (0, L1) (within workspace)
    q = torch.tensor([math.pi / 2, -math.pi / 2]).expand(N, 2)
    env.set_joint_state(q, torch.zeros((N, 2)))
    assert not env.workspace_violation.any()


def test_default_kp_attribute():
    env = make_env()
    assert env.kp == DEFAULT_KP
    env.kp = 99.0
    assert env.kp == 99.0  # plain attr — runtime mutable
