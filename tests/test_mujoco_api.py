"""Contract tests for `dummy_mujoco.MuJoCoSim`.

These tests are also living documentation — read them for canonical examples.
"""
import math

import numpy as np
import pytest

from dummy_mujoco import MuJoCoSim
from dummy_mujoco.sim import (
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


def make_sim(seed: int = 0) -> MuJoCoSim:
    return MuJoCoSim(n_parallel=N, seed=seed)


# ---------------------------------------------------------------- shapes / dtypes

def test_initial_shapes_and_dtypes():
    sim = make_sim()
    assert sim.read_qpos().shape == (N, 2) and sim.read_qpos().dtype == np.float32
    assert sim.read_qvel().shape == (N, 2)
    assert sim.compute_site_pos("ee").shape == (N, 2)
    assert sim.target_xy().shape == (N, 2)
    assert sim.qfrc_actuator().shape == (N, 2)
    assert sim.compute_jacp("ee").shape == (N, 2, 2)
    assert sim.contact_force().shape == (N,) and sim.contact_force().dtype == np.float32
    assert sim.last_ctrl.shape == (N, 2) and sim.last_ctrl.dtype == np.float32
    assert sim.t_steps.shape == (N,) and sim.t_steps.dtype == np.int32
    low, high = sim.ctrl_range
    assert low.shape == (2,) and high.shape == (2,)


def test_advance_returns_4_tuple_reward_first():
    sim = make_sim()
    sim.reset_all()
    out = sim.advance(np.zeros((N, 2), dtype=np.float32))
    assert len(out) == 4
    reward, obs, done, info = out
    assert reward.shape == (N,) and reward.dtype == np.float32
    assert obs.shape == (N, 4) and obs.dtype == np.float32
    assert done.shape == (N,) and done.dtype == np.bool_
    assert isinstance(info, dict)


# ---------------------------------------------------------------- decimation / mj_step / advance

def test_decimation_default_runtime_settable():
    sim = make_sim()
    assert sim.decimation == DEFAULT_DECIMATION
    sim.decimation = 1
    assert sim.decimation == 1


def test_advance_advances_t_steps_by_one_per_call_regardless_of_decimation():
    sim = make_sim()
    sim.reset_all()
    sim.decimation = 8
    sim.advance(np.zeros((N, 2), dtype=np.float32))
    assert int(sim.t_steps[0]) == 1
    sim.advance(np.zeros((N, 2), dtype=np.float32))
    assert int(sim.t_steps[0]) == 2


def test_mj_step_advances_state_but_not_t_steps():
    sim = make_sim()
    sim.reset_all()
    t0 = sim.t_steps.copy()
    qp0 = sim.read_qpos()
    sim.mj_step(np.full((N, 2), 1.0, dtype=np.float32))
    assert np.array_equal(sim.t_steps, t0)
    assert not np.allclose(sim.read_qpos(), qp0)


def test_mj_step_clips_torque_to_max_torque():
    sim = make_sim()
    sim.reset_all()
    sim.mj_step(np.full((N, 2), 100.0, dtype=np.float32))
    assert np.all(sim.qfrc_actuator() <= sim.max_torque + 1e-6)
    assert np.all(sim.qfrc_actuator() >= -sim.max_torque - 1e-6)
    assert sim.max_torque == DEFAULT_MAX_TORQUE


def test_advance_drives_joints_via_pd_to_position_target():
    sim = make_sim()
    sim.reset_all()
    sim.write_qpos(np.zeros((N, 2), dtype=np.float32))
    sim.write_qvel(np.zeros((N, 2), dtype=np.float32))
    target_ctrl = np.full((N, 2), 0.5, dtype=np.float32)  # target = 0.5*pi
    sim.write_target(np.full((N, 2), 100.0, dtype=np.float32))  # disable termination
    for _ in range(60):
        sim.advance(target_ctrl)
    expected = 0.5 * math.pi
    err = np.abs(sim.read_qpos() - expected)
    assert err.max() < 0.2, f"PD did not converge; max err = {err.max():.3f}"


def test_actuator_targets_persist_until_overwritten():
    sim = make_sim()
    sim.reset_all()
    qt = np.full((N, 2), 0.3, dtype=np.float32)
    sim.set_actuator_targets(qt)
    # mj_step with no tau uses actuator targets through gainprm/biasprm.
    sim.write_qpos(np.zeros((N, 2), dtype=np.float32))
    sim.write_qvel(np.zeros((N, 2), dtype=np.float32))
    sim.mj_step()
    expected_qfrc = np.clip(DEFAULT_KP * qt, -DEFAULT_MAX_TORQUE, DEFAULT_MAX_TORQUE)
    assert np.allclose(sim.qfrc_actuator(), expected_qfrc, atol=1e-4)


def test_actuator_gainprm_biasprm_runtime_mutable():
    sim = make_sim()
    sim.actuator_gainprm[:] = 5.0
    sim.actuator_biasprm[:, 1] = -5.0
    sim.actuator_biasprm[:, 2] = -0.5
    assert np.allclose(sim.actuator_gainprm, 5.0)


# ---------------------------------------------------------------- reset semantics

def test_no_auto_reset_done_envs_remain_terminal():
    sim = make_sim()
    sim.reset_all()
    sim.write_target(sim.compute_site_pos("ee"))  # immediate success
    sim.advance(np.zeros((N, 2), dtype=np.float32))
    # Without reset, t_steps does not advance and reward stays 0 for terminal envs.
    t_after_done = sim.t_steps.copy()
    r2, _, done2, _ = sim.advance(np.zeros((N, 2), dtype=np.float32))
    assert done2.all()
    assert np.array_equal(sim.t_steps, t_after_done)
    assert np.allclose(r2, 0.0)


def test_reset_idx_clears_terminal():
    sim = make_sim()
    sim.reset_all()
    sim.write_target(sim.compute_site_pos("ee"))
    _, _, done, _ = sim.advance(np.zeros((N, 2), dtype=np.float32))
    assert done.all()
    sim.reset_idx(np.flatnonzero(done))
    assert sim.t_steps.sum() == 0


def test_truncation_after_max_steps():
    sim = make_sim()
    sim.reset_all()
    sim.write_target(np.full((N, 2), 100.0, dtype=np.float32))
    a = np.zeros((N, 2), dtype=np.float32)
    last_done = None
    for _ in range(MAX_STEPS):
        _, _, last_done, _ = sim.advance(a)
    assert last_done.all()


# ---------------------------------------------------------------- I/O round-trips

def test_write_read_round_trips():
    sim = make_sim()
    sim.reset_all()
    qpos = np.random.uniform(-math.pi, math.pi, size=(N, 2)).astype(np.float32)
    qvel = np.random.randn(N, 2).astype(np.float32)
    sim.write_qpos(qpos)
    sim.write_qvel(qvel)
    assert np.allclose(sim.read_qpos(), qpos, atol=1e-5)
    assert np.allclose(sim.read_qvel(), qvel, atol=1e-5)
    tgt = np.random.randn(N, 2).astype(np.float32)
    sim.write_target(tgt)
    assert np.allclose(sim.target_xy(), tgt)


def test_read_methods_return_copies():
    sim = make_sim()
    sim.reset_all()
    a = sim.read_qpos()
    a[:] = 99.0
    assert not np.allclose(sim.read_qpos(), 99.0)


# ---------------------------------------------------------------- determinism / errors

def test_seed_determinism():
    a = MuJoCoSim(n_parallel=N, seed=7)
    b = MuJoCoSim(n_parallel=N, seed=7)
    assert np.allclose(a.read_qpos(), b.read_qpos())
    assert np.allclose(a.target_xy(), b.target_xy())
    ctrl = np.full((N, 2), 0.3, dtype=np.float32)
    r_a, o_a, d_a, _ = a.advance(ctrl)
    r_b, o_b, d_b, _ = b.advance(ctrl)
    assert np.allclose(r_a, r_b) and np.allclose(o_a, o_b) and np.array_equal(d_a, d_b)


def test_advance_after_shutdown_raises():
    sim = make_sim()
    sim.shutdown()
    with pytest.raises(RuntimeError):
        sim.advance(np.zeros((N, 2), dtype=np.float32))


def test_advance_rejects_non_ndarray():
    sim = make_sim()
    sim.reset_all()
    with pytest.raises(TypeError):
        sim.advance([[0.0, 0.0]] * N)


def test_invalid_shapes_raise():
    sim = make_sim()
    sim.reset_all()
    with pytest.raises(ValueError):
        sim.advance(np.zeros((N + 1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        sim.write_qpos(np.zeros((N + 1, 2), dtype=np.float32))


def test_compute_site_pos_unknown_name_raises():
    sim = make_sim()
    with pytest.raises(KeyError):
        sim.compute_site_pos("hand")
    with pytest.raises(KeyError):
        sim.compute_jacp("hand")


def test_jacp_basic_kinematics():
    sim = make_sim()
    sim.reset_all()
    sim.write_qpos(np.zeros((N, 2), dtype=np.float32))
    J = sim.compute_jacp("ee")
    assert J.shape == (N, 2, 2)
    # At q=(0,0): de/dq1 = (0, L1+L2); de/dq2 = (0, L2)
    expected = np.array([[0.0, 0.0], [L1 + L2, L2]], dtype=np.float32)
    for i in range(N):
        assert np.allclose(J[i], expected, atol=1e-5)


def test_contact_force_at_full_extension():
    sim = make_sim()
    sim.reset_all()
    sim.write_qpos(np.zeros((N, 2), dtype=np.float32))
    cf = sim.contact_force()
    assert cf.shape == (N,) and cf.dtype == np.float32
    # ‖ee‖ = 2.0; threshold 0.99*2 = 1.98 → contact_force = 0.02
    assert np.all(cf > 0.0)


def test_reward_is_negative_distance_to_goal():
    sim = make_sim()
    sim.reset_all()
    r, _, _, _ = sim.advance(np.zeros((N, 2), dtype=np.float32))
    assert (r <= 0).all()
    assert np.abs(r).max() < (2 * WORKSPACE_RADIUS)


def test_reward_zero_for_terminal_envs():
    sim = make_sim()
    sim.reset_all()
    sim.write_target(sim.compute_site_pos("ee"))
    sim.advance(np.zeros((N, 2), dtype=np.float32))  # terminate
    r2, _, _, _ = sim.advance(np.zeros((N, 2), dtype=np.float32))
    assert np.allclose(r2, 0.0)
