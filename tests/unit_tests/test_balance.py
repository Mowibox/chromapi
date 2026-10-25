# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the balance module."""

import pytest

from chromapi.locomotion.balance import stability_margin

SQUARE = [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]


def test_margin_square():
    assert stability_margin(SQUARE, (0.0, 0.0)) == pytest.approx(0.1)
    assert stability_margin(SQUARE, (0.07, 0.0)) == pytest.approx(0.03)
    assert stability_margin(SQUARE, (0.15, 0.0)) == pytest.approx(-0.05)


def test_margin_order_independent():
    shuffled = [SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3]]
    assert stability_margin(shuffled, (0.02, -0.03)) == pytest.approx(0.07)


def test_margin_degenerate():
    assert stability_margin([(0.0, 0.0), (0.2, 0.0)], (0.1, 0.05)) == pytest.approx(-0.05)
    assert stability_margin([(0.0, 0.0)], (0.03, 0.04)) == pytest.approx(-0.05)


def test_sent_targets_are_rate_limited_on_a_tilted_floor():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from chromapi.chromapi import MotorCommand, RobotState
    from chromapi.kinematics.poses import WAKE_UP_POSE
    from chromapi.kinematics.topology import JOINT_NAMES
    from chromapi.locomotion.balance import BalanceMove

    move = BalanceMove(dt=0.01)
    state = RobotState(joint_positions=dict(WAKE_UP_POSE))
    # Chassis 10 deg nose-down, IMU in the URDF "imu" frame (yaw -90 deg in the chassis).
    x, y, z, w = (Rotation.from_euler("x", 10, degrees=True) * Rotation.from_euler("z", -90, degrees=True)).as_quat()
    state.imu_quat_wxyz = (w, x, y, z)
    sent = dict(WAKE_UP_POSE)
    for _ in range(30):
        command = MotorCommand(target_angles=dict(sent))
        move.step(state, command, 0.01)
        assert max(abs(command.target_angles[j] - sent[j]) for j in JOINT_NAMES) <= move.gains.max_joint_speed * 0.01 + 1e-9
        sent = dict(command.target_angles)
    assert move.rpy_cmd[0] < 0  # nose-down measured -> commands nose-up
    assert any(abs(sent[j] - WAKE_UP_POSE[j]) > 1e-3 for j in JOINT_NAMES)  # and actually moves

