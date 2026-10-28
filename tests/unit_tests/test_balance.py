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


def _settle(move, q, ticks, dt=0.01):
    """Step ``move`` from targets ``q``, level IMU, all feet assumed down."""
    from chromapi.chromapi import MotorCommand, RobotState

    state = RobotState(joint_positions=dict(q))  # level, identity IMU
    command = MotorCommand(target_angles=dict(q))
    for _ in range(ticks):
        move.step(state, command, dt)
    return command.target_angles


def _feet_and_com(move, q):
    """Feet [m] and CoM [m] of ``q`` in the chassis frame, as the QP sees them."""
    import numpy as np

    from chromapi.kinematics.topology import FOOT_FRAME_NAMES, LEG_NAMES

    robot = move._est.robot
    for name, value in q.items():
        robot.set_joint(name, value)
    robot.set_T_world_fbase(np.eye(4))
    robot.update_kinematics()
    return {leg: robot.get_T_world_frame(FOOT_FRAME_NAMES[leg])[:3, 3].copy() for leg in LEG_NAMES}, robot.com_world().copy()


def test_body_pose_moves_the_com_with_the_feet_planted():
    """set_body_pose(xyz) shifts the CoM relative to the feet; zero brings the start pose back."""
    import numpy as np

    from chromapi.kinematics.poses import WAKE_UP_POSE
    from chromapi.locomotion.balance import BalanceMove

    move = BalanceMove(dt=0.01)
    move.on_start()
    start = _settle(move, WAKE_UP_POSE, 30)  # knees pulled back inside their limits
    feet0, com0 = _feet_and_com(move, start)
    move.set_body_pose((0.02, -0.01, 0.0))
    shifted = _settle(move, start, 100)
    feet1, com1 = _feet_and_com(move, shifted)
    centre0, centre1 = np.mean(list(feet0.values()), axis=0), np.mean(list(feet1.values()), axis=0)
    assert (com1 - centre1)[:2] - (com0 - centre0)[:2] == pytest.approx([0.02, -0.01], abs=4e-3)  # joint limits
    for leg in feet1:
        assert np.linalg.norm((feet1[leg] - centre1) - (feet0[leg] - centre0)) < 2e-3
    move.set_body_pose()
    assert _settle(move, shifted, 100) == pytest.approx(start, abs=2e-3)


def test_a_lifted_leg_follows_its_joints_and_the_others_stay_planted():
    """lift(leg, q) writes the leg's joints as given, faster than the rate limit; lift(None) lands it."""
    import numpy as np

    from chromapi.kinematics.poses import WAKE_UP_POSE
    from chromapi.locomotion.balance import BalanceMove

    move = BalanceMove(dt=0.01)
    move.on_start()
    start = _settle(move, WAKE_UP_POSE, 30)
    feet0, _ = _feet_and_com(move, start)
    up = {"tl_1": start["tl_1"], "tl_2": start["tl_2"] + 0.4, "tl_3": start["tl_3"]}
    move.lift("tl", up)
    lifted = _settle(move, start, 20)
    assert {j: lifted[j] for j in up} == pytest.approx(up)
    feet1, _ = _feet_and_com(move, lifted)
    for leg in ("tr", "bl", "br"):  # relative to each other: the chassis may move above them
        assert np.linalg.norm((feet1[leg] - feet1["br"]) - (feet0[leg] - feet0["br"])) < 2e-3
    move.lift(None)
    landed = _settle(move, lifted, 100)
    assert landed["tl_2"] == pytest.approx(start["tl_2"], abs=0.02)
