# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the attitude module."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chromapi.estimation.attitude import (
    ImuMount,
    projected_gravity,
    quat_from_roll_pitch,
    quat_multiply,
    remove_yaw,
    roll_pitch,
    tilt,
)


def _quat(yaw: float, pitch: float, roll: float) -> tuple:
    """(w, x, y, z) of the ZYX Euler angles [rad]."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return (w, x, y, z)


ANGLES = [(0.0, 0.0, 0.0), (1.2, 0.3, -0.2), (-2.5, -0.6, 0.9), (0.4, 1.2, 2.8)]


@pytest.mark.parametrize("yaw, pitch, roll", ANGLES)
def test_projected_gravity_matches_rotation(yaw: float, pitch: float, roll: float) -> None:
    """R(q).T (0, 0, -1), computed by scipy."""
    q = _quat(yaw, pitch, roll)
    w, x, y, z = q
    expected = Rotation.from_quat([x, y, z, w]).inv().apply([0.0, 0.0, -1.0])
    np.testing.assert_allclose(projected_gravity(q), expected, atol=1e-12)


@pytest.mark.parametrize("yaw, pitch, roll", ANGLES)
def test_roll_pitch_ignore_yaw(yaw: float, pitch: float, roll: float) -> None:
    """Roll/pitch come back from any yaw; the tilt matches the angle to the vertical."""
    q = _quat(yaw, pitch, roll)
    np.testing.assert_allclose(roll_pitch(q), (roll, pitch), atol=1e-12)
    np.testing.assert_allclose(roll_pitch(remove_yaw(q)), (roll, pitch), atol=1e-12)
    assert tilt(q) == pytest.approx(math.acos(math.cos(roll) * math.cos(pitch)), abs=1e-9)


def test_quat_from_roll_pitch_matches_scipy() -> None:
    """Ry(pitch) · Rx(roll), same convention as ZYX with zero yaw."""
    np.testing.assert_allclose(quat_from_roll_pitch(0.4, -0.7), _quat(0.0, -0.7, 0.4), atol=1e-12)


def test_tilt_upside_down() -> None:
    """Pi when flipped, whatever the yaw."""
    assert tilt(_quat(0.8, 0.0, math.pi)) == pytest.approx(math.pi)


def test_mount_removes_board_offset() -> None:
    """A board tilted on a level chassis reads level once calibrated, and keeps true tilts."""
    board = quat_from_roll_pitch(0.03, -0.05)
    mount = ImuMount.from_level_samples([quat_multiply(_quat(yaw, 0.0, 0.0), board) for yaw in (0.0, 1.0, 2.0)])
    chassis = _quat(0.5, 0.2, -0.1)
    measured = quat_multiply(chassis, board)
    np.testing.assert_allclose(roll_pitch(mount.apply(measured)), (-0.1, 0.2), atol=1e-12)


def test_mount_without_samples_raises() -> None:
    with pytest.raises(ValueError):
        ImuMount.from_level_samples([])


def test_mount_vector_matches_quaternion() -> None:
    """Accelerometer gravity rotated by the mount agrees with the corrected quaternion."""
    mount = ImuMount(quat_from_roll_pitch(0.04, -0.02))
    q_sensor = _quat(0.7, 0.3, -0.2)
    g_chassis = mount.apply_vector(projected_gravity(q_sensor))
    np.testing.assert_allclose(g_chassis, projected_gravity(mount.apply(q_sensor)), atol=1e-12)
