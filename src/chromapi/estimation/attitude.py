"""Attitude helpers for Chromapi - tilt from the IMU quaternion.

The STM32 runs the Mahony filter and sends a unit quaternion
(w, x, y, z), world Z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import numpy.typing as npt

#: Unit quaternion (w, x, y, z).
Quaternion = Tuple[float, float, float, float]

IDENTITY_QUAT: Quaternion = (1.0, 0.0, 0.0, 0.0)


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    """Hamilton product between two quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conjugate(q: Sequence[float]) -> Quaternion:
    """Inverse of a unit quaternion."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def projected_gravity(q: Sequence[float]) -> npt.NDArray[np.float64]:
    """Unit gravity vector in the body frame, (0, 0, -1) when level."""
    w, x, y, z = q
    return -np.array(
        [
            2.0 * (x * z - w * y),
            2.0 * (w * x + y * z),
            w * w - x * x - y * y + z * z,
        ]
    )


def roll_pitch_from_gravity(g_body: Sequence[float]) -> Tuple[float, float]:
    """(roll, pitch) [rad] of the ZYX (yaw-pitch-roll) convention, from projected gravity."""
    gx, gy, gz = g_body
    roll = math.atan2(-gy, -gz)
    pitch = math.atan2(gx, math.hypot(gy, gz))
    return roll, pitch


def roll_pitch(q: Sequence[float]) -> Tuple[float, float]:
    """(roll, pitch) [rad] of ``q`` - same as ZYX Euler angles, without the yaw."""
    return roll_pitch_from_gravity(projected_gravity(q))


def tilt(q: Sequence[float]) -> float:
    """Angle [rad] between the body Z axis and the vertical."""
    return math.acos(float(np.clip(-projected_gravity(q)[2], -1.0, 1.0)))


def quat_from_roll_pitch(roll: float, pitch: float) -> Quaternion:
    """Quaternion of ``Ry(pitch).Rx(roll)``."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    return (cp * cr, cp * sr, sp * cr, -sp * sr)


def remove_yaw(q: Sequence[float]) -> Quaternion:
    """Removing the yaw component: same tilt, safe to feed a controller or a policy."""
    return quat_from_roll_pitch(*roll_pitch(q))


@dataclass(frozen=True)
class ImuMount:
    """Fixed sensor-to-chassis tilt offset (board not perfectly level on the chassis)."""

    offset: Quaternion = IDENTITY_QUAT

    @classmethod
    def from_level_samples(cls, quats: Iterable[Sequence[float]]) -> "ImuMount":
        """Calibrate from quaternions recorded with the chassis level and still."""
        gravities = [projected_gravity(q) for q in quats]
        if not gravities:
            raise ValueError("ImuMount.from_level_samples: no samples")
        g_mean = np.mean(gravities, axis=0)  # averaged as vectors: no quaternion sign ambiguity
        return cls(quat_from_roll_pitch(*roll_pitch_from_gravity(g_mean / np.linalg.norm(g_mean))))

    def apply(self, q: Sequence[float]) -> Quaternion:
        """Chassis orientation from the sensor one: ``q ⊗ offset⁻¹``."""
        return quat_multiply(q, quat_conjugate(self.offset))

    def apply_vector(self, v: Sequence[float]) -> Tuple[float, float, float]:
        """Sensor-frame vector (gyro, accel) expressed in the chassis frame: ``R(offset) v``."""
        w, x, y, z = self.offset
        rotated = quat_multiply(quat_multiply(self.offset, (0.0, *v)), (w, -x, -y, -z))
        return (rotated[1], rotated[2], rotated[3])
