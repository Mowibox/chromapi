# SPDX-License-Identifier: Apache-2.0
"""Named joint configurations."""

from __future__ import annotations

from typing import Dict

from chromapi.kinematics.topology import (
    JOINT_NAMES,
    JOINT_SUFFIXES,
    LEG_NAMES,
    joint_name,
)

#: Every joint at 0 [rad]: the zero of the URDF, legs straight.
ZERO_POSE: Dict[str, float] = dict.fromkeys(JOINT_NAMES, 0.0)  # [rad]

def _symmetric_pose(hip_yaw: float, hip_pitch: float, knee_pitch: float) -> Dict[str, float]:
    """Build a 12-joint pose from one (hip yaw, hip pitch, knee pitch) triple [rad]."""
    pose: Dict[str, float] = {}  # [rad]
    for leg in LEG_NAMES:
        for suffix, value in zip(JOINT_SUFFIXES, (hip_yaw, hip_pitch, knee_pitch)):
            if suffix == "1" and leg in ("bl", "br"):
                value = -value
            pose[joint_name(leg, suffix)] = value
    return pose

#: Joint angles [rad] of :meth:`~chromapi.chromapi.Chromapi.approach`.
APPROACH_POSE: Dict[str, float] = _symmetric_pose(0.65, 1.2, -1.7)  # [rad]
#: Joint angles [rad] of the stand reached by :meth:`~chromapi.chromapi.Chromapi.wake_up`.
WAKE_UP_POSE: Dict[str, float] = _symmetric_pose(0.65, 0.65, -1.63)  # [rad]
#: Joint angles [rad] of the crouch reached by :meth:`~chromapi.chromapi.Chromapi.rest`.
REST_POSE: Dict[str, float] = _symmetric_pose(0.65, 1.42, -1.72)  # [rad]