"""Shared array/dict type aliases for the kinematics layer."""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
import numpy.typing as npt

#: A plain 3-vector (position, in meters) - most commonly a foot position.
Vector3 = npt.NDArray[np.float64]

#: A 3-element joint-angle vector, accepted as a plain sequence or a numpy array.
QVector = Union[Sequence[float], npt.NDArray[np.float64]]

#: The 12 joint angles (radians), keyed by joint name.
JointDict = Dict[str, float]

#: One foot position (in chassis frame) per leg.
FootTargets = Dict[str, Vector3]
