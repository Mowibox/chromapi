"""Robot topology - leg/joint naming."""

from __future__ import annotations

from typing import Dict, Tuple, Union

#: Leg tags as named in the URDF/MJCF (``model/urdf/robot.urdf``, ``model/mjcf/robot.xml``).
LEG_NAMES: Tuple[str, str, str, str] = ("tl", "tr", "bl", "br")

#: Per-leg joint suffixes.
JOINT_SUFFIXES: Tuple[str, str, str] = ("1", "2", "3")


def joint_name(leg: str, suffix: Union[int, str]) -> str:
    """Build a joint name (e.g. ``joint_name("tl", 1) == "tl_1"``)."""
    return f"{leg}_{suffix}"


#: The 12 joint names.
JOINT_NAMES: Tuple[str, ...] = tuple(
    joint_name(leg, s) for leg in LEG_NAMES for s in JOINT_SUFFIXES
)

#: Foot contact frame name per leg.
FOOT_FRAME_NAMES: Dict[str, str] = {leg: f"{leg}_foot" for leg in LEG_NAMES}
