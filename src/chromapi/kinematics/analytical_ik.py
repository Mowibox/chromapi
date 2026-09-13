"""Analytical leg IK for Chromapi."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciRotation

from chromapi.kinematics.topology import JOINT_SUFFIXES, LEG_NAMES, joint_name
from chromapi.kinematics.types import FootTargets, JointDict, Vector3

if TYPE_CHECKING:
    from chromapi.kinematics.kromatics import Kromatics

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyticalIK",
    "CalibrationReport",
    "LegGeometry",
    "leg_fk",
    "leg_ik",
    "reach_fraction",
]

# Keep the reachable workspace a little smaller than d = l2 + l3, to avoid the full extension singularity. 
WORKSPACE_MARGIN = 0.95

# ============================================================================================
# Pure closed-form math
# ============================================================================================


def leg_fk(q1: float, q2: float, q3: float, l1: float, l2: float, l3: float) -> Vector3:
    """Forward kinematics of one 3-DOF leg."""
    reach = l1 + l2 * np.cos(q2) + l3 * np.cos(q2 + q3)
    z = -l2 * np.sin(q2) - l3 * np.sin(q2 + q3)
    return np.array([np.cos(q1) * reach, np.sin(q1) * reach, z])


def leg_ik(
    p_hip_frame: Vector3,
    l1: float,
    l2: float,
    l3: float,
    knee_up: bool = True,
) -> Optional[Tuple[float, float, float]]:
    """Closed-form inverse kinematics of one 3-DOF leg."""
    x, y, z = (float(v) for v in p_hip_frame)
    q1 = np.arctan2(y, x)

    ell = np.hypot(x, y) - l1
    d = np.hypot(ell, z)
    if d > l2 + l3 or d < abs(l2 - l3):
        return None

    c3 = (d**2 - l2**2 - l3**2) / (2.0 * l2 * l3)
    c3 = float(np.clip(c3, -1.0, 1.0)) # ensuring numerical stability for arctan2
    s3_mag = np.sqrt(max(0.0, 1.0 - c3**2))
    s3 = s3_mag if knee_up else -s3_mag
    q3 = float(np.arctan2(s3, c3))
    q2 = float(np.arctan2(-z, ell) - np.arctan2(l3 * s3, l2 + l3 * c3))
    return q1, q2, q3


def reach_fraction(p_hip_frame: Vector3, l1: float, l2: float, l3: float) -> float:
    """Fraction of max knee extension used to reach ``p_hip_frame``."""
    x, y, z = (float(v) for v in p_hip_frame)
    ell = np.hypot(x, y) - l1
    d = np.hypot(ell, z)
    return float(d / (l2 + l3))


# ============================================================================================
# Per-leg geometry & 4-leg wrapper
# ============================================================================================


@dataclass
class LegGeometry:
    """One leg's segment lengths and the pose of its hip frame H in some outer frame."""
    l1: float
    l2: float
    l3: float
    hip_origin: Vector3
    hip_rotation: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    knee_up: bool = True

    def to_hip_frame(self, p_outer: Vector3) -> Vector3:
        """Convert a point from the outer frame to this leg's hip frame H."""
        return self.hip_rotation.T @ (np.asarray(p_outer, dtype=float) - self.hip_origin)

    def to_outer_frame(self, p_hip_frame: Vector3) -> Vector3:
        """Convert a point from this leg's hip frame H to the outer frame."""
        result = self.hip_origin + self.hip_rotation @ np.asarray(p_hip_frame, dtype=float)
        return np.asarray(result, dtype=np.float64)

    def as_vector(self) -> npt.NDArray[np.float64]:
        """Pack as ``(l1, l2, l3, *hip_origin, *hip_rotation.as_rotvec())`` for the optimizer."""
        rotvec = SciRotation.from_matrix(self.hip_rotation).as_rotvec()
        packed = np.concatenate([[self.l1, self.l2, self.l3], self.hip_origin, rotvec])
        return np.asarray(packed, dtype=np.float64)

    @classmethod
    def from_vector(cls, x: npt.NDArray[np.float64], knee_up: bool = True) -> "LegGeometry":
        """Inverse of :meth:`as_vector`."""
        l1, l2, l3 = x[0:3]
        hip_origin = np.asarray(x[3:6], dtype=float)
        hip_rotation = SciRotation.from_rotvec(x[6:9]).as_matrix()
        return cls(l1=float(l1), l2=float(l2), l3=float(l3), hip_origin=hip_origin,
                    hip_rotation=hip_rotation, knee_up=knee_up)


@dataclass
class CalibrationReport:
    """Per-leg fit quality from :meth:`AnalyticalIK.calibrate`."""
    rms_error_m: Dict[str, float]
    max_error_m: Dict[str, float]
    n_samples: Dict[str, int]

    def ok(self, tolerance_m: float = 0.002) -> bool:
        """True if every leg's ``max_error_m`` is within ``tolerance_m`` (default 2 mm)."""
        return all(err <= tolerance_m for err in self.max_error_m.values())

    def summary(self) -> str:
        """One line per leg: ``leg: rms=... mm  max=... mm  (n=...)``."""
        lines = [
            f"{leg}: rms={1000 * self.rms_error_m[leg]:.3f} mm  "
            f"max={1000 * self.max_error_m[leg]:.3f} mm  (n={self.n_samples[leg]})"
            for leg in self.rms_error_m
        ]
        return "\n".join(lines)


_Sample = Tuple[Tuple[float, float, float], Vector3]


class AnalyticalIK:
    """Closed-form IK/FK for all 4 Chromapi legs."""

    def __init__(self, geometries: Dict[str, LegGeometry]) -> None:
        """Wrap one :class:`LegGeometry` per leg (must cover every entry of ``LEG_NAMES``)."""
        missing = set(LEG_NAMES) - set(geometries)
        if missing:
            raise ValueError(f"AnalyticalIK is missing geometry for legs: {sorted(missing)}")
        self.geometries = dict(geometries)

    def forward_kinematics_leg(self, leg: str, q1: float, q2: float, q3: float) -> Vector3:
        """Foot position for one leg, in the chassis frame."""
        geom = self.geometries[leg]
        return geom.to_outer_frame(leg_fk(q1, q2, q3, geom.l1, geom.l2, geom.l3))

    def forward_kinematics(self, q: JointDict) -> FootTargets:
        """Foot positions for all 4 legs, in the chassis frame."""
        return {
            leg: self.forward_kinematics_leg(
                leg, q[joint_name(leg, 1)], q[joint_name(leg, 2)], q[joint_name(leg, 3)]
            )
            for leg in LEG_NAMES
        }

    def inverse_kinematics_leg(
        self, leg: str, p_outer: Vector3
    ) -> Optional[Tuple[float, float, float]]:
        """Closed-form ``(q1, q2, q3)`` for one leg's foot target, or ``None`` if unreachable."""
        geom = self.geometries[leg]
        p_hip = geom.to_hip_frame(p_outer)
        return leg_ik(p_hip, geom.l1, geom.l2, geom.l3, knee_up=geom.knee_up)

    def inverse_kinematics(self, foot_targets: FootTargets) -> Tuple[JointDict, bool]:
        """Closed-form joint angles for a set of foot targets (chassis frame)."""
        q: JointDict = {}
        converged = True
        for leg, target in foot_targets.items():
            solution = self.inverse_kinematics_leg(leg, target)
            if solution is None:
                converged = False
                continue
            q1, q2, q3 = solution
            for suffix, value in zip(JOINT_SUFFIXES, (q1, q2, q3)):
                q[joint_name(leg, suffix)] = value
        return q, converged

    # -- calibration ---------------------------------------------------------------------

    @staticmethod
    def _residuals(
        x_by_leg: Dict[str, npt.NDArray[np.float64]],
        samples: Dict[str, List[_Sample]],
        legs: Sequence[str],
    ) -> npt.NDArray[np.float64]:
        out = []
        for leg in legs:
            geom = LegGeometry.from_vector(x_by_leg[leg])
            for (q1, q2, q3), target in samples[leg]:
                pred = geom.to_outer_frame(leg_fk(q1, q2, q3, geom.l1, geom.l2, geom.l3))
                out.append(pred - np.asarray(target, dtype=float))
        return np.concatenate(out) if out else np.zeros(0)

    @classmethod
    def calibrate(
        cls,
        samples: Dict[str, List[_Sample]],
        initial_guess: Dict[str, LegGeometry],
        knee_up: Optional[Dict[str, bool]] = None,
    ) -> Tuple["AnalyticalIK", CalibrationReport]:
        """Fit a :class:`LegGeometry` per leg from forward-kinematics samples.

        Nonlinear least squares (per leg, independently) over ``(l1, l2, l3, hip_origin,
        hip_rotation)``, seeded from ``initial_guess``. Only converges to the true geometry from
        a reasonable starting point - Always check :meth:`CalibrationReport.ok` before trusting the result.

        Args:
            samples: Per leg, a list of ``((q1, q2, q3), foot_position_outer_frame)`` pairs.
            initial_guess: Per-leg starting :class:`LegGeometry` for the optimizer.
            knee_up: Per-leg branch selection for the returned :class:`AnalyticalIK` - defaults
                to whatever ``initial_guess`` specifies.

        Returns:
            ``(AnalyticalIK, CalibrationReport)``.

        """
        legs = list(samples)
        missing = set(legs) - set(initial_guess)
        if missing:
            raise ValueError(f"calibrate: missing initial_guess for legs: {sorted(missing)}")

        x0 = np.concatenate([initial_guess[leg].as_vector() for leg in legs])
        sizes = [9] * len(legs)

        def unpack(x: npt.NDArray[np.float64]) -> Dict[str, npt.NDArray[np.float64]]:
            offsets = np.cumsum([0] + sizes)
            return {leg: x[offsets[i]:offsets[i + 1]] for i, leg in enumerate(legs)}

        def residuals(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return cls._residuals(unpack(x), samples, legs)

        result = least_squares(residuals, x0, method="lm", max_nfev=50_000)
        fitted = unpack(result.x)

        geometries: Dict[str, LegGeometry] = {}
        rms_error: Dict[str, float] = {}
        max_error: Dict[str, float] = {}
        n_samples: Dict[str, int] = {}
        for leg in legs:
            branch = knee_up[leg] if knee_up and leg in knee_up else initial_guess[leg].knee_up
            geom = LegGeometry.from_vector(fitted[leg], knee_up=branch)
            geometries[leg] = geom
            errors = [
                np.linalg.norm(
                    geom.to_outer_frame(leg_fk(q1, q2, q3, geom.l1, geom.l2, geom.l3))
                    - np.asarray(target, dtype=float)
                )
                for (q1, q2, q3), target in samples[leg]
            ]
            rms_error[leg] = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
            max_error[leg] = float(np.max(errors)) if errors else float("nan")
            n_samples[leg] = len(errors)

        report = CalibrationReport(rms_error_m=rms_error, max_error_m=max_error, n_samples=n_samples)
        if not report.ok():
            logger.warning(
                "AnalyticalIK.calibrate: fit residual exceeds 2 mm for at least one leg - see "
                "CalibrationReport before trusting this solver for real foot placement:\n%s",
                report.summary(),
            )
        return cls(geometries), report

    @classmethod
    def from_kromatics(
        cls,
        kinematics: "Kromatics",
        n_samples: int = 300,
        joint_range: float = 1.0,
        seed: int = 0,
        knee_up: Optional[Dict[str, bool]] = None,
    ) -> Tuple["AnalyticalIK", CalibrationReport]:
        """Calibrate against a live :class:`~chromapi.kinematics.kromatics.Kromatics` model.

        Seeds the initial guess from the URDF's own hip transform and total leg reach at the
        all-zero configuration, then fits from ``n_samples`` random configurations per leg drawn
        from ``[-joint_range, +joint_range]`` radians. ``knee_up`` defaults to matching
        :data:`~chromapi.kinematics.poses.WAKE_UP_POSE`'s own knee sign. Check the returned
        :class:`CalibrationReport` before trusting the result for more than a QP warm start.
        """
        from chromapi.kinematics.kromatics import (
            _CHASSIS_FRAME,  # local: avoid import cycle
        )
        from chromapi.kinematics.poses import WAKE_UP_POSE

        rng = np.random.default_rng(seed)
        initial_guess: Dict[str, LegGeometry] = {}
        default_knee_up = {leg: WAKE_UP_POSE[joint_name(leg, 3)] >= 0.0 for leg in LEG_NAMES}
        knee_up = knee_up or default_knee_up

        for leg in LEG_NAMES:
            hip_transform = kinematics.robot.get_T_a_b(_CHASSIS_FRAME, joint_name(leg, 1))
            hip_origin = hip_transform[:3, 3].copy()
            hip_rotation = hip_transform[:3, :3].copy()
            initial_guess[leg] = LegGeometry(
                l1=0.0, l2=0.0, l3=0.0,  # filled in below, once total_reach is known
                hip_origin=hip_origin, hip_rotation=hip_rotation, knee_up=knee_up[leg],
            )

        zero_q = {name: 0.0 for name in (
            n for leg in LEG_NAMES for n in (joint_name(leg, 1), joint_name(leg, 2), joint_name(leg, 3))
        )}
        zero_feet = kinematics.forward_kinematics(zero_q)
        for leg in LEG_NAMES:
            geom = initial_guess[leg]
            total_reach = float(np.linalg.norm(zero_feet[leg] - geom.hip_origin))
            third = max(total_reach / 3.0, 1e-3)
            geom.l1 = geom.l2 = geom.l3 = third

        samples: Dict[str, List[_Sample]] = {leg: [] for leg in LEG_NAMES}
        for leg in LEG_NAMES:
            for _ in range(n_samples):
                q1, q2, q3 = rng.uniform(-joint_range, joint_range, size=3)
                q = dict(zero_q)
                q[joint_name(leg, 1)] = q1
                q[joint_name(leg, 2)] = q2
                q[joint_name(leg, 3)] = q3
                foot = kinematics.forward_kinematics(q)[leg]
                samples[leg].append(((q1, q2, q3), foot))

        return cls.calibrate(samples, initial_guess, knee_up=knee_up)

    # -- validation -------------------------------------------------------------------

    def validate(
        self,
        forward_kinematics: Callable[[str, float, float, float], Vector3],
        n_samples: int = 200,
        joint_range: float = 1.0,
        seed: int = 1,
    ) -> CalibrationReport:
        """Independent residual check against a ground-truth FK, on fresh random configurations."""
        rng = np.random.default_rng(seed)
        rms_error: Dict[str, float] = {}
        max_error: Dict[str, float] = {}
        n_used: Dict[str, int] = {}
        for leg in self.geometries:
            errors = []
            for _ in range(n_samples):
                q1, q2, q3 = rng.uniform(-joint_range, joint_range, size=3)
                truth = forward_kinematics(leg, q1, q2, q3)
                pred = self.forward_kinematics_leg(leg, q1, q2, q3)
                errors.append(np.linalg.norm(pred - np.asarray(truth, dtype=float)))
            rms_error[leg] = float(np.sqrt(np.mean(np.square(errors))))
            max_error[leg] = float(np.max(errors))
            n_used[leg] = len(errors)
        return CalibrationReport(rms_error_m=rms_error, max_error_m=max_error, n_samples=n_used)
