"""Analytical leg IK for Chromapi."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as SciRotation

from chromapi.kinematics.topology import JOINT_SUFFIXES, LEG_NAMES, joint_name
from chromapi.kinematics.types import FootTargets, JointDict, Vector3

if TYPE_CHECKING:
    from chromapi.kinematics.kromatics import Kromatics

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyticalIK",
    "CalibrationReport",
    "LegChain",
    "WORKSPACE_MARGIN",
]

# Keep the reachable workspace a little smaller than d = l2 + l3, to avoid the full extension singularity
WORKSPACE_MARGIN = 0.95

# ============================================================================================
# Exact per-leg chain & 4-leg wrapper
# ============================================================================================


def _rotation(axis: Vector3, angle: float) -> npt.NDArray[np.float64]:
    """Rotation matrix of ``angle`` [rad] about the unit vector ``axis``."""
    return np.asarray(SciRotation.from_rotvec(np.asarray(axis, dtype=float) * angle).as_matrix())


def _wrap(angle: float) -> float:
    """Wrap an angle [rad] to ``]-pi, pi]``."""
    return float(np.pi - (np.pi - angle) % (2.0 * np.pi))


def _angle2(v: npt.NDArray[np.float64]) -> float:
    return float(np.arctan2(v[1], v[0]))


def _rot2(v: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass
class LegChain:
    """Exact geometry of one leg, read from the URDF at the all-zero configuration.

    Nothing is assumed about the leg's shape: each joint is a
    rotation about its real axis line (``points[i]``, ``axes[i]``, chassis frame, joint at zero),
    so joint offsets, a bent tibia at ``q3 = 0`` and mirrored axes (tl/bl vs tr/br) are all exact.
    Closed-form IK only needs axis 1 perpendicular to axes 2 and 3, and axes 2 and 3 parallel.

    Attributes:
        points: Point on each joint axis (the joint origin), chassis frame [m].
        axes: Unit axis of each joint, chassis frame (URDF sign included).
        foot: Foot position at the all-zero configuration, chassis frame [m].
        lower: Joint lower limits ``(q1, q2, q3)`` [rad].
        upper: Joint upper limits ``(q1, q2, q3)`` [rad].
        workspace_margin: Largest usable fraction of full extension - the distance from joint 2
            to the foot may not exceed ``workspace_margin * (femur + tibia)``.

    """

    points: Tuple[Vector3, Vector3, Vector3]  # [m]
    axes: Tuple[Vector3, Vector3, Vector3]
    foot: Vector3  # [m]
    lower: Tuple[float, float, float] = (-np.pi, -np.pi, -np.pi)  # [rad]
    upper: Tuple[float, float, float] = (np.pi, np.pi, np.pi)  # [rad]
    workspace_margin: float = WORKSPACE_MARGIN

    def __post_init__(self) -> None:
        """Precompute the planar (radial, height) model of the q2/q3 sub-chain."""
        p1 = np.asarray(self.points[0], dtype=float)  # [m]
        e_h = np.asarray(self.axes[0], dtype=float)
        e_l = np.asarray(self.axes[1], dtype=float)
        if abs(float(e_h @ e_l)) > 1e-6 or abs(abs(float(e_l @ self.axes[2])) - 1.0) > 1e-6:
            raise ValueError("LegChain: needs axis 1 perpendicular to axes 2/3, axes 2/3 parallel")
        e_r = np.cross(e_l, e_h)  # (e_r, e_l, e_h) right-handed; q1 turns e_r toward e_l
        self._p1, self._e_r, self._e_l, self._e_h = p1, e_r, e_l, e_h

        def plane(p: Vector3) -> npt.NDArray[np.float64]:
            d = np.asarray(p, dtype=float) - p1  # [m]
            return np.array([float(d @ e_r), float(d @ e_h)])

        # A rotation about +e_l turns the (radial, height) plane clockwise: sigma = -1.
        self._sigma = [float(np.sign(np.cross(a, e_r) @ e_h)) for a in self.axes[1:]]
        self._j2 = plane(self.points[1])  # [m]
        self._femur = plane(self.points[2]) - self._j2  # [m]
        self._tibia = plane(self.foot) - plane(self.points[2])  # [m]
        self._lateral = float((np.asarray(self.foot, dtype=float) - p1) @ e_l)  # [m]
        self._radial_sign = 1.0 if plane(self.foot)[0] >= 0.0 else -1.0
        self._bend = _angle2(self._tibia) - _angle2(self._femur)  # [rad]

    def forward(self, q1: float, q2: float, q3: float) -> Vector3:
        """Exact foot position (chassis frame) [m] for joint angles ``q1..q3`` [rad]."""
        p = np.asarray(self.foot, dtype=float)  # [m]
        for point, axis, angle in zip(self.points[::-1], self.axes[::-1], (q3, q2, q1)):
            point = np.asarray(point, dtype=float)  # [m]
            p = point + _rotation(axis, angle) @ (p - point)
        return np.asarray(p, dtype=np.float64)

    def inverse(
        self, target: Vector3, reference: Optional[Tuple[float, float, float]] = None
    ) -> Optional[Tuple[float, float, float]]:
        """Closed-form ``(q1, q2, q3)`` [rad] reaching ``target`` [m] within the joint limits, or ``None``.

        Of the (up to four) hip/knee branches, returns the one closest to ``reference`` (the
        previous solution, typically) - or the first valid one without a reference. A target
        needing more than ``workspace_margin`` of full leg extension is refused (``None``).
        """
        d = np.asarray(target, dtype=float) - self._p1  # [m]
        h = float(d @ self._e_h)  # [m]
        dr, dl = float(d @ self._e_r), float(d @ self._e_l)  # [m]
        r_sq = dr * dr + dl * dl - self._lateral**2  # [m²]
        if r_sq < 0.0:
            return None
        a, b = float(np.linalg.norm(self._femur)), float(np.linalg.norm(self._tibia))  # [m]
        s2, s3 = self._sigma
        candidates = []
        # Foot on the zero-configuration side of axis 1 first; the other side (foot tucked
        # behind the hip) is a valid, if unusual, solution too.
        for r in (self._radial_sign * np.sqrt(r_sq), -self._radial_sign * np.sqrt(r_sq)):
            q1 = _wrap(np.arctan2(dl, dr) - np.arctan2(self._lateral, r))  # [rad]
            reach = np.array([r, h]) - self._j2  # [m]
            if float(np.linalg.norm(reach)) > self.workspace_margin * (a + b):
                continue  # too close to full extension (singularity)
            cos_knee = (float(reach @ reach) - a * a - b * b) / (2.0 * a * b)
            if abs(cos_knee) > 1.0:
                continue
            for knee in (np.arccos(cos_knee), -np.arccos(cos_knee)):
                psi = knee - self._bend  # [rad], planar rotation of the tibia relative to the femur
                phi = _angle2(reach) - _angle2(self._femur + _rot2(self._tibia, psi))  # [rad]
                q = (q1, _wrap(phi / s2), _wrap(psi / s3))
                if all(lo - 1e-9 <= v <= hi + 1e-9 for v, lo, hi in zip(q, self.lower, self.upper)):
                    candidates.append(q)
        if not candidates:
            return None
        if reference is None:
            return candidates[0]
        return min(candidates, key=lambda q: float(np.linalg.norm(np.subtract(q, reference))))


@dataclass
class CalibrationReport:
    """Per-leg agreement between the closed-form model and a ground-truth FK."""
    rms_error_m: Dict[str, float]  # [m]
    max_error_m: Dict[str, float]  # [m]
    n_samples: Dict[str, int]

    def ok(self, tolerance_m: float = 0.002) -> bool:
        """True if every leg's ``max_error_m`` is within ``tolerance_m`` [m] (default 2 mm)."""
        return all(err <= tolerance_m for err in self.max_error_m.values())

    def summary(self) -> str:
        """One line per leg: ``leg: rms=... mm  max=... mm  (n=...)``."""
        lines = [
            f"{leg}: rms={1000 * self.rms_error_m[leg]:.3f} mm  "
            f"max={1000 * self.max_error_m[leg]:.3f} mm  (n={self.n_samples[leg]})"
            for leg in self.rms_error_m
        ]
        return "\n".join(lines)


class AnalyticalIK:
    """Closed-form IK/FK for all 4 Chromapi legs."""

    def __init__(
        self, chains: Dict[str, LegChain], reference: Optional[JointDict] = None
    ) -> None:
        """Wrap one :class:`LegChain` per leg (must cover every entry of ``LEG_NAMES``).

        ``reference`` seeds the knee-branch choice (the solution closest to the previous one is
        kept, so the branch stays continuous); defaults to :data:`WAKE_UP_POSE`.
        """
        missing = set(LEG_NAMES) - set(chains)
        if missing:
            raise ValueError(f"AnalyticalIK is missing geometry for legs: {sorted(missing)}")
        if reference is None:
            from chromapi.kinematics.poses import WAKE_UP_POSE

            reference = WAKE_UP_POSE
        self.chains = dict(chains)
        self._last: Dict[str, Tuple[float, float, float]] = {
            leg: tuple(float(reference[joint_name(leg, s)]) for s in JOINT_SUFFIXES)  # type: ignore[misc]
            for leg in LEG_NAMES
        }

    def forward_kinematics_leg(self, leg: str, q1: float, q2: float, q3: float) -> Vector3:
        """Foot position [m] for one leg, in the chassis frame, from joint angles [rad]."""
        return self.chains[leg].forward(q1, q2, q3)

    def forward_kinematics(self, q: JointDict) -> FootTargets:
        """Foot positions [m] for all 4 legs, in the chassis frame, from joint angles [rad]."""
        return {
            leg: self.forward_kinematics_leg(
                leg, q[joint_name(leg, 1)], q[joint_name(leg, 2)], q[joint_name(leg, 3)]
            )
            for leg in LEG_NAMES
        }

    def inverse_kinematics_leg(
        self, leg: str, p_outer: Vector3
    ) -> Optional[Tuple[float, float, float]]:
        """Closed-form ``(q1, q2, q3)`` [rad] for one leg's foot target [m], or ``None`` if unreachable."""
        solution = self.chains[leg].inverse(p_outer, reference=self._last[leg])
        if solution is not None:
            self._last[leg] = solution
        return solution

    def inverse_kinematics(self, foot_targets: FootTargets) -> Tuple[JointDict, bool]:
        """Closed-form joint angles [rad] for a set of foot targets (chassis frame) [m]."""
        q: JointDict = {}
        converged = True
        for leg, target in foot_targets.items():
            solution = self.inverse_kinematics_leg(leg, target)
            if solution is None:
                converged = False
                continue
            for suffix, value in zip(JOINT_SUFFIXES, solution):
                q[joint_name(leg, suffix)] = value
        return q, converged

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_kromatics(
        cls,
        kinematics: "Kromatics",
        n_samples: int = 300,
        seed: int = 0,
    ) -> Tuple["AnalyticalIK", CalibrationReport]:
        """Read each leg's exact chain from a live :class:`~chromapi.kinematics.kromatics.Kromatics`.

        Joint origins come from the joint frames at the all-zero configuration; each axis from the
        rotation the foot frame undergoes when only that joint moves (so URDF axis signs and
        mirrored legs are taken as they are). The returned report checks the model against the
        URDF forward kinematics on ``n_samples`` random configurations within the joint limits
        per leg - it should be at numerical precision. The kinematics' joint state is restored.
        """
        from chromapi.kinematics.kromatics import (
            _CHASSIS_FRAME,  # local: avoid import cycle
        )
        from chromapi.kinematics.topology import FOOT_FRAME_NAMES, JOINT_NAMES

        robot = kinematics.robot
        saved = {name: float(robot.get_joint(name)) for name in JOINT_NAMES}  # [rad]
        zero = dict.fromkeys(JOINT_NAMES, 0.0)  # [rad]
        delta = 1e-3  # [rad]

        def foot_rotation(q: JointDict, leg: str) -> npt.NDArray[np.float64]:
            kinematics.forward_kinematics(q)
            return np.asarray(robot.get_T_a_b(_CHASSIS_FRAME, FOOT_FRAME_NAMES[leg])[:3, :3], dtype=np.float64)

        try:
            chains: Dict[str, LegChain] = {}
            for leg in LEG_NAMES:
                foot = kinematics.forward_kinematics(zero)[leg]  # [m]
                names = [joint_name(leg, s) for s in JOINT_SUFFIXES]
                points = [
                    np.asarray(robot.get_T_a_b(_CHASSIS_FRAME, n)[:3, 3], dtype=np.float64) for n in names
                ]  # [m]
                r0 = foot_rotation(zero, leg)
                axes = []
                for name in names:
                    moved = dict(zero)
                    moved[name] = delta
                    rotvec = SciRotation.from_matrix(foot_rotation(moved, leg) @ r0.T).as_rotvec()  # [rad]
                    axes.append(np.asarray(rotvec / np.linalg.norm(rotvec), dtype=np.float64))
                limits = [robot.get_joint_limits(n) for n in names]  # [rad]
                chains[leg] = LegChain(
                    points=(points[0], points[1], points[2]),
                    axes=(axes[0], axes[1], axes[2]),
                    foot=foot,
                    lower=(float(limits[0][0]), float(limits[1][0]), float(limits[2][0])),
                    upper=(float(limits[0][1]), float(limits[1][1]), float(limits[2][1])),
                )
            ik = cls(chains)

            def truth(leg: str, q1: float, q2: float, q3: float) -> Vector3:
                q = dict(zero)
                q.update(zip((joint_name(leg, s) for s in JOINT_SUFFIXES), (q1, q2, q3)))
                return kinematics.forward_kinematics(q)[leg]

            report = ik.validate(truth, n_samples=n_samples, seed=seed)
        finally:
            kinematics.forward_kinematics(saved)

        if not report.ok():
            logger.warning(
                "AnalyticalIK.from_kromatics: the closed-form model disagrees with the URDF by "
                "more than 2 mm - see CalibrationReport:\n%s",
                report.summary(),
            )
        return ik, report

    # -- validation -------------------------------------------------------------------

    def validate(
        self,
        forward_kinematics: Callable[[str, float, float, float], Vector3],
        n_samples: int = 200,
        seed: int = 1,
    ) -> CalibrationReport:
        """Residual check against a ground-truth FK, on random configurations within the limits."""
        rng = np.random.default_rng(seed)
        rms_error: Dict[str, float] = {}
        max_error: Dict[str, float] = {}
        n_used: Dict[str, int] = {}
        for leg, chain in self.chains.items():
            errors = []  # [m]
            for _ in range(n_samples):
                q1, q2, q3 = rng.uniform(chain.lower, chain.upper)  # [rad]
                truth = forward_kinematics(leg, q1, q2, q3)  # [m]
                pred = self.forward_kinematics_leg(leg, q1, q2, q3)  # [m]
                errors.append(np.linalg.norm(pred - np.asarray(truth, dtype=float)))
            rms_error[leg] = float(np.sqrt(np.mean(np.square(errors))))
            max_error[leg] = float(np.max(errors))
            n_used[leg] = len(errors)
        return CalibrationReport(rms_error_m=rms_error, max_error_m=max_error, n_samples=n_used)
