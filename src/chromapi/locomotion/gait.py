"""Gait engine for Chromapi."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from chromapi.chromapi import MotorCommand, Move, RobotState
from chromapi.kinematics.kromatics import Kromatics
from chromapi.kinematics.topology import LEG_NAMES
from chromapi.kinematics.types import FootTargets, JointDict, Vector3

if TYPE_CHECKING:
    from chromapi.kinematics.analytical_ik import AnalyticalIK, CalibrationReport

logger = logging.getLogger(__name__)

__all__ = [
    "WALK_POSTURE",
    "GaitEngine",
    "GaitParams",
    "WalkMove",
]

WALK_POSTURE: Tuple[float, float, float] = (0.10, 0.075, 0.09)

# Trot: diagonal pairs (tl+br), (tr+bl).
_TROT_PHASE_OFFSETS: Dict[str, float] = {"tl": 0.0, "br": 0.0, "tr": 0.5, "bl": 0.5}

# Crawl gait order: bl -> tl -> br -> tr.
_CRAWL_PHASE_OFFSETS: Dict[str, float] = {"bl": 0.75, "tl": 0.5, "br": 0.25, "tr": 0.0}

# A small epsilon for angular velocity to avoid divide-by-zero in integration.
_OMEGA_EPS = 1e-4


def _quintic(tau: float) -> float:
    r"""Normalized quintic polynomial time-scaling."""
    t = float(np.clip(tau, 0.0, 1.0))
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def _swing_apex_profile(tau: float) -> float:
    r"""Ground-clearance profile."""
    t = float(np.clip(tau, 0.0, 1.0))
    return _quintic(2.0 * t) if t <= 0.5 else _quintic(2.0 - 2.0 * t)


def _rotate2(vec_xy: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    """Rotate a 2-vector by ``angle`` radians (counterclockwise, +Z-up)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec_xy[0] - s * vec_xy[1], s * vec_xy[0] + c * vec_xy[1]])


def _integrate_se2(
    xy: npt.NDArray[np.float64], yaw: float, vx: float, vy: float, wz: float, dt: float
) -> Tuple[npt.NDArray[np.float64], float]:
    """Integrate a planar odometric pose by one body-frame twist step."""
    d_yaw = wz * dt
    if abs(d_yaw) < _OMEGA_EPS:
        d_body = np.array([vx, vy]) * dt
    else:
        d_body = np.array(
            [
                (vx * np.sin(d_yaw) + vy * (np.cos(d_yaw) - 1.0)) / wz,
                (vx * (1.0 - np.cos(d_yaw)) + vy * np.sin(d_yaw)) / wz,
            ]
        )
    new_xy = xy + _rotate2(d_body, yaw)
    new_yaw = yaw + d_yaw
    return new_xy, new_yaw


@dataclass
class GaitParams:
    """Gait Parameters.

    Attributes:
        pattern: Gait pattern name, ``"crawl"`` or ``"trot"``.
        step_frequency: Gait cycle frequency, Hz.
        duty_factor: Fraction of the cycle each leg spends in stance. Needs to be at least 0.75 for guaranteed 3-leg support at all times.
        swing_height: Apex ground clearance during swing, in meters.
        body_height: Nominal hip-to-foot drop, in meters.
        x_reach: Nominal footprint half-extent along chassis X, in meters.
        y_reach: Nominal footprint half-extent along chassis Y, in meters.
        phase_offsets: Phase offsets for each leg, in [0, 1[.
        workspace_reach_m: Usable leg workspace, defined by hardware limitations.
        max_acceleration_mps2: Linear-velocity ramp limit.
        max_angular_acceleration_rad_s2: Angular-velocity ramp limit.
        solver: ``"qp"`` (default) or ``"analytical"``.
        ik_iterations: Number of iterations for the QP solver.

    """

    pattern: str = "crawl"
    step_frequency: float = 0.6
    duty_factor: float = 0.8
    swing_height: float = 0.05
    body_height: float = WALK_POSTURE[2]
    x_reach: float = WALK_POSTURE[0]
    y_reach: float = WALK_POSTURE[1]
    phase_offsets: Dict[str, float] = field(default_factory=lambda: dict(_CRAWL_PHASE_OFFSETS))
    workspace_reach_m: float = 0.05
    max_acceleration_mps2: float = 0.15
    max_angular_acceleration_rad_s2: float = 1.0
    solver: str = "qp"
    ik_iterations: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate ``phase_offsets``, ``duty_factor``, and ``solver``."""
        missing = set(LEG_NAMES) - set(self.phase_offsets)
        if missing:
            raise ValueError(f"GaitParams.phase_offsets is missing legs: {sorted(missing)}")
        if not 0.0 < self.duty_factor < 1.0:
            raise ValueError(f"GaitParams.duty_factor must be in (0, 1), got {self.duty_factor}")
        if self.solver not in ("qp", "analytical"):
            raise ValueError(f"GaitParams.solver must be 'qp' or 'analytical', got {self.solver!r}")

    @classmethod
    def crawl(cls) -> "GaitParams":
        """Crawl tuning for stable walking. Move one leg at a time."""
        return cls()

    @classmethod
    def trot(cls) -> "GaitParams":
        """Trot tuning for faster walking. Move diagonal leg pairs at a time."""
        return cls(
            pattern="trot",
            step_frequency=1.5,
            duty_factor=0.5,
            swing_height=0.02,
            body_height=WALK_POSTURE[2],
            x_reach=WALK_POSTURE[0],
            y_reach=WALK_POSTURE[1],
            phase_offsets=dict(_TROT_PHASE_OFFSETS),
            workspace_reach_m=0.05,
        )

    @property
    def cycle_period_s(self) -> float:
        r"""Walk cycle period, in seconds."""
        return 1.0 / self.step_frequency

    @property
    def stance_duration_s(self) -> float:
        r"""Stance duration, in seconds."""
        return self.duty_factor * self.cycle_period_s

    @property
    def max_speed_mps(self) -> float:
        """Max sustainable body speed, in meters per second."""
        return self.workspace_reach_m / self.stance_duration_s

    @property
    def max_yaw_rate_rad_s(self) -> float:
        """Max sustainable yaw rate, in radians per second."""
        radius = max(float(np.hypot(self.x_reach, self.y_reach)), 1e-6)
        return self.workspace_reach_m / (self.stance_duration_s * radius)


@dataclass
class _LegSwingState:
    """Per-leg swing bookkeeping - odometric-frame (x, y, z) points."""

    lift_off: Vector3
    touch_down: Vector3


class GaitEngine:
    """Gait algorithm implementation."""

    def __init__(self, kinematics: Kromatics, params: Optional[GaitParams] = None) -> None:
        """Build the engine and reset it to a stand still pose."""
        self.kinematics = kinematics
        self.params = params or GaitParams()
        self._analytical: Optional["AnalyticalIK"] = None 
        self._analytical_report: Optional["CalibrationReport"] = None
        self._analytical_disabled = False
        self.reset()

    # -- setup / state ------------------------------------------------------------------

    def reset(self) -> None:
        """Return to a stand still pose."""
        self._nominal_footprint: FootTargets = self.kinematics.stance_targets(
            self.params.body_height, self.params.x_reach, self.params.y_reach
        )
        self._t = 0.0
        self._odom_xy = np.zeros(2)
        self._odom_yaw = 0.0
        self._target_twist = np.zeros(3)  # (vx, vy, wz), desired
        self._twist = np.zeros(3)  # (vx, vy, wz), actual
        self._swing: Dict[str, _LegSwingState] = {
            leg: _LegSwingState(lift_off=pos.copy(), touch_down=pos.copy())
            for leg, pos in self._nominal_footprint.items()
        }
        self._prev_in_stance: Dict[str, bool] = {
            leg: self._phase(leg, 0.0) < self.params.duty_factor for leg in LEG_NAMES
        }

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Set the desired body-frame twist, clamped to :class:`GaitParams`'s speed limits."""
        speed = np.hypot(vx, vy)
        if speed > self.params.max_speed_mps and speed > 0.0:
            scale = self.params.max_speed_mps / speed
            vx *= scale
            vy *= scale
        wz = float(np.clip(wz, -self.params.max_yaw_rate_rad_s, self.params.max_yaw_rate_rad_s))
        self._target_twist = np.array([vx, vy, wz])

    # -- phase ----------------------------------------------------------------------------

    def _phase(self, leg: str, t: float) -> float:
        """Phase variable, in [0, 1[."""
        cycles = t / self.params.cycle_period_s + self.params.phase_offsets[leg]
        return float(cycles % 1.0)

    # -- odometric-frame <-> chassis-frame ------------------------------------------------

    def _chassis_from_odom(self, p_odom: Vector3) -> Vector3:
        """Convert an odometric-frame point to the current chassis frame."""
        xy = _rotate2(p_odom[:2] - self._odom_xy, -self._odom_yaw)
        return np.array([xy[0], xy[1], p_odom[2]])

    def _odom_from_chassis(self, p_chassis: Vector3) -> Vector3:
        """Convert a current-chassis-frame point to the odometric frame."""
        xy = self._odom_xy + _rotate2(p_chassis[:2], self._odom_yaw)
        return np.array([xy[0], xy[1], p_chassis[2]])

    # -- foothold placement (.md §3.6, §4.4) ------------------------------------------------

    def _touchdown_liftoff(self, leg: str) -> Tuple[Vector3, Vector3]:
        """Symmetric lift-off targets around this leg's nominal footprint (in chassis frame)."""
        vx, vy, wz = self._twist
        nominal = self._nominal_footprint[leg]
        half_stance = self.params.stance_duration_s / 2.0

        if abs(wz) < _OMEGA_EPS:
            delta = np.array([vx, vy]) * half_stance
            touch_down_xy = nominal[:2] + delta
            lift_off_xy = nominal[:2] - delta
        else:
            center = (1.0 / wz) * np.array([-vy, vx])
            angle = wz * half_stance
            touch_down_xy = center + _rotate2(nominal[:2] - center, angle)
            lift_off_xy = center + _rotate2(nominal[:2] - center, -angle)

        touch_down = np.array([touch_down_xy[0], touch_down_xy[1], nominal[2]])
        lift_off = np.array([lift_off_xy[0], lift_off_xy[1], nominal[2]])
        return lift_off, touch_down

    # -- solving --------------------------------------------------------------------------

    def _solve_ik(self, foot_targets: FootTargets) -> JointDict:
        """Dispatch to the configured solver."""
        if self.params.solver == "analytical" and not self._analytical_disabled:
            if self._analytical is None:
                from chromapi.kinematics.analytical_ik import AnalyticalIK

                self._analytical, self._analytical_report = AnalyticalIK.from_kromatics(
                    self.kinematics
                )
                if not self._analytical_report.ok():
                    logger.warning(
                        "GaitEngine: analytical-IK calibration residual exceeds tolerance - "
                        "disabling it for this engine and using the QP instead.\n%s",
                        self._analytical_report.summary(),
                    )
                    self._analytical = None
                    self._analytical_disabled = True

        if self._analytical is not None:
            q, converged = self._analytical.inverse_kinematics(foot_targets)
            if converged:
                return q
            missing = {leg: p for leg, p in foot_targets.items() if leg not in q}
            logger.debug("GaitEngine: analytical solver missed %s - falling back to the QP.", sorted(missing))
            qp_q, _ = self.kinematics.inverse_kinematics(
                foot_targets, iterations=self.params.ik_iterations
            )
            qp_q.update(q)  # trust the analytical solution wherever it did converge
            return qp_q

        q, _ = self.kinematics.inverse_kinematics(
            foot_targets, iterations=self.params.ik_iterations
        )
        return q

    # -- main tick --------------------------------------------------------------------------

    def step(self, dt: float) -> JointDict:
        """Advance the gait by ``dt`` seconds and return the resulting 12 joint angles."""
        self._ramp_velocity(dt)
        vx, vy, wz = self._twist
        self._odom_xy, self._odom_yaw = _integrate_se2(
            self._odom_xy, self._odom_yaw, vx, vy, wz, dt
        )
        self._t += dt

        foot_targets: FootTargets = {}
        for leg in LEG_NAMES:
            in_stance = self._phase(leg, self._t) < self.params.duty_factor
            if in_stance and not self._prev_in_stance[leg]:
                anchor = self._swing[leg].touch_down
                self._swing[leg] = _LegSwingState(lift_off=anchor, touch_down=anchor)
            elif not in_stance and self._prev_in_stance[leg]:
                lift_off_odom = self._swing[leg].touch_down
                _, touch_down_chassis = self._touchdown_liftoff(leg)
                self._swing[leg] = _LegSwingState(
                    lift_off=lift_off_odom,
                    touch_down=self._odom_from_chassis(touch_down_chassis),
                )
            self._prev_in_stance[leg] = in_stance

            if in_stance:
                foot_targets[leg] = self._chassis_from_odom(self._swing[leg].touch_down)
            else:
                tau = (self._phase(leg, self._t) - self.params.duty_factor) / (
                    1.0 - self.params.duty_factor
                )
                swing = self._swing[leg]
                sigma = _quintic(tau)
                xy = swing.lift_off[:2] + sigma * (swing.touch_down[:2] - swing.lift_off[:2])
                ground_z = swing.lift_off[2]  # == touch_down[2]: flat-ground assumption
                z = ground_z + self.params.swing_height * _swing_apex_profile(tau)
                foot_targets[leg] = self._chassis_from_odom(np.array([xy[0], xy[1], z]))

        return self._solve_ik(foot_targets)

    def _ramp_velocity(self, dt: float) -> None:
        """Move actual twist to desired within the configured accel limits."""
        linear_error = self._target_twist[:2] - self._twist[:2]
        linear_step = self.params.max_acceleration_mps2 * dt
        error_norm = float(np.linalg.norm(linear_error))
        if error_norm <= linear_step or error_norm == 0.0:
            self._twist[:2] = self._target_twist[:2]
        else:
            self._twist[:2] += linear_error / error_norm * linear_step

        angular_error = self._target_twist[2] - self._twist[2]
        angular_step = self.params.max_angular_acceleration_rad_s2 * dt
        if abs(angular_error) <= angular_step:
            self._twist[2] = self._target_twist[2]
        else:
            self._twist[2] += np.sign(angular_error) * angular_step


class WalkMove(Move):
    """Walk Move class."""

    def __init__(self, kinematics: Kromatics, params: Optional[GaitParams] = None) -> None:
        """Build a fresh :class:`GaitEngine` (stand still pose) for ``kinematics``/``params``."""
        self.gait = GaitEngine(kinematics, params)

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Forward to :meth:`GaitEngine.set_velocity`."""
        self.gait.set_velocity(vx, vy, wz)

    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """Advance the gait and write the resulting joint targets into ``command``."""
        del state  # unused - GaitEngine is pure dead reckoning
        command.target_angles.update(self.gait.step(dt))
