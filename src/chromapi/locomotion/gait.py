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

WALK_POSTURE: Tuple[float, float, float] = (0.11, 0.09, 0.12)  # (x_reach, y_reach, body_height) [m]

# Trot: diagonal pairs (tl+br), (tr+bl).
_TROT_PHASE_OFFSETS: Dict[str, float] = {"tl": 0.0, "br": 0.0, "tr": 0.5, "bl": 0.5}  # [cycle]

# Crawl gait order: bl -> tl -> br -> tr.
_CRAWL_PHASE_OFFSETS: Dict[str, float] = {"bl": 0.75, "tl": 0.5, "br": 0.25, "tr": 0.0}  # [cycle]

# A small epsilon for angular velocity to avoid divide-by-zero in integration
_OMEGA_EPS = 1e-4  # [rad.s⁻¹] on a yaw rate, [rad] on a yaw step

# Gait pattern heading in the odometric frame
_PATTERN_HEADING = -np.pi / 2  # [rad]

# Linear speed (fraction of max_speed_mps) above which the travel direction picks the leg order
_ORDER_SPEED_FRACTION = 0.2

# Extra angle past the 45 deg quadrant boundary before the leg order switches
_ORDER_HYSTERESIS_RAD = np.radians(10.0)  # [rad]

# How far a stance foot may drift out of its nominal footprint before the body waits for it to swing back in
_STANCE_DRIFT_SLACK = 1.2

# Samples used to average the velocity ramp over an upcoming stance.
_PLAN_SAMPLES = 10

# Cap on how fast a re-planned touch-down point may move
_REPLAN_SPEED_FACTOR = 2.0


def _quintic(tau: float) -> float:
    """Normalized quintic polynomial time-scaling (``tau`` normalized time in [0, 1])."""
    t = float(np.clip(tau, 0.0, 1.0))
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def _swing_apex_profile(tau: float) -> float:
    """Ground-clearance profile."""
    t = float(np.clip(tau, 0.0, 1.0))
    return _quintic(2.0 * t) if t <= 0.5 else _quintic(2.0 - 2.0 * t)


def _rotate2(vec_xy: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    """Rotate a 2-vector by ``angle`` [rad] (counterclockwise, +Z-up)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec_xy[0] - s * vec_xy[1], s * vec_xy[0] + c * vec_xy[1]])


def _integrate_se2(
    xy: npt.NDArray[np.float64], yaw: float, vx: float, vy: float, wz: float, dt: float
) -> Tuple[npt.NDArray[np.float64], float]:
    """Integrate a planar odometric pose (``xy`` [m], ``yaw`` [rad]) by one body-frame twist step.

    ``vx``, ``vy`` [m.s⁻¹], ``wz`` [rad.s⁻¹], ``dt`` [s].
    """
    d_yaw = wz * dt  # [rad]
    if abs(d_yaw) < _OMEGA_EPS:
        d_body = np.array([vx, vy]) * dt  # [m]
    else:
        d_body = np.array(
            [
                (vx * np.sin(d_yaw) + vy * (np.cos(d_yaw) - 1.0)) / wz,
                (vx * (1.0 - np.cos(d_yaw)) + vy * np.sin(d_yaw)) / wz,
            ]
        )
    new_xy = xy + _rotate2(d_body, yaw)  # [m]
    new_yaw = yaw + d_yaw  # [rad]
    return new_xy, new_yaw


@dataclass
class GaitParams:
    """Gait Parameters.

    Attributes:
        pattern: Gait pattern name, ``"crawl"`` or ``"trot"``.
        step_frequency: Gait cycle frequency [Hz].
        duty_factor: Fraction of the cycle each leg spends in stance. Needs to be at least 0.75 for guaranteed 3-leg support at all times.
        swing_height: Apex ground clearance during swing [m].
        body_height: Nominal hip-to-foot drop [m].
        x_reach: Nominal footprint half-extent along chassis X [m].
        y_reach: Nominal footprint half-extent along chassis Y [m].
        phase_offsets: Phase offsets for each leg, in [0, 1[ [cycle].
        workspace_reach_m: Usable leg workspace (stride length), defined by hardware limitations [m].
        max_acceleration_mps2: Linear-velocity ramp limit [m.s⁻²].
        max_angular_acceleration_rad_s2: Angular-velocity ramp limit [rad.s⁻²].
        diagonal_speed_ratio: Diagonal speed limit, ]0.5, 1.0].
        solver: ``"qp"`` (default) or ``"analytical"``.
        ik_iterations: Number of iterations for the QP solver.

    """

    pattern: str = "crawl"
    step_frequency: float = 0.45  # [Hz]
    duty_factor: float = 0.75
    swing_height: float = 0.018  # [m]
    body_height: float = WALK_POSTURE[2]  # [m]
    x_reach: float = WALK_POSTURE[0]  # [m]
    y_reach: float = WALK_POSTURE[1]  # [m]
    phase_offsets: Dict[str, float] = field(default_factory=lambda: dict(_CRAWL_PHASE_OFFSETS))  # [cycle]
    workspace_reach_m: float = 0.116  # [m]
    max_acceleration_mps2: float = 0.15  # [m.s⁻²]
    max_angular_acceleration_rad_s2: float = 1.0  # [rad.s⁻²]
    diagonal_speed_ratio: float = 0.78
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
            step_frequency=0.8,
            duty_factor=0.6,
            swing_height=0.02,
            body_height=WALK_POSTURE[2],
            x_reach=WALK_POSTURE[0],
            y_reach=WALK_POSTURE[1],
            phase_offsets=dict(_TROT_PHASE_OFFSETS),
            workspace_reach_m=0.08,
        )

    @property
    def cycle_period_s(self) -> float:
        r"""Walk cycle period [s]."""
        return 1.0 / self.step_frequency

    @property
    def stance_duration_s(self) -> float:
        r"""Stance duration [s]."""
        return self.duty_factor * self.cycle_period_s

    @property
    def max_speed_mps(self) -> float:
        """Max sustainable body speed [m.s⁻¹]."""
        return self.workspace_reach_m / self.stance_duration_s

    @property
    def max_yaw_rate_rad_s(self) -> float:
        """Max sustainable yaw rate [rad.s⁻¹]."""
        radius = max(float(np.hypot(self.x_reach, self.y_reach)), 1e-6)  # [m]
        return self.workspace_reach_m / (self.stance_duration_s * radius)


@dataclass
class _LegSwingState:
    """Per-leg swing bookkeeping - odometric-frame (x, y, z) points."""

    lift_off: Vector3  # [m]
    touch_down: Vector3  # [m]


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
        self._nominal_footprint: FootTargets = self.kinematics.stance_targets(  # [m]
            self.params.body_height, self.params.x_reach, self.params.y_reach
        )
        # Twists are about the footprint center, not the chassis origin (offset from it).
        self._pivot = np.mean([p[:2] for p in self._nominal_footprint.values()], axis=0)  # [m]
        self._t = 0.0  # [s]
        self._odom_xy = np.zeros(2)  # [m]
        self._odom_yaw = 0.0  # [rad]
        self._target_twist = np.zeros(3)  # (vx, vy, wz) desired [m.s⁻¹, m.s⁻¹, rad.s⁻¹]
        self._twist = np.zeros(3)  # (vx, vy, wz) actual [m.s⁻¹, m.s⁻¹, rad.s⁻¹]
        self._swing: Dict[str, _LegSwingState] = {
            leg: _LegSwingState(lift_off=pos.copy(), touch_down=pos.copy())
            for leg, pos in self._nominal_footprint.items()
        }
        self._roles_by_turn = self._quarter_turn_roles()
        self._turn = 0 
        self._phase_shift = 0.0  # [cycle]
        self._prev_in_stance: Dict[str, bool] = {
            leg: self._phase(leg, 0.0) < self.params.duty_factor for leg in LEG_NAMES
        }

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Set the desired body-frame twist, clamped to :class:`GaitParams`'s speed limits.

        ``vx``, ``vy`` [m.s⁻¹] and ``wz`` [rad.s⁻¹], about the footprint center.
        """
        ratio = float(np.clip(self.params.diagonal_speed_ratio, 0.5, 1.0))
        p = 1.0 / (0.5 + np.log2(1.0 / ratio))
        speed = float((abs(vx) ** p + abs(vy) ** p) ** (1.0 / p))  # [m.s⁻¹]
        if speed > self.params.max_speed_mps and speed > 0.0:
            scale = self.params.max_speed_mps / speed
            vx *= scale
            vy *= scale
        max_wz = self.max_yaw_rate_rad_s  # [rad.s⁻¹]
        wz = float(np.clip(wz, -max_wz, max_wz))
        foot_speed = max(  # [m.s⁻¹]
            float(np.hypot(vx - wz * (p[1] - self._pivot[1]), vy + wz * (p[0] - self._pivot[0])))
            for p in self._nominal_footprint.values()
        )
        if foot_speed > self.params.max_speed_mps:
            scale = self.params.max_speed_mps / foot_speed
            vx, vy, wz = vx * scale, vy * scale, wz * scale
        self._target_twist = np.array([vx, vy, wz])

    @property
    def max_yaw_rate_rad_s(self) -> float:
        """Max sustainable yaw rate about the footprint center [rad.s⁻¹]."""
        radius = max(  # [m]
            float(np.linalg.norm(p[:2] - self._pivot)) for p in self._nominal_footprint.values()
        )
        own = self.params.workspace_reach_m / (self.params.stance_duration_s * max(radius, 1e-6))
        return min(own, self.params.max_yaw_rate_rad_s)

    def _origin_twist(self, twist: Optional[npt.NDArray[np.float64]] = None) -> Tuple[float, float, float]:
        """Twist [m.s⁻¹, m.s⁻¹, rad.s⁻¹], moved from the footprint center to the chassis origin."""
        vx, vy, wz = self._twist if twist is None else twist
        return vx + wz * self._pivot[1], vy - wz * self._pivot[0], wz

    # -- phase ----------------------------------------------------------------------------

    def _phase(self, leg: str, t: float, turn: Optional[int] = None, shift: Optional[float] = None) -> float:
        """Phase variable, in [0, 1[ [cycle], at time ``t`` [s]."""
        turn = self._turn if turn is None else turn
        shift = self._phase_shift if shift is None else shift  # [cycle]
        role = self._roles_by_turn[turn][leg]
        cycles = t / self.params.cycle_period_s + self.params.phase_offsets[role] + shift
        return float(cycles % 1.0)

    # -- direction-dependent leg order ----------------------------------------------------

    def _quarter_turn_roles(self) -> Dict[int, Dict[str, str]]:
        """For k quarter turns, map each leg to the leg whose phase offset it takes."""
        center = np.mean([p[:2] for p in self._nominal_footprint.values()], axis=0)  # [m]
        rel = {leg: p[:2] - center for leg, p in self._nominal_footprint.items()}  # [m]
        roles: Dict[int, Dict[str, str]] = {}
        for k in range(4):
            roles[k] = {}
            for leg in LEG_NAMES:
                source = _rotate2(rel[leg], -k * np.pi / 2)  # [m]
                roles[k][leg] = min(
                    LEG_NAMES, key=lambda other: float(np.linalg.norm(rel[other] - source))
                )
        return roles

    def _wanted_turn(self) -> int:
        """Quarter turn matching the actual travel direction."""
        vx, vy, _ = self._twist  # [m.s⁻¹]
        if np.hypot(vx, vy) < _ORDER_SPEED_FRACTION * self.params.max_speed_mps:
            return self._turn
        rel = float(np.arctan2(vy, vx)) - _PATTERN_HEADING  # [rad]
        current_heading = self._turn * np.pi / 2  # [rad]
        off = (rel - current_heading + np.pi) % (2 * np.pi) - np.pi  # [rad]
        if abs(off) <= np.pi / 4 + _ORDER_HYSTERESIS_RAD:
            return self._turn
        return int(np.round(rel / (np.pi / 2))) % 4

    def _update_leg_order(self) -> None:
        """Switch to the leg order of the travel direction, without a jump in any swing.

        The new order comes with a phase shift chosen so that a swinging leg keeps its phase
        and every stance leg stays in stance.
        """
        turn = self._wanted_turn()
        if turn == self._turn:
            return
        duty = self.params.duty_factor
        tol = 1e-9  # [cycle]
        current = {leg: self._phase(leg, self._t) for leg in LEG_NAMES}  # [cycle]
        swinging = [leg for leg in LEG_NAMES if current[leg] >= duty]
        for pivot in swinging or LEG_NAMES:
            shift = self._phase_shift + current[pivot] - self._phase(pivot, self._t, turn=turn)  # [cycle]
            new = {leg: self._phase(leg, self._t, turn=turn, shift=shift) for leg in LEG_NAMES}
            ok = all(
                abs((new[leg] - current[leg] + 0.5) % 1.0 - 0.5) < tol
                if leg in swinging
                else new[leg] < duty
                for leg in LEG_NAMES
            )
            if ok:
                self._turn, self._phase_shift = turn, shift % 1.0
                return

    # -- odometric-frame <-> chassis-frame ------------------------------------------------

    def _chassis_from_odom(
        self, p_odom: Vector3, pose: Optional[Tuple[npt.NDArray[np.float64], float]] = None
    ) -> Vector3:
        """Convert an odometric-frame point [m] to the chassis frame (current, or at ``pose``)."""
        odom_xy, odom_yaw = pose if pose is not None else (self._odom_xy, self._odom_yaw)  # [m], [rad]
        xy = _rotate2(p_odom[:2] - odom_xy, -odom_yaw)  # [m]
        return np.array([xy[0], xy[1], p_odom[2]])

    def _odom_from_chassis(
        self, p_chassis: Vector3, pose: Optional[Tuple[npt.NDArray[np.float64], float]] = None
    ) -> Vector3:
        """Convert a chassis-frame point [m] (current chassis, or at ``pose``) to the odometric frame."""
        odom_xy, odom_yaw = pose if pose is not None else (self._odom_xy, self._odom_yaw)  # [m], [rad]
        xy = odom_xy + _rotate2(p_chassis[:2], odom_yaw)  # [m]
        return np.array([xy[0], xy[1], p_chassis[2]])

    # -- foothold placement -------------- ------------------------------------------------

    def _touchdown_liftoff(
        self, leg: str, twist: Optional[npt.NDArray[np.float64]] = None
    ) -> Tuple[Vector3, Vector3]:
        """Symmetric lift-off/touch-down targets [m] around this leg's nominal footprint (in chassis frame)."""
        vx, vy, wz = self._origin_twist(twist)  # [m.s⁻¹], [m.s⁻¹], [rad.s⁻¹]
        nominal = self._nominal_footprint[leg]  # [m]
        half_stance = self.params.stance_duration_s / 2.0  # [s]

        if abs(wz) < _OMEGA_EPS:
            delta = np.array([vx, vy]) * half_stance  # [m]
            touch_down_xy = nominal[:2] + delta  # [m]
            lift_off_xy = nominal[:2] - delta  # [m]
        else:
            center = (1.0 / wz) * np.array([-vy, vx])  # [m]
            angle = wz * half_stance  # [rad]
            touch_down_xy = center + _rotate2(nominal[:2] - center, angle)
            lift_off_xy = center + _rotate2(nominal[:2] - center, -angle)

        touch_down = np.array([touch_down_xy[0], touch_down_xy[1], nominal[2]])
        lift_off = np.array([lift_off_xy[0], lift_off_xy[1], nominal[2]])
        return lift_off, touch_down

    def _planned_stance_twist(self, delay_s: float) -> npt.NDArray[np.float64]:
        """Mean twist [m.s⁻¹, m.s⁻¹, rad.s⁻¹] over a stance starting in ``delay_s`` [s], following the velocity ramp."""
        twist = self._twist.copy()  # [m.s⁻¹, m.s⁻¹, rad.s⁻¹]
        dt = delay_s / _PLAN_SAMPLES  # [s]
        for _ in range(_PLAN_SAMPLES):
            twist = self._ramped(twist, dt)
        dt = self.params.stance_duration_s / _PLAN_SAMPLES  # [s]
        total = np.zeros(3)  # [m.s⁻¹, m.s⁻¹, rad.s⁻¹]
        for _ in range(_PLAN_SAMPLES):
            nxt = self._ramped(twist, dt)
            total += 0.5 * (twist + nxt)
            twist = nxt
        return total / _PLAN_SAMPLES

    def _stance_speed_scale(self, dt: float) -> float:
        """Largest fraction of the current twist that keeps every stance foot in its workspace over ``dt`` [s]."""
        limit = _STANCE_DRIFT_SLACK * 0.5 * self.params.workspace_reach_m  # [m]
        stance = [leg for leg in LEG_NAMES if self._prev_in_stance[leg]]
        twist = np.array(self._origin_twist())  # [m.s⁻¹, m.s⁻¹, rad.s⁻¹]
        now = {  # [m]
            leg: float(np.linalg.norm(
                self._chassis_from_odom(self._swing[leg].touch_down)[:2]
                - self._nominal_footprint[leg][:2]
            ))
            for leg in stance
        }

        def ok(scale: float) -> bool:
            """Check if the stance feet stay within the workspace when moving at ``scale * twist`` for ``dt`` seconds."""
            vx, vy, wz = (float(c) for c in scale * twist)  # [m.s⁻¹], [m.s⁻¹], [rad.s⁻¹]
            pose = _integrate_se2(self._odom_xy, self._odom_yaw, vx, vy, wz, dt)  # [m], [rad]
            for leg in stance:
                p = self._chassis_from_odom(self._swing[leg].touch_down, pose)  # [m]
                drift = float(np.linalg.norm(p[:2] - self._nominal_footprint[leg][:2]))  # [m]
                if drift > limit and drift > now[leg]:
                    return False
            return True

        if ok(1.0):
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

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
        """Advance the gait by ``dt`` [s] and return the resulting 12 joint angles [rad]."""
        self._twist = self._ramped(self._twist, dt)
        scale = self._stance_speed_scale(dt)
        vx, vy, wz = (scale * c for c in self._origin_twist())  # [m.s⁻¹], [m.s⁻¹], [rad.s⁻¹]
        self._odom_xy, self._odom_yaw = _integrate_se2(
            self._odom_xy, self._odom_yaw, vx, vy, wz, dt
        )
        self._t += dt
        self._update_leg_order()

        foot_targets: FootTargets = {}  # [m]
        for leg in LEG_NAMES:
            in_stance = self._phase(leg, self._t) < self.params.duty_factor
            if in_stance and not self._prev_in_stance[leg]:
                anchor = self._swing[leg].touch_down
                self._swing[leg] = _LegSwingState(lift_off=anchor, touch_down=anchor)
            elif not in_stance and self._prev_in_stance[leg]:
                anchor = self._swing[leg].touch_down
                self._swing[leg] = _LegSwingState(lift_off=anchor, touch_down=anchor.copy())
            self._prev_in_stance[leg] = in_stance

            if in_stance:
                foot_targets[leg] = self._chassis_from_odom(self._swing[leg].touch_down)
            else:
                remaining_swing_s = (  # [s]
                    1.0 - self._phase(leg, self._t)
                ) * self.params.cycle_period_s
                _, touch_down_chassis = self._touchdown_liftoff(
                    leg, self._planned_stance_twist(remaining_swing_s)
                )
                pose_at_touch_down = _integrate_se2(  # [m], [rad]
                    self._odom_xy, self._odom_yaw, vx, vy, wz, remaining_swing_s
                )
                planned = self._odom_from_chassis(touch_down_chassis, pose_at_touch_down)  # [m]
                swing_s = (1.0 - self.params.duty_factor) * self.params.cycle_period_s  # [s]
                max_move = _REPLAN_SPEED_FACTOR * self.params.workspace_reach_m / swing_s * dt  # [m]
                move = planned[:2] - self._swing[leg].touch_down[:2]  # [m]
                norm = float(np.linalg.norm(move))  # [m]
                if norm > max_move:
                    move *= max_move / norm
                self._swing[leg].touch_down[:2] += move
                tau = (self._phase(leg, self._t) - self.params.duty_factor) / (
                    1.0 - self.params.duty_factor
                )
                swing = self._swing[leg]
                sigma = _quintic(tau)
                xy = swing.lift_off[:2] + sigma * (swing.touch_down[:2] - swing.lift_off[:2])  # [m]
                ground_z = swing.lift_off[2]  # [m], == touch_down[2]: flat-ground assumption
                z = ground_z + self.params.swing_height * _swing_apex_profile(tau)  # [m]
                foot_targets[leg] = self._chassis_from_odom(np.array([xy[0], xy[1], z]))

        return self._solve_ik(foot_targets)

    def _ramped(self, twist: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        """Move actual twist [m.s⁻¹, m.s⁻¹, rad.s⁻¹] to desired within the acceleration limits, over ``dt`` [s]."""
        twist = twist.copy()
        linear_error = self._target_twist[:2] - twist[:2]  # [m.s⁻¹]
        linear_step = self.params.max_acceleration_mps2 * dt  # [m.s⁻¹]
        error_norm = float(np.linalg.norm(linear_error))  # [m.s⁻¹]
        if error_norm <= linear_step or error_norm == 0.0:
            twist[:2] = self._target_twist[:2]
        else:
            twist[:2] += linear_error / error_norm * linear_step

        angular_error = self._target_twist[2] - twist[2]  # [rad.s⁻¹]
        angular_step = self.params.max_angular_acceleration_rad_s2 * dt  # [rad.s⁻¹]
        if abs(angular_error) <= angular_step:
            twist[2] = self._target_twist[2]
        else:
            twist[2] += np.sign(angular_error) * angular_step
        return twist


class WalkMove(Move):
    """Walk Move class."""

    def __init__(self, kinematics: Kromatics, params: Optional[GaitParams] = None) -> None:
        """Build a fresh :class:`GaitEngine` (stand still pose) for ``kinematics``/``params``."""
        self.gait = GaitEngine(kinematics, params)

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Forward to :meth:`GaitEngine.set_velocity` (``vx``, ``vy`` [m.s⁻¹], ``wz`` [rad.s⁻¹])."""
        self.gait.set_velocity(vx, vy, wz)

    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """Advance the gait and write the resulting joint targets into ``command``."""
        del state  # unused - GaitEngine is pure dead reckoning
        command.target_angles.update(self.gait.step(dt))
