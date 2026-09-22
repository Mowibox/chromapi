"""Kromatics - Kinematics Module for Chromapi."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as SciRotation

from chromapi.kinematics.poses import APPROACH_POSE, REST_POSE, WAKE_UP_POSE, ZERO_POSE
from chromapi.kinematics.topology import (
    FOOT_FRAME_NAMES,
    JOINT_NAMES,
    JOINT_SUFFIXES,
    LEG_NAMES,
    joint_name,
)
from chromapi.kinematics.types import FootTargets, JointDict, QVector, Vector3

if TYPE_CHECKING:
    from chromapi.kinematics.analytical_ik import AnalyticalIK, CalibrationReport

try:
    import placo

    PLACO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    placo = None
    PLACO_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    "APPROACH_POSE",
    "FOOT_FRAME_NAMES",
    "FootTargets",
    "JOINT_NAMES",
    "JOINT_SUFFIXES",
    "JointDict",
    "Kromatics",
    "LEG_NAMES",
    "PLACO_AVAILABLE",
    "QVector",
    "REST_POSE",
    "Vector3",
    "WAKE_UP_POSE",
    "ZERO_POSE",
    "joint_name",
]


_CHASSIS_FRAME = "chassis_assembly"

#: QP solve iterations per call - cheap enough (~0.15 ms at 5 iterations) that there's no
#: real-time-budget reason to run fewer for a per-tick gait call.
_DEFAULT_ITERATIONS = 8

#: A foot is considered "reached" within this distance (meters) of its target.
_CONVERGENCE_TOL_M = 1e-3

_ZERO3: Vector3 = np.zeros(3)


class Kromatics:
    """Whole-body kinematics for Chromapi - one QP over all 12 joints."""

    _STAND_REACH = (0.195, 0.0, 0.12)  # (x_reach, y_reach, height), meters
    _REST_REACH = (0.225, 0.0, 0.05)

    def __init__(
        self,
        urdf_path: Union[str, Path],
        dt: float = 0.01,
        initial_pose: Optional[JointDict] = None,
    ) -> None:
        """Load the robot URDF and set up the per-foot QP tasks.

        Args:
            urdf_path: Path to ``model/urdf/robot.urdf`` (or its containing directory).
            dt: Solver integration time step, in seconds - should match the control-loop period.
            initial_pose: Joint angles to seed ``self.robot`` with before the foot tasks are set
                up, so their initial targets reflect this configuration rather than the URDF's
                zero pose (legs splayed flat).

        """
        if not PLACO_AVAILABLE:
            raise RuntimeError(
                "Kromatics requires the 'placo' package (pip install placo), which was not found."
            )

        self.robot = placo.RobotWrapper(str(urdf_path), placo.Flags.ignore_collisions)
        if initial_pose is not None:
            self._apply_q(dict(initial_pose))

        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.dt = dt

        self.foot_tasks: Dict[str, Any] = {}
        for leg in LEG_NAMES:
            frame = FOOT_FRAME_NAMES[leg]
            target = self.robot.get_T_world_frame(frame)[:3, 3]
            task = self.solver.add_position_task(frame, target)
            task.configure(frame, "soft", 1.0)
            self.foot_tasks[leg] = task

        self._stand_pose_cache: Dict[Tuple[float, float, float], JointDict] = {}
        self.last_solve_error_m: float = 0.0

    # -- conversions ---------------------------------------------------------------

    @staticmethod
    def to_array(q: Union[JointDict, Sequence[float]]) -> npt.NDArray[np.float64]:
        """Convert a joint dict (or an already-ordered sequence) to a length-12 array."""
        if isinstance(q, dict):
            return np.array([q[name] for name in JOINT_NAMES], dtype=float)
        arr = np.asarray(q, dtype=float)
        if arr.shape != (12,):
            raise ValueError(f"Expected 12 joint values, got shape {arr.shape}")
        return arr

    @staticmethod
    def to_dict(q: Union[JointDict, Sequence[float]]) -> JointDict:
        """Convert a length-12 array (or an already-named dict) to a joint dict."""
        if isinstance(q, dict):
            return dict(q)
        arr = np.asarray(q, dtype=float)
        return dict(zip(JOINT_NAMES, arr.tolist()))

    # -- low-level robot-state helpers -----------------------------------------------

    def _apply_q(self, q: JointDict) -> None:
        """Push a joint dict into ``self.robot`` and refresh its kinematics."""
        for name, value in q.items():
            self.robot.set_joint(name, value)
        self.robot.update_kinematics()

    def joint_limits(self, joint: str) -> Tuple[float, float]:
        """Return the (lower, upper) URDF joint-limit bound for ``joint`` in radians."""
        lower, upper = self.robot.get_joint_limits(joint)
        return float(lower), float(upper)

    def _hip_origin_chassis(self, leg: str) -> Vector3:
        """Hip-yaw joint origin for ``leg``, in the chassis frame."""
        origin: Vector3 = self.robot.get_T_a_b(_CHASSIS_FRAME, joint_name(leg, 1))[:3, 3]
        return origin

    # -- forward / inverse kinematics ------------------------------------------------

    def forward_kinematics(self, q: Union[JointDict, Sequence[float]]) -> FootTargets:
        """Compute the 4 foot positions (chassis frame) for a given 12-joint configuration."""
        self._apply_q(self.to_dict(q))
        return {
            leg: self.robot.get_T_a_b(_CHASSIS_FRAME, FOOT_FRAME_NAMES[leg])[:3, 3].copy()
            for leg in LEG_NAMES
        }

    def _run_qp(
        self,
        body_xyz: Vector3,
        body_rpy: Vector3,
        foot_targets: FootTargets,
        seed: Optional[JointDict],
        n_iterations: int,
    ) -> int:
        """Impose the base pose/foot tasks and iterate; return how many iterations succeeded."""
        if seed is not None:
            self._apply_q(seed)

        trunk_pose = np.eye(4)
        trunk_pose[:3, :3] = SciRotation.from_euler("xyz", body_rpy).as_matrix()
        trunk_pose[:3, 3] = body_xyz
        self.robot.set_T_world_fbase(trunk_pose)
        self.robot.update_kinematics()

        for leg, target in foot_targets.items():
            self.foot_tasks[leg].target_world = np.asarray(target, dtype=float)

        completed = 0
        for _ in range(n_iterations):
            try:
                self.solver.solve(True)
            except RuntimeError as exc:
                logger.warning("Kromatics: QP solve stopped early (%s)", exc)
                break
            self.robot.update_kinematics()
            completed += 1
        return completed

    def _solve(
        self,
        body_xyz: Vector3,
        body_rpy: Vector3,
        foot_targets: FootTargets,
        q_init: Optional[Union[JointDict, Sequence[float]]],
        iterations: Optional[int],
    ) -> Tuple[JointDict, bool]:
        """Shared QP solve: impose the base pose, set foot tasks, iterate and measure residual."""
        n_iterations = _DEFAULT_ITERATIONS if iterations is None else iterations
        seed = self.to_dict(q_init) if q_init is not None else None

        completed = self._run_qp(body_xyz, body_rpy, foot_targets, seed, n_iterations)
        if completed == 0:
            logger.warning(
                "Kromatics: seed rejected by the QP on its first iteration - retrying from "
                "WAKE_UP_POSE"
            )
            self._run_qp(body_xyz, body_rpy, foot_targets, dict(WAKE_UP_POSE), n_iterations)

        q = {name: float(self.robot.get_joint(name)) for name in JOINT_NAMES}
        max_error = 0.0
        for leg, target in foot_targets.items():
            achieved = self.robot.get_T_world_frame(FOOT_FRAME_NAMES[leg])[:3, 3]
            max_error = max(max_error, float(np.linalg.norm(achieved - np.asarray(target))))
        self.last_solve_error_m = max_error
        return q, max_error < _CONVERGENCE_TOL_M

    def inverse_kinematics(
        self,
        foot_targets: FootTargets,
        q_init: Optional[Union[JointDict, Sequence[float]]] = None,
        use_warm_start: bool = True,
        iterations: Optional[int] = None,
    ) -> Tuple[JointDict, bool]:
        """Compute the 12 joint angles placing each foot at its target (in chassis frame).

        Args:
            foot_targets: Mapping leg -> desired (x, y, z) in the chassis frame. Legs absent
                from the mapping keep their current target.
            q_init: Optional joint configuration - re-syncs the solver's internal state to it before solving.
            use_warm_start: If True (default) and ``q_init`` is None, continue from the
                solver's current configuration. If False, reset to
                :data:`~chromapi.kinematics.poses.WAKE_UP_POSE` first.
            iterations: QP iterations for this call - defaults to :data:`_DEFAULT_ITERATIONS`.

        Returns:
            (q, converged): the 12 joint angles.

        """
        if q_init is None and not use_warm_start:
            self._apply_q(dict(WAKE_UP_POSE))
        return self._solve(_ZERO3, _ZERO3, foot_targets, q_init, iterations)

    # -- stance generation -------------------------------------------------------------

    def stance_targets(
        self,
        height: float,
        x_reach: float = _STAND_REACH[0],
        y_reach: float = _STAND_REACH[1],
    ) -> FootTargets:
        """Build a symmetric 4-foot footprint under the hips, in the chassis frame.

        Each foot is placed ``height`` below its own hip, offset by ``x_reach``/``y_reach``
        away from the chassis centerline (sign follows each hip, so this stays symmetric
        regardless of which leg is physically front/back/left/right).

        Args:
            height: Vertical drop from hip to foot, in meters (positive = down).
            x_reach: Horizontal offset along chassis X, in meters (magnitude only).
            y_reach: Horizontal offset along chassis Y, in meters (magnitude only).

        """
        targets: FootTargets = {}
        for leg in LEG_NAMES:
            hip = self._hip_origin_chassis(leg)
            sign_x = 1.0 if hip[0] >= 0 else -1.0
            sign_y = 1.0 if hip[1] >= 0 else -1.0
            targets[leg] = hip + np.array(
                [sign_x * x_reach, sign_y * y_reach, -abs(height)]
            )
        return targets

    def default_stand_pose(
        self,
        height: float = _STAND_REACH[2],
        x_reach: float = _STAND_REACH[0],
        y_reach: float = _STAND_REACH[1],
    ) -> JointDict:
        """Joint angles for a nominal standing pose."""
        key = (height, x_reach, y_reach)
        if key not in self._stand_pose_cache:
            targets = self.stance_targets(height, x_reach, y_reach)
            pose, converged = self.inverse_kinematics(
                targets, use_warm_start=False, iterations=4 * _DEFAULT_ITERATIONS
            )
            if not converged:
                raise ValueError(
                    f"Stand pose (height={height}, x_reach={x_reach}, "
                    f"y_reach={y_reach}) is not reachable by all 4 legs."
                )
            self._stand_pose_cache[key] = pose
        return dict(self._stand_pose_cache[key])

    def default_rest_pose(
        self,
        height: float = _REST_REACH[2],
        x_reach: float = _REST_REACH[0],
        y_reach: float = _REST_REACH[1],
    ) -> JointDict:
        """Joint angles for a rest pose."""
        return self.default_stand_pose(height=height, x_reach=x_reach, y_reach=y_reach)

    # -- analytical (closed-form) IK ----------------------------------------------------

    def build_analytical_ik(
        self, n_samples: int = 300, seed: int = 0
    ) -> Tuple["AnalyticalIK", "CalibrationReport"]:
        """Fallback alternative analytical IK to this instance's QP."""
        from chromapi.kinematics.analytical_ik import AnalyticalIK

        return AnalyticalIK.from_kromatics(self, n_samples=n_samples, seed=seed)

    # -- quasi-static body posing -------------------------------------------------------

    def body_ik(
        self,
        body_xyz: Vector3,
        body_rpy: Vector3,
        foot_targets_world: Optional[FootTargets] = None,
        reference_height: float = _STAND_REACH[2],
        q_init: Optional[Union[JointDict, Sequence[float]]] = None,
        iterations: Optional[int] = None,
    ) -> Tuple[JointDict, bool]:
        """Solve joint angles for a desired chassis pose with the feet planted on the ground.

        Args:
            body_xyz: Desired chassis origin position, in the world/ground frame.
            body_rpy: Desired chassis orientation (roll, pitch, yaw), in the world frame.
            foot_targets_world: Foot positions to hold fixed, in the world frame.
            reference_height: Stance height for the default footprint when ``foot_targets_world`` is not given.
            q_init: Optional joint configuration (dict or length-12 array).
            iterations: QP iterations for this call - defaults to :data:`_DEFAULT_ITERATIONS`.

        Returns:
            (q, converged).

        """
        if foot_targets_world is None:
            nominal = self.stance_targets(reference_height)
            foot_targets_world = {leg: pos.copy() for leg, pos in nominal.items()}

        return self._solve(
            np.asarray(body_xyz, dtype=float),
            np.asarray(body_rpy, dtype=float),
            foot_targets_world,
            q_init,
            iterations,
        )
