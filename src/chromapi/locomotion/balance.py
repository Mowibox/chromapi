# SPDX-License-Identifier: Apache-2.0
"""Balance and chassis posing module for Chromapi - stability margin, attitude PD and a free-base QP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import placo
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as SciRotation

from chromapi.chromapi import DEFAULT_URDF_PATH, MotorCommand, Move, RobotState
from chromapi.kinematics.kromatics import Kromatics
from chromapi.kinematics.poses import APPROACH_POSE, REST_POSE, WAKE_UP_POSE
from chromapi.kinematics.topology import FOOT_FRAME_NAMES, JOINT_NAMES, LEG_NAMES

logger = logging.getLogger(__name__)

_CHASSIS_FRAME = "chassis_assembly"


def stability_margin(feet_xy: Sequence[Sequence[float]], com_xy: Sequence[float]) -> float:
    """Signed distance [m] from the CoM to the support polygon edge (> 0 inside).

    With fewer than 3 feet the polygon degenerates: 2 feet give minus the distance to the
    segment between them, 1 foot minus the distance to that foot.
    """
    pts = np.asarray(feet_xy, dtype=float)
    c = np.asarray(com_xy, dtype=float)
    if len(pts) == 0:
        return float("-inf")
    if len(pts) < 3:
        a, b = pts[0], pts[-1]
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip((c - a) @ ab / (ab @ ab), 0.0, 1.0))
        return -float(np.linalg.norm(c - (a + t * ab)))
    poly = pts[ConvexHull(pts).vertices]  # counter-clockwise in 2D
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)  # inward for CCW
    return float(np.min(np.sum(normals * (c - poly), axis=1) / np.linalg.norm(edges, axis=1)))


def _measured_q(state: RobotState) -> Dict[str, float]:
    return {name: state.joint_positions.get(name, WAKE_UP_POSE[name]) for name in JOINT_NAMES}


def _stance_legs(state: RobotState) -> Sequence[str]:
    # ponytail: no contact sensing on hardware yet -> all feet assumed planted.
    if not state.foot_contacts:
        return LEG_NAMES
    return [leg for leg in LEG_NAMES if state.foot_contacts.get(leg)]


@dataclass
class BalanceGains:
    """Attitude PD and QP weights.

    The PD output is the angular-velocity target of the orientation task.
    """

    kp: float = 4.0  # [s⁻¹]
    kd: float = 0.1  # [-]
    max_tilt: float = 0.3  # [rad] bound on the commanded correction
    w_feet: float = 1e3
    w_attitude: float = 10.0
    w_com: float = 10.0
    w_posture: float = 1e-3
    iterations: int = 8
    max_joint_speed: float = 1.5  # [rad.s⁻¹]


class AttitudeCorrector:
    """Attitude PD: measured chassis tilt (IMU) -> chassis tilt to command in the feet frame."""

    def __init__(self, kinematics: Kromatics, gains: Optional[BalanceGains] = None) -> None:
        """Build the robot model for the IMU frame and chassis frame."""
        self.gains = gains or BalanceGains()
        self._R_chassis_imu = kinematics.robot.get_T_a_b(_CHASSIS_FRAME, "imu")[:3, :3].copy()
        self.roll_pitch = (0.0, 0.0)  # [rad] measured, chassis frame
        self.gyro = np.zeros(3)  # [rad.s⁻¹] chassis frame
        self.rpy_cmd = np.zeros(2)  # [rad] commanded chassis (roll, pitch) in the feet frame
        self.correction = np.zeros(2)  # [rad] the PD's integrated part of rpy_cmd (a slope, a push)

    def measure(self, state: RobotState) -> None:
        """Chassis roll/pitch and gyro from the IMU."""
        w, x, y, z = state.imu_quat_wxyz
        R_world_chassis = SciRotation.from_quat([x, y, z, w]).as_matrix() @ self._R_chassis_imu.T
        _, pitch, roll = SciRotation.from_matrix(R_world_chassis).as_euler("ZYX")
        self.roll_pitch = (float(roll), float(pitch))
        self.gyro = self._R_chassis_imu @ np.asarray(state.imu_gyro)

    def correct(self, dt: float, target: npt.ArrayLike = (0.0, 0.0)) -> npt.NDArray[np.float64]:
        """``rpy_cmd`` for the measured tilt ``target`` (roll, pitch) [rad], after ``dt`` [s].

        The target goes straight through, and the PD integrates only
        what is left: the measured error to the target, i.e. a slope or a push.
        """
        g = self.gains
        target = np.asarray(target, dtype=float)
        w_star = -g.kp * (np.asarray(self.roll_pitch) - target) - g.kd * self.gyro[:2]
        self.correction = np.clip(self.correction + w_star * dt, -g.max_tilt, g.max_tilt)
        self.rpy_cmd = target + self.correction
        return self.rpy_cmd


class BalanceMove(Move):
    """Stand still, feet planted where they were at start; the chassis level, or posed by :meth:`set_body_pose`."""

    def __init__(
        self,
        urdf_path: Union[str, Path] = DEFAULT_URDF_PATH,
        dt: float = 0.02,
        gains: Optional[BalanceGains] = None,
        enabled: bool = True,
    ) -> None:
        """Build two private robot models: one for the QP, one for estimation.

        Args:
            urdf_path: Robot URDF.
            dt: Control period [s] - the QP integration step.
            gains: PD gains and task weights.
            enabled: If False, only estimate (margin, tilt) and hold the start pose: the
                baseline of the push test.

        """
        self.gains = gains or BalanceGains()
        self.enabled = enabled
        # Own models (10 ms each): no shared placo state with the gait or go_to_pose.
        self._est = Kromatics(urdf_path, dt=dt)
        self._kin = Kromatics(urdf_path, dt=dt)
        robot = self._kin.robot
        for name in JOINT_NAMES:
            lower, upper = robot.get_joint_limits(name)
            poses = [pose[name] for pose in (WAKE_UP_POSE, REST_POSE, APPROACH_POSE)]
            robot.set_joint_limits(name, min(lower, *poses), max(upper, *poses))

        solver = placo.KinematicsSolver(robot)  # Free the base!!
        solver.enable_joint_limits(True)
        solver.dt = dt
        self._solver = solver
        self._feet = {}
        for leg in LEG_NAMES:
            task = solver.add_position_task(FOOT_FRAME_NAMES[leg], np.zeros(3))
            task.configure(f"foot_{leg}", "soft", self.gains.w_feet)
            self._feet[leg] = task
        self._attitude = solver.add_orientation_task(_CHASSIS_FRAME, np.eye(3))
        self._attitude.configure("attitude", "soft", self.gains.w_attitude)
        self._com = solver.add_com_task(np.zeros(3))
        self._com.configure("com", "soft", self.gains.w_com)
        posture = solver.add_joints_task()
        posture.set_joints(dict(WAKE_UP_POSE))
        posture.configure("posture", "soft", self.gains.w_posture)
        self._posture = posture

        self.attitude = AttitudeCorrector(self._est, self.gains)

        self._started = False
        self.margin = 0.0  # [m]
        self.com0 = np.zeros(3)  # [m] CoM at start, feet frame
        self._body_xyz = np.zeros(3)  # [m] CoM offset from com0, feet frame
        self._body_rp = np.zeros(2)  # [rad] chassis (roll, pitch) target, measured from level
        self._body_yaw = 0.0  # [rad] chassis yaw over the feet, open loop (the IMU yaw drifts)
        self._lifted: Tuple[Optional[str], Dict[str, float]] = (None, {})
        self._masked: Optional[str] = None  # leg whose joints the QP currently leaves alone

    @property
    def roll_pitch(self) -> Tuple[float, float]:
        """Measured chassis (roll, pitch) [rad]."""
        return self.attitude.roll_pitch

    @property
    def rpy_cmd(self) -> npt.NDArray[np.float64]:
        """Commanded chassis (roll, pitch) in the feet frame [rad]."""
        return self.attitude.rpy_cmd

    @property
    def feet(self) -> Dict[str, npt.NDArray[np.float64]]:
        """Foot positions held, feet frame (= chassis frame at start) [m]."""
        return {leg: task.target_world.copy() for leg, task in self._feet.items()}

    def set_body_pose(self, xyz: npt.ArrayLike = (0.0, 0.0, 0.0), rpy: npt.ArrayLike = (0.0, 0.0)) -> None:
        """Pose the chassis over the planted feet; zero is the start pose, level.

        Args:
            xyz: CoM offset from its start position, in the feet frame (+X left, +Y backward, +Z up) [m].
            rpy: Chassis (roll, pitch) [rad], measured by the IMU: roll about chassis X tilts nose-down for
                roll > 0, pitch about Y leans sideways. The attitude PD reaches it on a slope too. An optional
                third value turns the chassis over the feet (yaw, +Z, to the left), without feedback.

        """
        rpy = np.asarray(rpy, dtype=float)
        self._body_xyz = np.array(xyz, dtype=float)
        self._body_rp = rpy[:2].copy()
        self._body_yaw = float(rpy[2]) if len(rpy) > 2 else 0.0

    def lift(self, leg: Optional[str], q_leg: Optional[Dict[str, float]] = None) -> None:
        """Let ``leg`` off the ground, its 3 joints at ``q_leg`` [rad] (written as is); None: all feet down again."""
        self._lifted = (leg, dict(q_leg or {}))

    def on_start(self) -> None:
        """Re-anchor the feet on the next tick, start pose, all feet down."""
        self._started = False
        self.set_body_pose()
        self.lift(None)

    def _anchor(self, q: Dict[str, float]) -> None:
        """QP world frame = chassis frame at start; feet held where they are."""
        robot = self._kin.robot
        for name, value in q.items():
            robot.set_joint(name, value)
        robot.set_T_world_fbase(np.eye(4))
        robot.update_kinematics()
        for leg in LEG_NAMES:
            self._feet[leg].target_world = robot.get_T_world_frame(FOOT_FRAME_NAMES[leg])[:3, 3].copy()
        self.com0 = robot.com_world().copy()
        # Hold the start posture, not WAKE_UP_POSE: after a walk the robot does not stand exactly there.
        self._posture.set_joints(dict(q))
        self.attitude.rpy_cmd[:] = 0.0
        self.attitude.correction[:] = 0.0
        self._started = True

    def _estimate(self, state: RobotState) -> float:
        """Stability margin [m] from the measured joints and IMU tilt."""
        robot = self._est.robot
        for name, value in _measured_q(state).items():
            robot.set_joint(name, value)
        robot.set_T_world_fbase(np.eye(4))
        robot.update_kinematics()
        roll, pitch = self.roll_pitch
        R = SciRotation.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()  # chassis -> level
        feet = [R @ robot.get_T_world_frame(FOOT_FRAME_NAMES[leg])[:3, 3] for leg in _stance_legs(state)]
        com = R @ robot.com_world()
        return stability_margin([f[:2] for f in feet], com[:2])

    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """Estimate, run the attitude PD, solve the free-base QP."""
        self.attitude.measure(state)
        self.margin = self._estimate(state)
        if not self._started:
            # The targets the servos hold, not the measured joints: those sag under the load.
            start = {**_measured_q(state), **{j: command.target_angles[j] for j in JOINT_NAMES if j in command.target_angles}}
            self._anchor(start)
            command.target_angles.update(start)
        if not self.enabled:
            return

        g = self.gains
        rpy_cmd = self.attitude.correct(dt, self._body_rp)
        self._attitude.R_world_frame = SciRotation.from_euler(
            "ZYX", [self._body_yaw, rpy_cmd[1], rpy_cmd[0]]
        ).as_matrix()
        # ponytail: CoM target in the feet frame, = level frame only on flat ground.
        centre = np.mean([task.target_world for task in self._feet.values()], axis=0)
        self._com.target_world = np.array([centre[0], centre[1], self.com0[2]]) + self._body_xyz

        robot = self._kin.robot
        lifted, q_lifted = self._lifted
        if lifted != self._masked:
            for leg in (self._masked, lifted):
                if leg is not None:
                    down = leg != lifted
                    self._feet[leg].configure(f"foot_{leg}", "soft", g.w_feet if down else 0.0)
                    for joint in (f"{leg}_1", f"{leg}_2", f"{leg}_3"):
                        (self._solver.unmask_dof if down else self._solver.mask_dof)(joint)
            self._masked = lifted
        for joint, value in q_lifted.items():
            robot.set_joint(joint, value)
        for _ in range(g.iterations):
            try:
                self._solver.solve(True)
            except RuntimeError as exc:
                logger.warning("BalanceMove: QP solve stopped early (%s)", exc)
                break
            robot.update_kinematics()
        # Rate-limit against the targets actually sent, and resume the QP from them next tick.
        step = g.max_joint_speed * dt  # [rad]
        for name in JOINT_NAMES:
            if name in q_lifted:
                command.target_angles[name] = q_lifted[name]  # the gesture is already smooth, and faster
                continue
            sent = command.target_angles.get(name, float(robot.get_joint(name)))
            target = sent + float(np.clip(robot.get_joint(name) - sent, -step, step))
            command.target_angles[name] = target
            robot.set_joint(name, target)
        robot.update_kinematics()
