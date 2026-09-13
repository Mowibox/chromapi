"""Chromapi class for controlling a Chromapi robot.

High-level SDK entry point: one :class:`Chromapi` object drives either the real robot (over
the STM32 UART bridge, see :mod:`chromapi.hardware.motherboard_bridge`) or a MuJoCo
simulation of it (``model/mjcf/scene.xml``).

Standing is not the zero configuration
---------------------------------------
Note that for Chromapi q = 0 on every joint means the legs are splayed out flat, roughly level (see
Onshape assembly). It is a valid, structurally stable configuration (the motors are strong), but it is 
not what you want to power on into for locomotion. Because of this:

- :meth:`Chromapi.connect` never commands ``ZERO_POSE`` on its own; it only reads back
  whatever configuration the robot/sim is already in.
- :meth:`Chromapi.wake_up` is the only way to reach :meth:`Chromapi.stand_pose` - always
  through a smooth interpolation, never a jump, regardless of the starting configuration.
- :meth:`Chromapi.rest` returns to a compact, *load-bearing* crouch
  (:data:`chromapi.kinematics.kromatics.REST_POSE`), not to
  :data:`~chromapi.kinematics.kromatics.ZERO_POSE`.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
import yaml

from chromapi.hardware.motherboard_bridge import BridgeClient
from chromapi.kinematics import kromatics as K

if TYPE_CHECKING:
    from chromapi.locomotion.gait import GaitParams, WalkMove

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _PACKAGE_DIR / "config" / "config.yaml"
DEFAULT_URDF_PATH = _PACKAGE_DIR / "model" / "urdf" / "robot.urdf"
DEFAULT_MJCF_PATH = _PACKAGE_DIR / "model" / "mjcf" / "scene.xml"
DEFAULT_CHROMAPI_SCREAM_PATH = _PACKAGE_DIR / "assets" / "audio" / "chromapi_scream.wav"

#: STS3215 magnetic-encoder resolution (steps per revolution) and center step, standard
#: Feetech convention (see stm32-sts3215-lib's ``sts3215_regs.h``/memory table).
STS3215_STEPS_PER_REV = 4096
STS3215_CENTER_STEP = 2048
_STEPS_PER_RAD = STS3215_STEPS_PER_REV / (2.0 * np.pi)

# ========================================================================================
# Robot state
# ========================================================================================
@dataclass
class RobotState:
    """A timestamped snapshot of the robot's proprioceptive state."""

    joint_positions: Dict[str, float] = field(default_factory=dict) # in rad
    joint_velocities: Dict[str, float] = field(default_factory=dict) # in rad/s
    joint_loads: Dict[str, float] = field(default_factory=dict)
    imu_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    imu_gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0) # in rad/s, IMU frame
    imu_accel: Tuple[float, float, float] = (0.0, 0.0, 0.0) # in m/s^2, IMU frame
    foot_contacts: Dict[str, bool] = field(default_factory=dict)
    voltage_v: float = 0.0
    current_a: float = 0.0
    power_w: float = 0.0
    timestamp: float = 0.0

# ========================================================================================
# Backend abstraction - real hardware vs MuJoCo, behind one interface
# ========================================================================================
class RobotBackend(ABC):
    """What Chromapi needs from a hardware or simulated backend.

    Interface implemented by both a real-servo controller 
    and a MuJoCo/MeshCat one - keeping :class:`Chromapi` itself 
    entirely backend-agnostic.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Open the connection (serial port, or instantiate the sim). Returns success."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection/simulation resources."""

    @abstractmethod
    def send_joint_targets(self, q: Dict[str, float]) -> bool:
        """Send target joint angles (radians), keyed by joint name. Returns success."""

    @abstractmethod
    def read_state(self) -> Optional[RobotState]:
        """Read back the current proprioceptive state, or None on failure/timeout."""

    def set_torque_enabled(
        self, enabled: bool, joints: Optional[Sequence[str]] = None
    ) -> None:
        """Enable/disable actuator torque on the given joints (default: all). No-op if unsupported."""

    def set_led_all(self, r: int, g: int, b: int) -> None:
        """Set every ring LED to one color. No-op if unsupported."""

    def set_led(self, index: int, r: int, g: int, b: int) -> None:
        """Set a single ring LED (0-17). No-op if unsupported."""

    def set_led_bulk(self, colors: Sequence[Tuple[int, int, int]]) -> None:
        """Set all 18 ring LEDs at once from a list of (r, g, b) tuples. No-op if unsupported."""

    def is_alive(self) -> bool:
        """Whether the backend is still usable - False once it can no longer be driven.

        Checked once per :meth:`Chromapi._control_loop` tick; returning False ends the
        control loop (see :class:`MuJoCoBackend`'s override, for the interactive viewer
        window being closed). Always True by default (real hardware has no equivalent
        "closed" state short of :meth:`close`).
        """
        return True


class HardwareBackend(RobotBackend):
    """Real Chromapi hardware, over the STM32 UART bridge (:class:`BridgeClient`)."""

    def __init__(
        self,
        servo_config: Dict[str, Dict[str, Any]],
        port: str = "/dev/ttyAMA0",
        baudrate: int = 1_000_000,
    ) -> None:
        """Build the backend from the ``servos:`` section of ``config.yaml``.

        Args:
            servo_config: The ``config["servos"]`` mapping (``id_map``, ``sign``,
                ``zero_offset_steps`` - see ``config/config.yaml``).
            port: Serial device for the RPi4 <-> STM32 UART bridge.
            baudrate: Must match the bridge firmware (1 Mbps).

        """
        self._bridge = BridgeClient(port=port, baudrate=baudrate)
        self._id_map: Dict[str, int] = dict(servo_config["id_map"])
        self._sign: Dict[str, int] = dict(servo_config["sign"])
        self._zero_offset: Dict[str, int] = dict(servo_config["zero_offset_steps"])
        self._id_to_joint: Dict[int, str] = {v: k for k, v in self._id_map.items()}
        missing = set(K.JOINT_NAMES) - set(self._id_map)
        if missing:
            raise ValueError(f"servo config is missing joints: {sorted(missing)}")

    def connect(self) -> bool:
        """Open the UART serial port to the STM32 bridge."""
        return self._bridge.connect()

    def close(self) -> None:
        """Close the UART serial port."""
        self._bridge.close()

    def _rad_to_steps(self, joint: str, angle_rad: float) -> int:
        raw = (
            STS3215_CENTER_STEP
            + self._sign[joint] * angle_rad * _STEPS_PER_RAD
            + self._zero_offset[joint]
        )
        return int(np.clip(round(raw), 0, STS3215_STEPS_PER_REV - 1))

    def _steps_to_rad(self, joint: str, raw_steps: int) -> float:
        raw = raw_steps - STS3215_CENTER_STEP - self._zero_offset[joint]
        return self._sign[joint] * raw / _STEPS_PER_RAD

    def send_joint_targets(self, q: Dict[str, float]) -> bool:
        """Pack joint angles into the 12-servo raw-step frame and send SET_POSITIONS."""
        raw_steps = [STS3215_CENTER_STEP] * 12
        for joint, angle in q.items():
            servo_id = self._id_map.get(joint)
            if servo_id is None:
                continue
            raw_steps[servo_id - 1] = self._rad_to_steps(joint, angle)
        return self._bridge.set_positions(raw_steps)

    def read_state(self) -> Optional[RobotState]:
        """Read the STATE_FEEDBACK snapshot and convert it to a :class:`RobotState`."""
        snapshot = self._bridge.get_state()
        if snapshot is None:
            return None

        state = RobotState(timestamp=time.monotonic())
        for servo in snapshot["servos"]:
            joint = self._id_to_joint.get(servo["id"])
            if joint is None:
                continue
            state.joint_positions[joint] = self._steps_to_rad(joint, servo["pos"])
            # STS3215 "Present Speed" shares the position register's step resolution
            # (Feetech convention: raw units are steps/second).
            state.joint_velocities[joint] = (
                self._sign[joint] * servo["speed"] / _STEPS_PER_RAD
            )
            state.joint_loads[joint] = float(servo["load"])

        quat = snapshot["imu"]["quat"]
        state.imu_quat_wxyz = (quat[0], quat[1], quat[2], quat[3])
        gyro = snapshot["imu"]["gyro_rps"]
        state.imu_gyro = (gyro[0], gyro[1], gyro[2])
        accel = snapshot["imu"]["acc_mps2"]
        state.imu_accel = (accel[0], accel[1], accel[2])
        state.foot_contacts = {
            "tl": snapshot["switches"]["TL"],
            "tr": snapshot["switches"]["TR"],
            "bl": snapshot["switches"]["BL"],
            "br": snapshot["switches"]["BR"],
        }
        state.voltage_v = snapshot["power"]["voltage_V"]
        state.current_a = snapshot["power"]["current_A"]
        state.power_w = snapshot["power"]["power_W"]
        return state

    def set_torque_enabled(
        self, enabled: bool, joints: Optional[Sequence[str]] = None
    ) -> None:
        """Write the STS3215 ``TORQUE_SWITCH`` register (0x28) on the given joints."""
        target_joints = joints if joints is not None else K.JOINT_NAMES
        for joint in target_joints:
            servo_id = self._id_map.get(joint)
            if servo_id is None:
                continue
            self._bridge.write_servo_register(servo_id, 0x28, 1, 1 if enabled else 0)

    def set_led_all(self, r: int, g: int, b: int) -> None:
        """Set every ring LED via a single bridge frame."""
        self._bridge.set_led_color_all(r, g, b)

    def set_led(self, index: int, r: int, g: int, b: int) -> None:
        """Set a single ring LED via the bridge."""
        self._bridge.set_led_color(index, r, g, b)

    def set_led_bulk(self, colors: Sequence[Tuple[int, int, int]]) -> None:
        """Set all 18 ring LEDs via a single bridge frame."""
        self._bridge.set_led_color_bulk(list(colors))


class MuJoCoBackend(RobotBackend):
    """MuJoCo simulation backend, driving ``model/mjcf/scene.xml``."""

    def __init__(
        self,
        mjcf_path: Union[str, Path] = DEFAULT_MJCF_PATH,
        realtime: bool = True,
        launch_viewer: bool = False,
        key_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Load the MJCF model.

        Args:
            mjcf_path: Path to ``model/mjcf/scene.xml`` 
            launch_viewer: If True, opens an interactive ``mujoco.viewer`` window.
            key_callback: Receives a raw GLFW keycode on every keypress in
            the viewer window for teleop use.

        """
        import mujoco  # local import: mujoco is an optional ("rl") dependency

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self._data = mujoco.MjData(self._model)
        self._realtime = realtime
        self._last_step_time: Optional[float] = None
        self._viewer = None
        self._launch_viewer = launch_viewer
        self._key_callback = key_callback

        self._chassis_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "chassis_assembly"
        )
        self._foot_body_ids = {
            leg: mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot"
            )
            for leg in K.LEG_NAMES
        }
        self._floor_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        self._led_ring_material_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_MATERIAL, "ws2812b_cob_pixel_ring_27mm_material"
        )
        self._led_ring_default_rgb: Optional[npt.NDArray[np.float64]] = (
            self._model.mat_rgba[self._led_ring_material_id][:3].copy()
            if self._led_ring_material_id >= 0
            else None
        )

        # A foot is "in contact" if any geom belonging to its tibia body touches the floor.
        self._foot_geom_ids: Dict[str, List[int]] = {leg: [] for leg in K.LEG_NAMES}
        tibia_body_of_leg = {
            "tl": "left_tibia_assembly",
            "bl": "left_tibia_assembly_2",
            "br": "right_tibia_assembly",
            "tr": "right_tibia_assembly_2",
        }
        for leg, body_name in tibia_body_of_leg.items():
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self._foot_geom_ids[leg] = [
                g for g in range(self._model.ngeom) if self._model.geom_bodyid[g] == body_id
            ]

        mujoco.mj_forward(self._model, self._data)

    def connect(self) -> bool:
        """No-op (the model is already loaded); optionally opens the interactive viewer."""
        if self._launch_viewer and self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self._model, self._data, key_callback=self._key_callback
            )
        return True

    def close(self) -> None:
        """Close the interactive viewer, if one was opened."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                logger.debug("MuJoCo viewer close() raised (already closed?) - ignoring", exc_info=True)
            self._viewer = None

    def is_alive(self) -> bool:
        """Check if the interactive viewer window is still open."""
        return self._viewer is None or self._viewer.is_running()

    @property
    def viewer(self) -> Any:
        """The ``mujoco.viewer`` handle opened by :meth:`connect`, or None if not launched."""
        return self._viewer

    _MAX_CATCHUP_S = 0.1

    def send_joint_targets(self, q: Dict[str, float]) -> bool:
        """Write joint targets to the position actuators and step the simulation forward."""
        for joint, angle in q.items():
            try:
                self._data.actuator(joint).ctrl[0] = angle
            except KeyError:
                continue

        now = time.monotonic()
        if self._realtime and self._last_step_time is not None:
            elapsed_sim_needed = min(
                self._MAX_CATCHUP_S, max(0.0, now - self._last_step_time)
            )
        else:
            elapsed_sim_needed = self._model.opt.timestep
        self._last_step_time = now

        n_substeps = max(1, round(elapsed_sim_needed / self._model.opt.timestep))
        for _ in range(n_substeps):
            self._mujoco.mj_step(self._model, self._data)

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
        return True

    def read_state(self) -> Optional[RobotState]:
        """Read joint/IMU/contact state directly from MjData."""
        state = RobotState(timestamp=time.monotonic())
        for joint in K.JOINT_NAMES:
            try:
                state.joint_positions[joint] = float(self._data.joint(joint).qpos[0])
                state.joint_velocities[joint] = float(self._data.joint(joint).qvel[0])
            except KeyError:
                continue

        try:
            quat = self._data.sensor("orientation").data
            state.imu_quat_wxyz = (
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
            )
            gyro = self._data.sensor("imu_ang_vel").data
            state.imu_gyro = (float(gyro[0]), float(gyro[1]), float(gyro[2]))
            accel = self._data.sensor("imu_accel").data
            state.imu_accel = (float(accel[0]), float(accel[1]), float(accel[2]))
        except KeyError:
            logger.warning("MJCF model has no 'imu' sensors - IMU state left at defaults")

        for leg, geom_ids in self._foot_geom_ids.items():
            state.foot_contacts[leg] = self._is_contact(geom_ids)

        return state

    def _is_contact(self, geom_ids: List[int]) -> bool:
        geom_id_set = set(geom_ids)
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if self._floor_geom_id in pair and pair & geom_id_set:
                return True
        return False

    def _set_led_ring_rgb(self, rgb: npt.NDArray[np.float64]) -> None:
        """Apply an (r, g, b) triplet (0-255 scale) to the ring material"""
        if self._led_ring_material_id < 0:
            return
        peak = float(np.max(rgb))
        if peak <= 0.0:
            normalized = (
                self._led_ring_default_rgb
                if self._led_ring_default_rgb is not None
                else np.zeros(3)
            )
        else:
            normalized = rgb / peak
        self._model.mat_rgba[self._led_ring_material_id][:3] = normalized

    def set_led_all(self, r: int, g: int, b: int) -> None:
        """Recolor the LED ring material according to the given RGB triplet (0-255 scale) - see :meth:`_set_led_ring_rgb`."""
        self._set_led_ring_rgb(np.array([r, g, b], dtype=np.float64))

    def set_led_bulk(self, colors: Sequence[Tuple[int, int, int]]) -> None:
        """Recolor the LED ring material to the *average* of ``colors`` - see :meth:`_set_led_ring_rgb`."""
        if not colors:
            return
        mean_rgb = np.mean(np.asarray(colors, dtype=np.float64), axis=0)
        self._set_led_ring_rgb(mean_rgb)

    def get_chassis_pose(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (xyz, quat_wxyz) of the chassis free joint - simulation-only ground truth."""
        return self._data.qpos[0:3].copy(), self._data.qpos[3:7].copy()

    def set_chassis_pose(
        self, xyz: Sequence[float], quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
    ) -> None:
        """Set the chassis pose - simulation-only, for tests & resets."""
        self._data.qpos[0:3] = xyz
        self._data.qpos[3:7] = quat_wxyz
        self._data.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, self._data)


# ========================================================================================
# Move / behavior abstraction - the RL residual & scripted-motion extension point
# ========================================================================================

@dataclass
class MotorCommand:
    """The joint-target output of a :class:`Move`, built on top of a kinematic reference."""

    target_angles: Dict[str, float]

class Move(ABC):
    """A stateful behavior driving joint targets on every control-loop tick.
    """

    @abstractmethod
    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """Update ``command.target_angles`` in place for this tick.

        Args:
            state: Latest proprioceptive snapshot.
            command: Target joints to update - starts each tick pre-filled with the previous
                tick's targets.
            dt: Time since the previous tick, in seconds.

        """

    def on_start(self) -> None:
        """Run once when the move becomes active (:meth:`Chromapi.play_move`)."""

    def on_stop(self) -> None:
        """Run once when the move is deactivated (:meth:`Chromapi.stop_move`)."""

# ========================================================================================
# Chromapi
# ========================================================================================


def _load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path) as f:
        return dict(yaml.safe_load(f))


def _minimum_jerk(t01: float) -> float:
    """Minimum-jerk time-scaling s(t) in [0, 1] for t01 in [0, 1] (zero vel/accel at ends)."""
    t01 = float(np.clip(t01, 0.0, 1.0))
    return 10.0 * t01**3 - 15.0 * t01**4 + 6.0 * t01**5


_YAW_PRESTAGE_THRESHOLD_RAD = 0.2
_YAW_PRESTAGE_SETTLE_S = 0.4


class Chromapi:
    """High-level SDK for the Chromapi 12-DOF quadruped, real or simulated.

    ::

        with Chromapi(backend="mujoco") as robot:
            robot.wake_up()
            robot.go_to_pose(body_rpy=(0.0, 0.1, 0.0))  # look down
            state = robot.get_state()
            robot.rest()

    A background thread runs the control loop at ``config["control"]["loop_hz"]``: on every
    tick it lets the active :class:`Move` (if any) update the joint targets, sends them to
    the backend, and refreshes the latest :class:`RobotState`.
    """

    def __init__(
        self,
        backend: str = "hardware",
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        mjcf_path: Union[str, Path] = DEFAULT_MJCF_PATH,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 1_000_000,
        launch_viewer: bool = False,
        key_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Build a Chromapi instance without connecting yet (see :meth:`connect`).

        Args:
            backend: ``"hardware"`` (real robot over UART) or ``"mujoco"`` (simulation).
            config_path: Path to ``config.yaml`` (servo ID map, control-loop frequency, ...).
            mjcf_path: MJCF scene to load for the ``"mujoco"`` backend.
            port: Serial port for the ``"hardware"`` backend.
            baudrate: Baud rate for the ``"hardware"`` backend (must match the firmware).
            launch_viewer: For the ``"mujoco"`` backend, open an interactive viewer window.
            key_callback: For the ``"mujoco"`` backend with ``launch_viewer=True``, a
                keyboard callback forwarded to the viewer.

        """
        self.config = _load_config(config_path)
        control_cfg = self.config.get("control", {})
        self.loop_hz: float = float(control_cfg.get("loop_hz", 50.0))
        self.wake_up_duration_s: float = float(control_cfg.get("wake_up_duration_s", 3.5))
        self.rest_duration_s: float = float(control_cfg.get("rest_duration_s", 2.0))
        self.approach_duration_s: float = float(control_cfg.get("approach_duration_s", 2.0))
        self.kinematics = K.Kromatics(
            DEFAULT_URDF_PATH, dt=1.0 / self.loop_hz, initial_pose=dict(K.ZERO_POSE)
        )

        if backend == "hardware":
            self.backend: RobotBackend = HardwareBackend(
                self.config["servos"], port=port, baudrate=baudrate
            )
        elif backend == "mujoco":
            self.backend = MuJoCoBackend(
                mjcf_path, launch_viewer=launch_viewer, key_callback=key_callback
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}, expected 'hardware' or 'mujoco'")

        self._connected = False
        self._state_lock = threading.Lock()
        self._latest_state: Optional[RobotState] = None
        self._targets_lock = threading.Lock()
        self._current_targets: Dict[str, float] = dict(K.ZERO_POSE)
        self._active_move: Optional[Move] = None
        self._registered_moves: Dict[str, Move] = {}
        self._walk_move: Optional["WalkMove"] = None
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """Open the backend connection and start the background control loop."""
        if self._connected:
            return True
        if not self.backend.connect():
            return False

        state = self.backend.read_state()
        with self._targets_lock:
            if state is not None and state.joint_positions:
                self._current_targets.update(state.joint_positions)
            else:
                logger.warning(
                    "Could not read initial joint state - holding the built-in ZERO_POSE "
                    "(legs splayed) until the first command; call wake_up() promptly."
                )

        self._connected = True
        self._stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._control_loop, name="chromapi-control-loop", daemon=True
        )
        self._loop_thread.start()
        return True

    def disconnect(self, disable_torque: bool = False) -> None:
        """Stop the control loop and close the backend connection.

        Args:
            disable_torque: If True, disable motor torque after the control loop thread has actually stopped.

        """
        if not self._connected:
            return
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
            if self._loop_thread.is_alive():
                logger.warning(
                    "Control loop did not stop within 2s - leaving the backend connection "
                    "open rather than risk closing it out from under the still-running loop thread."
                )
                return
        if disable_torque:
            self.backend.set_torque_enabled(False)
        self.backend.close()
        self._connected = False

    def __enter__(self) -> "Chromapi":
        """Connect on entering a ``with`` block."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Disconnect on exiting a ``with`` block."""
        self.disconnect()

    def __del__(self) -> None:
        """Best-effort cleanup if :meth:`disconnect` was not called explicitly."""
        try:
            self.disconnect()
        except Exception:
            pass

    def _control_loop(self) -> None:
        period = 1.0 / self.loop_hz
        last_tick = time.monotonic()
        while not self._stop_event.is_set():
            if not self.backend.is_alive():
                logger.info("Backend no longer alive (viewer closed?) - stopping control loop")
                return
            tick_start = time.monotonic()
            dt = tick_start - last_tick
            last_tick = tick_start

            state = self.backend.read_state()
            if state is not None:
                with self._state_lock:
                    self._latest_state = state

            with self._targets_lock:
                if self._active_move is not None and state is not None:
                    command = MotorCommand(target_angles=dict(self._current_targets))
                    try:
                        self._active_move.step(state, command, dt)
                    except Exception:
                        logger.exception("Move.step() raised - holding last targets")
                    else:
                        self._current_targets = command.target_angles
                targets = dict(self._current_targets)

            self.backend.send_joint_targets(targets)

            elapsed = time.monotonic() - tick_start
            time.sleep(max(period - elapsed, 0.0))
            if elapsed > 2 * period:
                logger.warning(
                    "Control loop overrun: tick took %.1f ms (target period %.1f ms)",
                    elapsed * 1000,
                    period * 1000,
                )

    # -- state --------------------------------------------------------------------------

    def get_state(self) -> Optional[RobotState]:
        """Return the latest :class:`RobotState` snapshot (updated at ``loop_hz``)."""
        with self._state_lock:
            return self._latest_state

    def get_current_joint_positions(self) -> Dict[str, float]:
        """Return the last commanded joint targets (not the measured state)."""
        with self._targets_lock:
            return dict(self._current_targets)

    def get_imu_orientation(self) -> Tuple[float, float, float, float]:
        """Return the trunk orientation quaternion (w, x, y, z), or identity if no state yet."""
        state = self.get_state()
        return state.imu_quat_wxyz if state is not None else (1.0, 0.0, 0.0, 0.0)

    # -- direct joint control ------------------------------------------------------------

    def set_joint_targets(
        self, q: Union[Dict[str, float], Sequence[float]], duration: float = 0.0
    ) -> None:
        """Command joint targets directly, optionally with a minimum-jerk transition.

        Args:
            q: Target joint angles (in radians), as a joint dict or a length-12 array in
                ``kromatics.JOINT_NAMES`` order. Joints not present in a dict keep their
                current target.
            duration: If > 0, interpolate from the current targets over this many seconds. If 0,
                jump immediately (only safe for small corrections).

        """
        target_dict = K.Kromatics.to_dict(q) if not isinstance(q, dict) else dict(q)
        if duration <= 0.0:
            with self._targets_lock:
                self._current_targets.update(target_dict)
                self._active_move = None
            return

        with self._targets_lock:
            start = dict(self._current_targets)
        self.play_move(_InterpolationMove(start, target_dict, duration), name="_goto")

    # -- high-level poses -----------------------------------------------------------------

    def stand_pose(self) -> Dict[str, float]:
        """Return the nominal standing joint configuration (``kromatics.WAKE_UP_POSE``)."""
        return dict(K.WAKE_UP_POSE)

    def approach_pose(self) -> Dict[str, float]:
        """Return the compact "legs pulled in" pose (``kromatics.APPROACH_POSE``)."""
        return dict(K.APPROACH_POSE)

    def rest_pose(self) -> Dict[str, float]:
        """Return the compact resting joint configuration (``kromatics.REST_POSE``)."""
        return dict(K.REST_POSE)

    def approach(self, duration: Optional[float] = None) -> None:
        """Smoothly pull the legs in to :meth:`approach_pose`, from whatever configuration."""
        self.set_joint_targets(
            self.approach_pose(), duration=duration or self.approach_duration_s
        )

    def wake_up(self, duration: Optional[float] = None, play_sound: bool = True) -> None:
        """Smoothly stand up from whatever configuration the robot is currently in.

        Args:
            duration: Transition time, in seconds (default: ``config["control"]["wake_up_duration_s"]``).
            play_sound: If True (default), plays :data:`DEFAULT_CHROMAPI_SCREAM_PATH`.

        """
        self.enable_motors()
        total_duration = duration or self.wake_up_duration_s
        target = self.stand_pose()
        current = self.get_current_joint_positions()

        yaw_joints = [K.joint_name(leg, 1) for leg in K.LEG_NAMES]
        max_yaw_delta = max(
            abs(target[joint] - current.get(joint, 0.0)) for joint in yaw_joints
        )
        if max_yaw_delta <= _YAW_PRESTAGE_THRESHOLD_RAD:
            self.set_joint_targets(target, duration=total_duration)
            return

        logger.info(
            "wake_up: pre-staging a %.1f deg hip-yaw correction at the current stance "
            "height before rising",
            np.degrees(max_yaw_delta),
        )
        yaw_prestage = dict(current)
        for joint in yaw_joints:
            yaw_prestage[joint] = target[joint]
        prestage_duration = total_duration * 0.6
        rise_duration = total_duration * 0.4
        self.set_joint_targets(yaw_prestage, duration=prestage_duration)
        time.sleep(prestage_duration + _YAW_PRESTAGE_SETTLE_S)
        self.set_joint_targets(target, duration=rise_duration)
        # time.sleep(rise_duration + 0.1)
        # if play_sound:
        #     from chromapi.media.audio import play_wav
        #     play_wav(DEFAULT_WAKE_UP_SOUND_PATH, wait=False)

    def rest(self, duration: Optional[float] = None, disable_torque: bool = True) -> None:
        """Smoothly crouch down to :meth:`rest_pose` and optionally release motor torque.

        Args:
            duration: Transition time, in seconds (default: ``config["control"]["rest_duration_s"]``).
            disable_torque: If True (default), disable torque once crouch is reached.

        """
        transition_duration = duration or self.rest_duration_s
        self.set_joint_targets(self.rest_pose(), duration=transition_duration)
        if disable_torque:
            time.sleep(transition_duration + 0.1)
            self.disable_motors()

    def go_to_pose(
        self,
        body_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        body_rpy: Sequence[float] = (0.0, 0.0, 0.0),
        duration: float = 1.0,
        reference_height: Optional[float] = None,
        x_reach: Optional[float] = None,
        y_reach: Optional[float] = None,
    ) -> bool:
        """Tilt the trunk while standing, feet planted (quasi-static body posing).

        Args:
            body_xyz: Translation of the chassis relative to the nominal stand pose, in
                meters (world/ground frame).
            body_rpy: Orientation of the chassis relative to level, in radians.
            duration: Transition time, in seconds.
            reference_height: Stance height for the default footprint, in meters.
            x_reach: Footprint half-width along chassis X for the same default footprint.
                y_reach: Footprint half-width along chassis Y for the same default footprint.

        Returns:
            True if the target pose was reachable by all 4 legs (the transition is still
            started either way, converging to the closest reachable configuration).

        """
        height = (
            reference_height
            if reference_height is not None
            else K.Kromatics._STAND_REACH[2]
        )
        x_reach_m = x_reach if x_reach is not None else K.Kromatics._STAND_REACH[0]
        y_reach_m = y_reach if y_reach is not None else K.Kromatics._STAND_REACH[1]
        footprint = self.kinematics.stance_targets(height, x_reach_m, y_reach_m)
        q_target, converged = self.kinematics.body_ik(
            np.asarray(body_xyz),
            np.asarray(body_rpy),
            foot_targets_world=footprint,
            reference_height=height,
        )
        if not converged:
            logger.warning("go_to_pose target is outside the reachable envelope for some leg")
        self.set_joint_targets(q_target, duration=duration)
        return converged

    # -- walking --------------------------------------------------------------------------

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        pattern: str = "crawl",
        params: Optional["GaitParams"] = None,
    ) -> "WalkMove":
        """Start (or update) open-loop walking at the given body-frame velocity.

        Args:
            vx: Commanded forward velocity, in m/s (positive = forward).
            vy: Commanded lateral velocity, in m/s (positive = left).
            wz: Commanded yaw rate, in rad/s (positive = turn left).
            pattern: ``"crawl"`` (default, statically stable) or ``"trot"`` (faster, dynamic).
            params: Explicit :class:`~chromapi.locomotion.gait.GaitParams`, overriding
                ``pattern``.

        Returns:
            The active :class:`~chromapi.locomotion.gait.WalkMove`

        """
        from chromapi.locomotion.gait import GaitParams as _GaitParams
        from chromapi.locomotion.gait import WalkMove as _WalkMove

        if params is not None:
            resolved_params = params
        elif pattern == "trot":
            resolved_params = _GaitParams.trot()
        elif pattern == "crawl":
            resolved_params = _GaitParams.crawl()
        else:
            resolved_params = _GaitParams(pattern=pattern)

        current = self._walk_move
        if current is None or resolved_params != current.gait.params:
            current = _WalkMove(self.kinematics, resolved_params)
            self._walk_move = current
        current.set_velocity(vy, -vx, wz)  # chassis +X=left, +Y=backward - see this method's docstring
        if self._active_move is not current:
            self.play_move(current, name="_walk")
        return current

    def stop_walk(self) -> None:
        """Ramp the walking velocity to zero, settling the feet at the neutral stance footprint."""
        if self._walk_move is not None:
            self._walk_move.set_velocity(0.0, 0.0, 0.0)

    def walk_distance(
        self,
        distance_m: float,
        speed: float = 0.08,
        lateral: bool = False,
        **walk_kwargs: Any,
    ) -> None:
        """Walk forward (or sideways) approximately ``distance_m``, then stop.

        Args:
            distance_m: Signed distance to walk, in meters. Positive = forward (or left, with
                ``lateral=True``).
            speed: Commanded walking speed, in m/s (magnitude only).
            lateral: If True, move sideways (left/right) instead of forward/backward.
            **walk_kwargs: Forwarded to :meth:`walk` (``pattern``, ``params``).

        """
        if speed <= 0.0:
            raise ValueError(f"speed must be positive, got {speed}")
        direction = 1.0 if distance_m >= 0.0 else -1.0
        if lateral:
            self.walk(vx=0.0, vy=direction * speed, wz=0.0, **walk_kwargs)
        else:
            self.walk(vx=direction * speed, vy=0.0, wz=0.0, **walk_kwargs)
        time.sleep(abs(distance_m) / speed)
        self.stop_walk()

    def turn(
        self,
        degrees: float,
        angular_speed_deg_s: float = 30.0,
        **walk_kwargs: Any,
    ) -> None:
        """Turn in place by approximately ``degrees``, then stop.

        Args:
            degrees: Signed angle to turn, in degrees. Positive = left.
            angular_speed_deg_s: Commanded turning speed, in degrees/s (magnitude only).
            **walk_kwargs: Forwarded to :meth:`walk` (``pattern``, ``params``).

        """
        if angular_speed_deg_s <= 0.0:
            raise ValueError(f"angular_speed_deg_s must be positive, got {angular_speed_deg_s}")
        direction = 1.0 if degrees >= 0.0 else -1.0
        angular_speed_rad_s = np.radians(angular_speed_deg_s)
        self.walk(vx=0.0, vy=0.0, wz=direction * angular_speed_rad_s, **walk_kwargs)
        time.sleep(abs(np.radians(degrees)) / angular_speed_rad_s)
        self.stop_walk()

    # -- motors / LEDs ----------------------------------------------------------------

    def enable_motors(self, joints: Optional[Sequence[str]] = None) -> None:
        """Enable actuator torque (STS3215 ``TORQUE_SWITCH``) on the given joints (default: all)."""
        self.backend.set_torque_enabled(True, joints)

    def disable_motors(self, joints: Optional[Sequence[str]] = None) -> None:
        """Disable actuator torque on the given joints (default: all) - motors go limp."""
        self.backend.set_torque_enabled(False, joints)

    def set_led_all(self, r: int, g: int, b: int) -> None:
        """Set every ring LED to one RGB color."""
        self.backend.set_led_all(r, g, b)

    def set_led(self, index: int, r: int, g: int, b: int) -> None:
        """Set a single ring LED (0-17)."""
        self.backend.set_led(index, r, g, b)

    def set_led_bulk(self, colors: Sequence[Tuple[int, int, int]]) -> None:
        """Set all 18 ring LEDs at once from a list of (r, g, b) tuples."""
        self.backend.set_led_bulk(colors)

    # -- moves / behaviors --------------------------------------------------------------

    def register_move(self, name: str, move: Move) -> None:
        """Register a :class:`Move` under a name, for later :meth:`play_move`."""
        self._registered_moves[name] = move

    def play_move(self, move: Union[str, Move], name: str = "") -> None:
        """Activate a :class:`Move` - either by registered name, or directly.

        Stops whichever move is currently active (calling its ``on_stop``) first.
        """
        resolved = self._registered_moves[move] if isinstance(move, str) else move
        with self._targets_lock:
            previous = self._active_move
            self._active_move = None
        if previous is not None:
            previous.on_stop()
        resolved.on_start()
        with self._targets_lock:
            self._active_move = resolved

    def stop_move(self) -> None:
        """Deactivate the current move, freezing the joint targets at their last value."""
        with self._targets_lock:
            move = self._active_move
            self._active_move = None
        if move is not None:
            move.on_stop()

    def cancel_move(self) -> None:
        """Alias for :meth:`stop_move`."""
        self.stop_move()


class _InterpolationMove(Move):
    """Internal one-shot minimum-jerk transition between two joint configurations."""

    def __init__(self, start: Dict[str, float], target: Dict[str, float], duration: float) -> None:
        """Store the endpoints; ``duration`` <= 0 completes on the first tick."""
        self._start = start
        self._target = target
        self._duration = max(duration, 1e-6)
        self._elapsed = 0.0

    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """Advance the interpolation by ``dt`` and write the blended targets."""
        self._elapsed += dt
        s = _minimum_jerk(self._elapsed / self._duration)
        for joint, target in self._target.items():
            start = self._start.get(joint, target)
            command.target_angles[joint] = (1.0 - s) * start + s * target
