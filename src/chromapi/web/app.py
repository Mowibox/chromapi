# SPDX-License-Identifier: Apache-2.0
"""Chromapi web app: drive Chromapi from a phone or a PC browser, watch its state, look after it.

Chromapi only moves on "Wake up"; the page sends intentions, never joint angles, and its sticks centre after
:data:`INPUT_TIMEOUT_S` of silence.

Usage::

    chromapi run                      # on the robot
    chromapi run --sim                # on a PC, in simulation (MuJoCo)
"""

import argparse
import functools
import getpass
import io
import json
import logging
import math
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from flask import Flask, Response, request, send_file, send_from_directory, session
from flask_socketio import SocketIO

from chromapi.chromapi import (
    AUDIO_DIR,
    IDLE_LED_RGB,
    Chromapi,
    HardwareBackend,
    MuJoCoBackend,
    load_config,
    sdk_version,
)
from chromapi.estimation.attitude import tilt
from chromapi.hardware import fan as fan_service
from chromapi.hardware.battery import CRITICAL, EMPTY, LOW, OK
from chromapi.input.gamepad import GamepadInput
from chromapi.locomotion.balance import BalanceMove
from chromapi.locomotion.gait import WALK_POSTURE
from chromapi.locomotion.recorded_move import (
    EMOTE_MAX_TILT_DEG,
    RecordedLegMove,
    RecordedMove,
    RecordedMoves,
)
from chromapi.media import led
from chromapi.utils.flash import shipped_version, update_available
from chromapi.utils.report import write_report
from chromapi.web.access import MIN_CODE_LENGTH, Access, tls_certificate

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INPUT_TIMEOUT_S = 0.4  # [s]
JOYSTICK_DEADZONE = 0.12
NOMINAL_LINEAR_MPS = 0.12  # [m.s⁻¹] at 100 %
MAX_SPEED_SCALE = 2.0
LOW_BATTERY_SPEED_SCALE = 1.0  # speed cap when the battery is low
MIN_SPEED_SCALE = 0.25
BATTERY_MOODS: Dict[Optional[str], str] = {LOW: "tired", CRITICAL: "alert"}
_LEVEL_RANK: Dict[Optional[str], int] = {None: 0, OK: 0, LOW: 1, CRITICAL: 2, EMPTY: 3}
MAX_ANGULAR_RAD_S = math.radians(40.0)  # [rad.s⁻¹]
MAX_LEAN_RAD = math.radians(6.0)  # [rad] at full stick
MAX_LEAN_SHIFT_M = 0.03  # [m] at full stick
HEIGHT_RANGE_M = (-0.04, 0.02)  # [m]
TICK_S = 0.02  # [s] teleop loop
# Gamepad: DualShock-like raw joystick numbers (``jstest /dev/input/js0``)
PAD_DEVICE = "/dev/input/js0"
PAD_CROSS, PAD_CIRCLE, PAD_TRIANGLE, PAD_SQUARE = "0", "1", "2", "3"
PAD_L1, PAD_L2, PAD_R2, PAD_SHARE, PAD_START, PAD_L3, PAD_R3 = "4", "6", "7", "8", "9", "11", "12"
PAD_IDLE_S = 1.0  # [s] at rest this long, the gamepad hands the sticks back to the page
PAD_RETRY_S = 2.0  # [s] how often to look for a gamepad
PAD_SHUTDOWN_HOLD_S = 3.0  # [s] START held
PAD_HEIGHT_RATE_MPS = 0.02  # [m.s⁻¹] D-pad up/down held
PAD_SPEED_STEP = 0.25  # per D-pad press
PAD_TILT_FULL_RAD = math.radians(30.0)  # [rad] controller tilt for a full lean
PAD_TILT_SIGNS = (1.0, -1.0)  # (forward, right)
PAD_FLASH_S = led.BATTERY_FLASH_S  # [s] battery or mood shown on the light bar
PAD_BATTERY_LOW_PERCENT = 20
PAD_BATTERY_CHECK_S = 10.0  # [s]
TELEMETRY_PERIOD_S = 0.2  # [s]
STATE_STALE_S = 1.0  # [s]
RECONNECT_S = 5.0  # [s]
HELD_LOAD = 30.0  # [0.1 % rated torque] above it at start-up, the firmware still holds the robot up
EMPTY_SHUTDOWN_S = 10.0  # [s] at 0 %, time to lie down before powering the Pi off
MAINTENANCE = {
    "calibrate_imu": (["calibrate", "imu"], "Calibrating the IMU: keep Chromapi still on a level floor"),
    "calibrate_servos": (["calibrate", "servos", "--all", "--yes"], "Calibrating the servos: keep the legs clear"),
    "update_firmware": (["flash", "--yes"], "Updating the firmware: do not switch the robot off"),
    "pair_gamepad": (["gamepad", "pair"], "Pairing: hold the controller's pair button until its light flashes fast"),
}
# New servos come with ID 1: set them up from the highest ID down
SERVO_SETUP = sorted(load_config()["servos"]["id_map"].items(), key=lambda joint_id: -joint_id[1])
MAINTENANCE.update({f"setup_servo_{i}": (["motors", "setup", "--id", str(i)], f"Setting up servo {i} ({joint})")
                    for joint, i in SERVO_SETUP})
MAINTENANCE_TIMEOUT_S = 15 * 60  # [s]
MAINTENANCE_CANCEL_S = 5.0  # [s] then killed
UNCANCELLABLE = {"update_firmware"}  # stopped mid-flash, the board stays in reset
POWER_COMMANDS = {"shutdown": "poweroff", "reboot": "reboot", "quit": None}  # page action -> systemctl command
# Talkie-walkie
TALK_RATE = 16000  # [Hz]
TALK_IDLE_S = 0.5  # [s]
TALK_MAX_CHUNK = 2 * TALK_RATE

RECORDING_SUFFIXES = ("_photo.jpg", "_camera.avi")
OPEN_ENDPOINTS = {"index", "favicon", "static", "auth_state", "auth_setup", "auth_login"}
SESSION_LIFETIME_S = 365 * 24 * 3600  # [s]
CODE_CHECK_S = 1.0  # [s] a new access code closes the open connections within this
GREETING_SOUND = AUDIO_DIR / "greeting.wav"
DATASET_MOOD = "happy"  # ring and sound of a dataset move that names no (known) mood
FAVICON = AUDIO_DIR.parent / "favicon.ico"


def _in_circle(x: float, y: float) -> Tuple[float, float]:
    """Clamp a stick to the unit circle."""
    norm = math.hypot(x, y)
    return (x / norm, y / norm) if norm > 1.0 else (x, y)


class WebInput:
    """Latest sticks from the page, centred when the page goes silent."""

    def __init__(self) -> None:
        """Sticks centred, no message yet."""
        self._lock = threading.Lock()
        self._axes: Dict[str, float] = {}
        self._last_s = float("-inf")  # [s]

    def set_axes(self, axes: Dict[str, Any]) -> None:
        """Store ``lx``, ``ly``, ``rx``, ``ry`` in [-1, 1] (y up-positive), with a deadzone."""
        clean = {}
        for key in ("lx", "ly", "rx", "ry"):
            value = min(max(float(axes.get(key, 0.0)), -1.0), 1.0)
            clean[key] = 0.0 if abs(value) < JOYSTICK_DEADZONE else value
        with self._lock:
            self._axes, self._last_s = clean, time.monotonic()

    def release(self) -> None:
        """Centre the sticks now (last page disconnected)."""
        with self._lock:
            self._axes = {}

    def read(self) -> Dict[str, float]:
        """The sticks, all zero if the page has been silent for :data:`INPUT_TIMEOUT_S`."""
        with self._lock:
            if time.monotonic() - self._last_s >= INPUT_TIMEOUT_S:
                self._axes = {}
            return {key: self._axes.get(key, 0.0) for key in ("lx", "ly", "rx", "ry")}


class GamepadControl:
    """The gamepad, driving the same :class:`Teleop` as the page; it takes over while moved."""

    def __init__(self, device: str = PAD_DEVICE, open_pad: Callable[[str], Any] = GamepadInput) -> None:
        """Nothing opened yet: :meth:`poll` looks for the device every :data:`PAD_RETRY_S`."""
        self.device, self._open = device, open_pad
        self._pad: Any = None
        self._next_try = float("-inf")  # [s]
        self._failures = 0  # opening attempts failed in a row
        self._held_before: Optional[Set[str]] = None
        self._polled = float("-inf")  # [s] last poll
        self._shutdown_since: Optional[float] = None  # [s] START held since; -inf once the shutdown is asked
        self._last_active = float("-inf")  # [s]
        self.sticks = dict.fromkeys(("lx", "ly", "rx", "ry"), 0.0)
        self.mood = 0  # index in led.MOODS, chosen with L2/R2
        self.tilt_lean = False  # Settings: in Lean mode, the controller's tilt leans the robot too
        self._tilt_zero: Optional[Tuple[float, float]] = None  # [rad] the controller's pose when tilting started
        self._color: Optional[Tuple[int, int, int]] = None  # last light bar colour sent
        self._flash: Tuple[Tuple[int, int, int], float] = (IDLE_LED_RGB, float("-inf"))  # (colour, until [s])
        self.battery: Optional[Tuple[int, bool]] = None  # (percent, charging) of the controller
        self._battery_next = float("-inf")  # [s]
        self._battery_warned = False

    @property
    def connected(self) -> bool:
        """A gamepad is open and still there."""
        return self._pad is not None and self._pad.connected

    def active(self, now: float) -> bool:
        """The gamepad moved within :data:`PAD_IDLE_S`: it drives, the page's sticks wait."""
        return self.connected and now - self._last_active < PAD_IDLE_S

    def rumble(self, pulses: int) -> None:
        """Buzz the controller, if there is one."""
        if self.connected:
            self._pad.rumble(pulses)

    def close(self) -> None:
        """The app stops: the light bar gets its own colour back."""
        if self._pad is not None:
            self._pad.stop()
            self._pad = None

    def _reconnect(self, now: float) -> bool:
        if self.connected:
            return True
        self.close()
        self.sticks = dict.fromkeys(self.sticks, 0.0)
        self._held_before, self._shutdown_since, self._tilt_zero, self._color = None, None, None, None
        self.battery, self._battery_next, self._battery_warned = None, float("-inf"), False
        if now < self._next_try or not os.path.exists(self.device):
            return False
        self._next_try = now + PAD_RETRY_S
        pad = self._open(self.device)
        try:
            pad.start()
        except OSError as exc:
            self._failures += 1  # a just-paired pad fails once, before udev sets js0's group
            if self._failures == 2:
                logger.warning("Gamepad: %s", exc)
            return False
        logger.info("Gamepad connected on %s", self.device)
        self._pad, self._failures = pad, 0
        return True

    def poll(self, teleop: "Teleop", now: float) -> None:
        """Read the gamepad: sticks for :meth:`Teleop._drive`, buttons as the page's actions."""
        if not self._reconnect(now):
            return
        user = self._pad.read()
        self.sticks = {"lx": user.lateral, "ly": user.forward, "rx": -user.turn, "ry": user.vertical}
        self._pad.motion = self.tilt_lean and teleop.mode == "lean"
        if self._pad.motion:
            self._tilt_sticks()
        else:
            self._tilt_zero = None
        held = set(user.held)
        if held or any(self.sticks.values()):
            self._last_active = now
        # The D-pad only reports held directions: a press is a new one
        before = held if self._held_before is None else self._held_before
        self._held_before = held
        for button in sorted(set(user.pressed) | (held - before)):
            self._press(button, teleop, now)
        self._hold_to_shutdown(held, teleop, now)
        dt, self._polled = min(now - self._polled, 2 * TICK_S), now  # [s]
        rise = ("dpad_up" in held) - ("dpad_down" in held)
        if rise:
            teleop.height = min(max(teleop.height + rise * PAD_HEIGHT_RATE_MPS * dt, HEIGHT_RANGE_M[0]),
                                HEIGHT_RANGE_M[1])
        color = self._flash[0] if now < self._flash[1] else led.dimmed(led.idle_color())
        if color != self._color:
            self._pad.set_color(color)
            self._color = color
        self._check_battery(teleop, now)

    def _tilt_sticks(self) -> None:
        """The controller's tilt since Lean started, as the right stick."""
        gravity = self._pad.tilt()
        if gravity is None:
            return  # no motion sensors: the right stick still leans
        x, y, z = gravity
        norm = math.sqrt(x * x + y * y + z * z) or 1.0
        forward, right = math.atan2(z, y), math.asin(min(max(x / norm, -1.0), 1.0))  # [rad]
        if self._tilt_zero is None:
            self._tilt_zero = (forward, right)
        if self.sticks["rx"] or self.sticks["ry"]:
            return
        tilt = ((forward - self._tilt_zero[0] + math.pi) % (2 * math.pi) - math.pi, right - self._tilt_zero[1])
        for key, angle, sign in zip(("ry", "rx"), tilt, PAD_TILT_SIGNS):
            value = min(max(sign * angle / PAD_TILT_FULL_RAD, -1.0), 1.0)
            self.sticks[key] = 0.0 if abs(value) < JOYSTICK_DEADZONE else value

    def _press(self, button: str, teleop: "Teleop", now: float) -> None:
        names = [mood.name for mood in led.MOODS]
        if button == PAD_CROSS:
            teleop.request("rest" if teleop.awake else "wake")
        elif button == PAD_SQUARE:
            teleop.request("greet")
        elif button == PAD_TRIANGLE:
            teleop.request("kick")
        elif button == PAD_CIRCLE:
            teleop.request(f"mood:{names[self.mood]}")
        elif button in (PAD_L2, PAD_R2):
            self.mood = (self.mood + (1 if button == PAD_R2 else -1)) % len(names)
            teleop.status = f"Mood: {names[self.mood]} (Circle plays it)"
            self._flash = (led.PAD_MOOD_COLORS[names[self.mood]], now + PAD_FLASH_S)
        elif button == PAD_L1:
            teleop.ring.torch = not teleop.ring.torch
        elif button == PAD_L3:
            teleop.request("crawl" if teleop.gait == "trot" else "trot")
        elif button == PAD_R3:
            teleop.request("lean" if teleop.mode == "turn" else "turn")
        elif button == PAD_SHARE:
            teleop.request("show_battery")
            percent = teleop.robot.battery.percent
            if percent is not None:
                self._flash = (led.battery_color(percent), now + PAD_FLASH_S)
        elif button in ("dpad_left", "dpad_right"):
            step = PAD_SPEED_STEP if button == "dpad_right" else -PAD_SPEED_STEP
            teleop.speed_scale = min(max(teleop.speed_scale + step, MIN_SPEED_SCALE), teleop.max_speed_scale)

    def _hold_to_shutdown(self, held: Set[str], teleop: "Teleop", now: float) -> None:
        """START held :data:`PAD_SHUTDOWN_HOLD_S` powers off."""
        if PAD_START not in held:
            self._shutdown_since = None
            return
        if self._shutdown_since is None:
            self._shutdown_since = now
        left = PAD_SHUTDOWN_HOLD_S - (now - self._shutdown_since)  # [s]; -inf once asked
        if left <= 0.0 and self._shutdown_since != float("-inf"):
            self._shutdown_since = float("-inf")  # once per hold
            teleop.request("shutdown")
        elif left > 0.0:
            teleop.status = f"Keep holding START to power off ({math.ceil(left):.0f} s)"

    def _check_battery(self, teleop: "Teleop", now: float) -> None:
        if now < self._battery_next:
            return
        self._battery_next = now + PAD_BATTERY_CHECK_S
        self.battery = self._pad.battery()
        if self.battery is None:
            return
        percent, charging = self.battery
        if charging or percent > PAD_BATTERY_LOW_PERCENT:
            self._battery_warned = False
        elif not self._battery_warned:
            self._battery_warned = True
            self._pad.rumble(2)
            teleop.status = f"Controller battery low: {percent} %"


class Teleop:
    """What Chromapi is doing; every robot call happens in :meth:`tick`'s thread."""

    def __init__(self, robot: Chromapi, web_input: WebInput, hardware: bool,
                 pad: Optional[GamepadControl] = None) -> None:
        """Start lying down, trot, turn mode, full speed."""
        self.robot = robot
        self.web_input = web_input
        self.pad = pad
        self.hardware = hardware
        self.ring = led.LedRing(robot)
        self.awake = False
        self.gait = "trot"
        self.mode = "turn"  # right stick turns; "lean": it tilts the chassis
        self._lean_target: Optional[Tuple[float, ...]] = None  # last (x, y, roll, pitch) sticks sent
        self._leaning = False
        self._lean_height0 = 0.0  # [m] height offset when Lean started
        self.height = 0.0  # [m] chassis height offset, both modes
        self.speed_scale = 1.0
        self.status = "Ready: press Wake up"
        self.shutting_down = False
        self.on_shutdown: Callable[[str], None] = lambda power: None  # "shutdown", "reboot" or "quit", to the pages
        self._actions: "queue.Queue[str]" = queue.Queue()
        self._emote: Optional[Union[RecordedLegMove, RecordedMove]] = None
        self._next_greeting_leg = "tl"
        self._next_kick_leg = "tr"
        self._body = BalanceMove(dt=1.0 / robot.loop_hz)  # Lean and the emotes' weight shift
        self._moves = RecordedMoves()  # mood gestures, greeting and kick: assets/moves
        dataset = (getattr(robot, "config", None) or {}).get("moves", {}).get("dataset")
        if dataset:  # downloaded (or the cache) without holding the start-up
            threading.Thread(target=self._load_dataset, args=(dataset,), daemon=True).start()
        self._empty_since: Optional[float] = None  # [s]
        self._battery_rank = 0  # lowest level seen (_LEVEL_RANK)
        self._battery_mood: Optional[str] = None  # played once the emote ends
        self._greeted = False
        self.maintenance: Dict[str, Any] = {}  # job, running, ok, cancelled, log
        self._tool: Optional["subprocess.Popen[str]"] = None
        self._talk_out: Any = None  # PersistentAudioOutput
        self._talk_last = 0.0  # [s]
        self._talk_lock = threading.Lock()
        self._mic: Any = None  # MicStream
        self._listen_next: Optional[Callable[[bytes], None]] = None  # opens once the talk ends

    def request(self, action: str) -> None:
        """Queue an action (see :meth:`_do`); ``maintenance_cancel`` acts at once."""
        if action == "maintenance_cancel":
            self.cancel_maintenance()
        else:
            self._actions.put(action)

    def talk(self, pcm: bytes) -> None:
        """Play a chunk of the page's microphone (``TALK_RATE`` mono int16) on Chromapi's speaker."""
        if not pcm or len(pcm) > TALK_MAX_CHUNK or len(pcm) % 2:
            return
        self.listen_stop()  # one way at a time
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) * (self.robot.sound_volume / 32768.0)
        with self._talk_lock:
            if self._talk_out is None:
                self._talk_out = self._open_speaker()
                if self._talk_out is None:
                    return
            self._talk_out.write(samples, TALK_RATE)
            self._talk_last = time.monotonic()

    def _open_speaker(self) -> Any:
        """A live output on Chromapi's speaker, at TALK_RATE if the device takes it."""
        from chromapi.media.audio import PersistentAudioOutput, find_output_device

        hint = self.robot._speaker_hint
        device = find_output_device(hint) if hint else None
        for rate in (TALK_RATE, None):  # None: the device's rate
            out = PersistentAudioOutput(samplerate=rate, device=device)
            if out.start():
                return out
        self.status = "Talkie-walkie: no speaker"
        return None

    @property
    def talking(self) -> bool:
        """The page's voice is playing on the robot."""
        return self._talk_out is not None

    def listen_start(self, send: Callable[[bytes], None]) -> bool:
        """Stream Chromapi's microphone (``TALK_RATE`` mono int16 chunks) to ``send``, after any talk."""
        if self._mic is not None:
            return False
        with self._talk_lock:
            if self.talking:
                self._listen_next = send
                return True
        from chromapi.media.audio import MicStream

        cfg = self.robot.config.get("audio", {}) if self.hardware else {}
        mic = MicStream(send, samplerate=TALK_RATE, gain=float(cfg.get("mic_gain", 1.0)),
                        denoise=bool(cfg.get("mic_denoise", True)))
        if not mic.start():
            self.status = "Listening: no microphone"
            return False
        self._mic = mic
        self.ring.listening = True
        return True

    def listen_stop(self) -> None:
        """Close Chromapi's microphone."""
        self._listen_next = None
        mic, self._mic = self._mic, None
        if mic is not None:
            mic.stop()
            self.ring.listening = False

    def _close_idle_talk(self) -> None:
        with self._talk_lock:
            if self._talk_out is not None and time.monotonic() - self._talk_last > TALK_IDLE_S:
                self._talk_out.stop()
                self._talk_out = None
            resume, self._listen_next = self._listen_next, None
        if resume is not None:
            self.listen_start(resume)

    def cancel_maintenance(self) -> None:
        """Ctrl-C the running maintenance tool (not a flash)."""
        tool = self._tool
        if self.maintenance.get("job") in UNCANCELLABLE or tool is None or tool.poll() is not None:
            return
        self.maintenance["cancelled"] = True
        self.status = "Cancelling..."
        tool.send_signal(signal.SIGINT)
        threading.Timer(MAINTENANCE_CANCEL_S, tool.kill).start()

    @property
    def walking(self) -> bool:
        """A walk is under way (not settled on the neutral stance)."""
        return self.robot.is_walking

    @property
    def max_speed_scale(self) -> float:
        """Speed slider limit, lower when the battery is low."""
        return LOW_BATTERY_SPEED_SCALE if self.robot.battery.level in (LOW, CRITICAL, EMPTY) else MAX_SPEED_SCALE

    def _cap_speed(self) -> None:
        if self.speed_scale > self.max_speed_scale:
            self.speed_scale = self.max_speed_scale
            self.status = f"Battery low: speed capped at {self.max_speed_scale:.0%}"

    @property
    def emote_running(self) -> bool:
        """An emote is playing."""
        return self._emote is not None

    def tick(self) -> None:
        """One loop step: actions, emote, battery, sticks."""
        self._close_idle_talk()
        if self.maintenance.get("running"):
            return
        if self.pad is not None:
            self.pad.poll(self, time.monotonic())
        while not self._actions.empty():
            self._do(self._actions.get())
        if self._emote is not None and (self._emote.done or self.robot.active_move is not self._emote):
            if self.robot.active_move is self._emote:
                self.robot.stop_move()
            self._emote = None
        self._check_battery()
        self._cap_speed()
        if self.awake and self._emote is None:
            self._drive()

    def _do(self, action: str) -> None:
        if action == "wake" and not self.awake:
            self.status = "Waking up"
            self.robot.wake_up(wait=True)
            self.ring.set_persistent(led.render_idle)
            self.awake = True
            self._lean_target, self._leaning = None, False
            self.status = "Standing"
        elif action == "rest" and self.awake:
            self._lie_down()
            self.status = "Lying down"
        elif action in ("trot", "crawl"):
            if self.walking:
                self.status = "Stop walking before changing the gait"
            else:
                self.gait = action
                self.status = f"Gait: {action}"
        elif action in ("turn", "lean"):
            if self.walking or self._emote is not None:
                self.status = "Stop moving before changing the mode"
            else:
                if action == "turn" and self._leaning:
                    self._body.set_body_pose()  # back to the start pose; walking takes over from there
                self.mode, self._lean_target, self._leaning = action, None, False
                self.status = f"Mode: {action}"
        elif action in ("greet", "kick", "kick_left", "kick_right"):
            self._start_emote(action)
        elif action == "identify":
            self.ring.play_mood("scream")
        elif action == "show_battery":
            percent = self.robot.battery.percent
            if percent is None:
                self.status = "Battery not read yet"
            else:
                self.ring.flash(led.render_battery(percent), led.BATTERY_FLASH_S)
        elif action.startswith("mood:") and (action[5:] in led.MOODS_BY_NAME or action[5:] in self.extra_moves()):
            self._play_mood(action[5:])
        elif action in MAINTENANCE:
            self._start_maintenance(action)
        elif action in POWER_COMMANDS:
            self._shutdown(POWER_COMMANDS[action])

    def resume_if_held(self) -> None:
        """At start-up, take over a robot the firmware still holds standing."""
        state = self.robot.get_state()
        load = max((abs(v) for v in state.joint_loads.values()), default=0.0) if state else 0.0
        logger.info("Start-up servo load: %.0f (held above %.0f)", load, HELD_LOAD)
        if self.hardware and load > HELD_LOAD:
            self._do("wake")
            self.status = "Still standing: resumed"

    def _lie_down(self) -> None:
        self.robot.stop_walk(wait=True)
        self.ring.set_persistent(led.render_off)
        self.robot.rest()
        self.awake = False
        self._emote, self._leaning = None, False

    def _start_maintenance(self, job: str) -> None:
        if not isinstance(self.robot.backend, HardwareBackend):
            self.status = "Maintenance works on the real robot only"
        elif self.awake:
            self.status = "Maintenance: lay Chromapi down first"
        else:
            self.maintenance = {"job": job, "running": True, "ok": None, "cancelled": False, "log": ""}
            self.status = MAINTENANCE[job][1]
            threading.Thread(target=self._maintain, args=(job,), name="chromapi-maintenance", daemon=True).start()

    def _maintain(self, job: str) -> None:
        """Hand the serial port to the ``chromapi`` tool, then take it back with the new calibration."""
        self.ring.stop()
        self.robot.disconnect()
        try:
            with subprocess.Popen([sys.executable, "-m", "chromapi", *MAINTENANCE[job][0]], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as tool:
                self._tool = tool
                try:
                    log = tool.communicate(timeout=MAINTENANCE_TIMEOUT_S)[0]
                except subprocess.TimeoutExpired:
                    tool.kill()
                    log = tool.communicate()[0] + f"\ntimed out after {MAINTENANCE_TIMEOUT_S:.0f} s"
            ok, log = tool.returncode == 0, log.strip()
        except OSError as exc:
            ok, log = False, str(exc)
        self._tool = None
        cancelled = bool(self.maintenance.get("cancelled"))
        logger.info("Maintenance %s %s:\n%s", job, "cancelled" if cancelled else "done" if ok else "failed", log)
        if ok and isinstance(self.robot.backend, HardwareBackend):
            self.robot.config = load_config()
            self.robot.backend.apply_calibration(self.robot.config)
        if self.robot.connect():
            self.robot.disable_motors()  # the servo sweep leaves torque on
        self.ring.start()
        last = log.splitlines()[-1] if log else ""
        outcome = "Cancelled" if cancelled else "Done" if ok else "Failed"
        self.status = f"{outcome}: {last}" if last else outcome
        self.maintenance = {"job": job, "running": False, "ok": ok, "cancelled": cancelled,
                            "log": "\n".join(log.splitlines()[-40:])}

    def _load_dataset(self, dataset: str) -> None:
        """Add the moves of the Hugging Face dataset ``config["moves"]["dataset"]`` to the bundled ones."""
        try:
            self._moves.update(RecordedMoves(dataset))
        except Exception as exc:  # no huggingface_hub, never downloaded and offline, unknown dataset
            logger.warning("Moves of %s not available: %s", dataset, exc)

    def extra_moves(self) -> List[str]:
        """Body moves without a mood of their own (from a dataset): listed after the moods."""
        return [n for n in self._moves.list_moves() if n not in led.MOODS_BY_NAME and not self._moves.is_leg_move(n)]

    def _play_mood(self, name: str) -> None:
        """Sound and ring, plus the recorded move of the same name when Chromapi can move."""
        duration_s = 0.0  # [s]
        if name in self._moves.list_moves() and self._emote_refusal() is None:
            gesture = self._moves.get(name, self._body)
            self._emote, self._leaning = gesture, False
            self.robot.play_move(gesture)
            duration_s = gesture.duration
        mood = name if name in led.MOODS_BY_NAME else self._moves.mood(name)
        self.ring.play_mood(mood if mood in led.MOODS_BY_NAME else DATASET_MOOD, duration_s)

    def _emote_refusal(self) -> Optional[str]:
        """Why Chromapi cannot play an emote now; None if it can."""
        refusal = None
        if not self.awake:
            refusal = "wake Chromapi up first"
        elif self.mode != "turn":
            refusal = "switch back to Turn mode"
        elif self._emote is not None:
            refusal = "an emote is already playing"
        elif self.robot.is_walking:
            refusal = "wait for Chromapi to stand still"
        else:
            tilt_deg = math.degrees(tilt(self.robot.get_imu_orientation()))
            if tilt_deg > EMOTE_MAX_TILT_DEG:
                refusal = f"Chromapi is not level ({tilt_deg:.0f}°)"
        return refusal

    def _start_emote(self, name: str) -> None:
        refusal = self._emote_refusal()
        if refusal:
            self.status = f"{name.capitalize()}: {refusal}"
            return
        if name == "greet":
            leg, self._next_greeting_leg = self._next_greeting_leg, "tr" if self._next_greeting_leg == "tl" else "tl"
            self._emote = self._moves.get("greeting", self._body, leg)
            self.robot.play_sound(GREETING_SOUND)
        else:
            leg = {"kick_left": "tl", "kick_right": "tr"}.get(name, self._next_kick_leg)
            self._next_kick_leg = "tl" if leg == "tr" else "tr"
            self._emote = self._moves.get("kick", self._body, leg)
        self._leaning = False
        self.robot.play_move(self._emote)
        self.status = name.replace("_", " ").capitalize()

    def _lean(self, x_in: float, y_in: float, roll_in: float, pitch_in: float, height: float) -> None:
        """Pose the chassis, feet planted, from sticks in [-1, 1]."""
        self._lean_target = (x_in, y_in, roll_in, pitch_in, height)
        if not self._leaning or self.robot.active_move is not self._body:
            self.robot.play_move(self._body)
            self._leaning, self._lean_height0 = True, height
        k = min(self.speed_scale, 1.0)
        self._body.set_body_pose(
            (x_in * MAX_LEAN_SHIFT_M * k, y_in * MAX_LEAN_SHIFT_M * k, height - self._lean_height0),
            (roll_in * MAX_LEAN_RAD * k, pitch_in * MAX_LEAN_RAD * k),
        )

    def _drive(self) -> None:
        pad = self.pad
        sticks = pad.sticks if pad is not None and pad.active(time.monotonic()) else self.web_input.read()
        if self.mode == "lean":
            # Chassis frame: +X forward, +Y left; roll > 0 right, pitch > 0 nose down
            shift, tilt = _in_circle(sticks["ly"], -sticks["lx"]), _in_circle(sticks["rx"], sticks["ry"])
            target = (*shift, *tilt, self.height)
            # No ramp: the stick goes straight through, BalanceMove's joint speed limit smooths it (stick circles,
            # MuJoCo: feet slide ~1 mm; 12 mm with shift and tilt circling together at 2 s a turn).
            if target != self._lean_target and (self._lean_target is not None or any(target)):
                self._lean(*target)
            return
        vx = sticks["ly"] * NOMINAL_LINEAR_MPS * self.speed_scale
        vy = -sticks["lx"] * NOMINAL_LINEAR_MPS * self.speed_scale  # stick right, walk() positive left
        wz = -sticks["rx"] * MAX_ANGULAR_RAD_S * self.speed_scale
        height = WALK_POSTURE[2] + self.height  # [m]
        move = self.robot.walk_move
        if vx == vy == wz == 0.0 and self.robot.active_move is not move:
            if (move.height_target if move is not None else WALK_POSTURE[2]) == height:
                return
        try:
            self.robot.walk(vx=vx, vy=vy, wz=wz, pattern=self.gait).set_body_height(height)
        except RuntimeError as exc:  # battery too low
            self.status = str(exc)

    def startup_mood(self) -> str:
        """The startup mood: scream, or the battery's mood if already low."""
        level = self.robot.battery.level
        self._battery_rank = _LEVEL_RANK[level]
        self._greeted = True
        return BATTERY_MOODS.get(level, "alert" if level == EMPTY else "scream")

    def _check_battery(self) -> None:
        level = self.robot.battery.level
        if self._greeted and _LEVEL_RANK[level] > self._battery_rank:
            self._battery_rank = _LEVEL_RANK[level]
            self._battery_mood = BATTERY_MOODS.get(level, self._battery_mood)
            if self.pad is not None and level in (LOW, CRITICAL):
                self.pad.rumble(2 if level == LOW else 3)
        if self._battery_mood is not None and self._emote is None:
            self.ring.play_mood(self._battery_mood)
            self._battery_mood = None
        if level != EMPTY:
            self._empty_since = None
            return
        self.awake = False  # the SDK lies Chromapi down
        self.status = "Battery empty: lying down" + (", then powering off" if self.hardware else "")
        if self._empty_since is None:
            self._empty_since = time.monotonic()
        elif self.hardware and time.monotonic() - self._empty_since > EMPTY_SHUTDOWN_S:
            self._shutdown()

    def _shutdown(self, command: Optional[str] = "poweroff") -> None:
        """Lie down, release the servos, stop the app, then ``systemctl <command>`` (None: stop only)."""
        self.shutting_down = True
        self.on_shutdown({v: k for k, v in POWER_COMMANDS.items()}[command])
        self.status = {"reboot": "Restarting", "poweroff": "Powering off"}.get(command or "", "Developer mode: app stopped")
        if self.awake:
            self._lie_down()
        self.ring.stop()
        self.robot.disconnect(disable_torque=True)
        if command is None:
            logger.info("Developer mode: app stopped, serial port free. Back with 'sudo systemctl start chromapi@%s' "
                        "or by switching Chromapi off and on.", getpass.getuser())
            return
        if self.hardware and subprocess.run(["systemctl", "--no-ask-password", command], check=False).returncode:
            logger.error("Not allowed to %s the Pi: the robot is lying down and limp, use its switch. "
                         "'chromapi run' started as the system service can do it.", command)


def telemetry(robot: Chromapi, teleop: Teleop, camera: Any, connected: bool) -> Dict[str, Any]:
    """State sent to the page."""
    state = robot.get_state()
    stale = state is None or time.monotonic() - state.timestamp > STATE_STALE_S
    fan = fan_service.status() if teleop.hardware else None
    firmware, shipped = getattr(robot.backend, "firmware_version", None), shipped_version()
    data: Dict[str, Any] = {
        "link": "no serial port" if not connected else "STM32 not answering" if stale else "ok",
        "simulation": not teleop.hardware,
        "awake": teleop.awake,
        "gait": teleop.gait,
        "mode": teleop.mode,
        "torch": teleop.ring.torch,
        "ring": "rainbow" if led.idle_rainbow() else "#%02x%02x%02x" % led.idle_color(),
        "fan": fan[0] if fan is not None else None,
        "fan_auto": fan is not None and fan[1],
        "speed_scale": teleop.speed_scale,
        "speed_max": teleop.max_speed_scale,
        "height_mm": round(teleop.height * 1000),
        "walking": teleop.walking,
        "emote": teleop.emote_running,
        "maintenance": teleop.maintenance,
        "status": teleop.status,
        "cpu_temp": fan_service.cpu_temp_c(),
        "host": socket.gethostname(),
        "versions": {"sdk": sdk_version(), "firmware": firmware, "firmware_shipped": shipped,
                     "firmware_update": teleop.hardware and connected and not stale and update_available(firmware, shipped)},
        "battery": {"percent": robot.battery.percent, "level": robot.battery.level},
        "cam_rec": camera.recording_status() if camera is not None else None,
        "viewers": camera.viewers if camera is not None else 0,
        "gamepad": {"battery": teleop.pad.battery and teleop.pad.battery[0], "tilt_lean": teleop.pad.tilt_lean}
        if teleop.pad is not None and teleop.pad.connected else None,
    }
    if state is not None and not stale:
        data.update(
            voltage=state.voltage_v or None,
            current=state.current_a,
            power=state.power_w,
            tilt_deg=math.degrees(tilt(state.imu_quat_wxyz)),
            contacts={leg: bool(on) for leg, on in state.foot_contacts.items()},
            servos={
                "missing": state.missing_servos,
                "max_temp": max(state.joint_temperatures.values(), default=None),
            },
        )
    return data


def build_server(teleop: Teleop, web_input: WebInput, camera: Any, record_dir: Path,
                 snapshot: Any, access: Access, https: bool = False) -> "tuple[Flask, SocketIO]":
    """Flask app (page, camera, recordings, report) and its Socket.IO events, behind the access code."""
    app = Flask(__name__, static_folder=str(STATIC_DIR))
    app.secret_key = access.secret_key
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict", SESSION_COOKIE_SECURE=https,
                      PERMANENT_SESSION_LIFETIME=SESSION_LIFETIME_S)
    socketio = SocketIO(app, async_mode="threading")
    # sid -> access generation, in arrival order: the first page drives, the others watch
    clients: Dict[str, str] = {}
    clients_lock = threading.Lock()

    def pilot_only(handler: Callable[[Any], None]) -> Callable[[Any], None]:
        """Drop the commands of watching pages."""
        @functools.wraps(handler)
        def checked(data: Any = None) -> None:
            with clients_lock:
                pilot = next(iter(clients), None)
            if request.sid == pilot:  # type: ignore[attr-defined]
                handler(data)
        return checked

    def announce_roles() -> None:
        with clients_lock:
            sids = list(clients)
        for i, sid in enumerate(sids):
            socketio.emit("role", {"pilot": i == 0, "watchers": len(sids) - 1}, to=sid)

    def signed_in() -> bool:
        return access.configured and session.get("access") == access.generation

    def sign_in() -> None:
        session.permanent = True
        session["access"] = access.generation

    def posted(key: str = "code") -> str:
        return str((request.get_json(silent=True) or {}).get(key, ""))

    def refused(error: str, wait_s: float) -> Any:
        return {"error": error + (f": wait {wait_s:.0f} s" if wait_s else ""), "wait_s": wait_s}, 403

    def sign_out_on_new_code() -> None:
        """Close the connections opened with an older access code."""
        while True:
            socketio.sleep(CODE_CHECK_S)
            current = access.generation
            with clients_lock:
                stale = [sid for sid, generation in clients.items() if generation != current]
            for sid in stale:
                socketio.server.disconnect(sid, namespace="/")

    socketio.start_background_task(sign_out_on_new_code)

    @app.before_request
    def require_access() -> Any:
        """Everything but the page and its sign-in screen needs a session."""
        if request.endpoint not in OPEN_ENDPOINTS and not signed_in():
            return Response("sign in first", status=401)
        return None

    @app.route("/auth/state")
    def auth_state() -> Any:
        return {"configured": access.configured, "signed_in": signed_in(), "min_length": MIN_CODE_LENGTH}

    @app.route("/auth/setup", methods=["POST"])
    def auth_setup() -> Any:
        """The first visitor chooses the access code."""
        try:
            if not access.choose(posted()):
                return {"error": "an access code is already set: enter it"}, 409
        except ValueError as exc:
            return {"error": str(exc)}, 400
        sign_in()
        return {"ok": True}

    @app.route("/auth/login", methods=["POST"])
    def auth_login() -> Any:
        ok, wait_s = access.check(posted())
        if not ok:
            return refused("wrong code", wait_s)
        sign_in()
        return {"ok": True}

    @app.route("/auth/change", methods=["POST"])
    def auth_change() -> Any:
        """A new code, given the current one: every other browser must enter it."""
        ok, wait_s = access.check(posted("current"))
        if not ok:
            return refused("wrong current code", wait_s)
        try:
            access.set_code(posted())
        except ValueError as exc:
            return {"error": str(exc)}, 400
        sign_in()
        return {"ok": True}

    @app.route("/")
    def index() -> Any:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/favicon.ico")
    def favicon() -> Any:
        return send_file(FAVICON)

    @app.route("/photo")
    def photo() -> Any:
        path = None if camera is None else camera.photo(record_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_photo.jpg")
        if path is None:
            return Response("no camera image", status=404)
        return send_file(path, as_attachment=True)

    @app.route("/video_feed")
    def video_feed() -> Any:
        if camera is None:
            return Response("camera disabled", status=404)
        return Response(camera.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/recordings/<path:name>")
    def recording(name: str) -> Any:
        return send_from_directory(record_dir.resolve(), name)

    @app.route("/recordings/<name>", methods=["DELETE"])
    def delete_recording(name: str) -> Any:
        """A photo or video of the gallery, nothing else."""
        path = record_dir / name
        if path.parent != record_dir or not name.endswith(RECORDING_SUFFIXES) or not path.is_file():
            return Response("no such recording", status=404)
        path.unlink()
        return {"ok": True}

    @app.route("/gallery")
    def gallery() -> Any:
        """Photos and videos taken from the page, newest first."""
        files = [f for suffix in RECORDING_SUFFIXES
                 for f in sorted(record_dir.glob("*" + suffix), key=lambda f: f.name, reverse=True)]
        return {"files": [{"name": f.name, "kind": "photo" if f.suffix == ".jpg" else "video",
                           "size_mb": round(f.stat().st_size / 1e6, 1)} for f in files]}

    @app.route("/moods")
    def moods() -> Any:
        return {"moods": [{"name": m.name, "emoji": m.emoji} for m in led.MOODS]
                + [{"name": name, "emoji": "✨"} for name in teleop.extra_moves()]}

    @app.route("/servos")
    def servos() -> Any:
        return {"servos": [{"id": i, "joint": joint} for joint, i in SERVO_SETUP]}

    @app.route("/report")
    def report() -> Any:
        """No copy is kept on the robot."""
        name = f"chromapi_report_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        live = "$ live state (web app)\n" + json.dumps(snapshot(), indent=2, default=str) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            data = write_report(Path(directory) / name, use_bridge=False, extra=live).read_bytes()
        return send_file(io.BytesIO(data), mimetype="text/plain", as_attachment=True, download_name=name)

    @socketio.on("connect")
    def on_connect(auth: Any = None) -> Any:
        if not signed_in():
            return False
        with clients_lock:
            clients[request.sid] = session["access"]  # type: ignore[attr-defined]
        announce_roles()
        return None

    @socketio.on("disconnect")
    def on_disconnect(*_args: Any) -> None:
        with clients_lock:
            was_pilot = next(iter(clients), None) == request.sid  # type: ignore[attr-defined]
            clients.pop(request.sid, None)  # type: ignore[attr-defined]
        if was_pilot:
            web_input.release()
            teleop.listen_stop()
            announce_roles()
        elif clients:
            announce_roles()

    @socketio.on("axes")
    @pilot_only
    def on_axes(data: Any) -> None:
        web_input.set_axes(data or {})

    @socketio.on("action")
    @pilot_only
    def on_action(data: Any) -> None:
        teleop.request(str((data or {}).get("name", "")))

    @socketio.on("set_speed")
    @pilot_only
    def on_speed(data: Any) -> None:
        teleop.speed_scale = min(max(float((data or {}).get("value", 1.0)), MIN_SPEED_SCALE), teleop.max_speed_scale)

    @socketio.on("set_height")
    @pilot_only
    def on_height(data: Any) -> None:
        teleop.height = min(max(float((data or {}).get("mm", 0.0)) / 1000.0, HEIGHT_RANGE_M[0]), HEIGHT_RANGE_M[1])

    @socketio.on("talk")
    @pilot_only
    def on_talk(data: Any) -> None:
        if isinstance(data, (bytes, bytearray)):
            teleop.talk(bytes(data))

    @socketio.on("listen_start")
    @pilot_only
    def on_listen_start(data: Any = None) -> None:
        sid = request.sid  # type: ignore[attr-defined]
        teleop.listen_start(lambda pcm: socketio.emit("listen", pcm, to=sid))

    @socketio.on("listen_stop")
    @pilot_only
    def on_listen_stop(data: Any = None) -> None:
        teleop.listen_stop()

    @socketio.on("set_pad_tilt")
    @pilot_only
    def on_pad_tilt(data: Any) -> None:
        if teleop.pad is not None:
            teleop.pad.tilt_lean = bool((data or {}).get("on"))

    @socketio.on("set_torch")
    @pilot_only
    def on_torch(data: Any) -> None:
        teleop.ring.torch = bool((data or {}).get("on"))

    @socketio.on("set_ring_color")
    @pilot_only
    def on_ring_color(data: Any) -> None:
        """``{"rgb": [r, g, b]}`` (0-255), ``{"rainbow": true}``, or ``{}`` for the default."""
        data = data or {}
        rgb = data.get("rgb")
        if data.get("rainbow"):
            led.set_idle_rainbow()
        else:
            led.set_idle_color((int(rgb[0]), int(rgb[1]), int(rgb[2])) if isinstance(rgb, list) and len(rgb) == 3 else None)

    @socketio.on("set_fan")
    @pilot_only
    def on_fan(data: Any) -> None:
        if teleop.hardware:
            data = data or {}
            fan_service.request(None if data.get("auto") else float(data.get("percent", 100.0)))

    @socketio.on("camera_record")
    @pilot_only
    def on_camera_record(data: Any) -> None:
        if camera is None:
            return
        if (data or {}).get("on"):
            camera.start_recording(record_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_camera.avi")
        else:
            camera.stop_recording()

    return app, socketio


def _lan_ip() -> str:
    """This machine's LAN address (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def main() -> int:
    """Serve the web app until Ctrl-C (or the page's power-off button)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", action="store_true", help="run in MuJoCo instead of on the robot")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--record-dir", type=Path, default=Path.home() / "chromapi_recordings",
                        help="where the camera Record button saves its videos")
    parser.add_argument("--http", action="store_true",
                        help="plain HTTP instead of HTTPS (no phone microphone, traffic readable on the network)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    hardware = not args.sim

    # Heavy setup before connect(): it would starve the control loop
    robot = Chromapi(backend="hardware" if hardware else "mujoco", launch_viewer=not hardware)
    if isinstance(robot.backend, MuJoCoBackend):
        robot.backend.start_in_pose(robot.rest_pose())
    web_input = WebInput()
    teleop = Teleop(robot, web_input, hardware, GamepadControl())
    teleop.on_shutdown = lambda power: socketio.emit("goodbye", power)
    camera = None
    if hardware and not args.no_camera:
        from chromapi.web.camera import CameraStreamer

        camera = CameraStreamer(load_config().get("camera", {}), device=args.camera_index)
        camera.start()

    connected = robot.connect()
    if not connected:
        logger.error("Cannot open the serial port: retrying every %.0f s", RECONNECT_S)
    teleop.ring.start()
    certificate = None if args.http else tls_certificate()
    app, socketio = build_server(teleop, web_input, camera, args.record_dir,
                                 lambda: telemetry(robot, teleop, camera, connected), Access(),
                                 https=certificate is not None)
    threading.Thread(
        target=lambda: socketio.run(app, host=args.host, port=args.port, use_reloader=False,
                                    allow_unsafe_werkzeug=True, log_output=False, ssl_context=certificate),
        name="chromapi-web", daemon=True,
    ).start()
    scheme = "https" if certificate else "http"
    if args.host == "0.0.0.0":
        logger.info("Web app on %s://%s.local:%d (or %s://%s:%d)", scheme, socket.gethostname(), args.port,
                    scheme, _lan_ip(), args.port)
    else:
        logger.info("Web app on %s://%s:%d", scheme, args.host, args.port)

    next_telemetry = next_connect = time.monotonic()
    greeted = False
    try:
        while not teleop.shutting_down and robot.backend.is_alive():
            now = time.monotonic()
            if not connected and now >= next_connect:
                connected, next_connect = robot.connect(), now + RECONNECT_S
            if connected:
                if not greeted and robot.get_state() is not None:
                    teleop.resume_if_held()
                    # Threaded: the first sound loads PortAudio
                    threading.Thread(target=teleop.ring.play_mood, args=(teleop.startup_mood(),), name="chromapi-greeting",
                                     daemon=True).start()
                    greeted = True
                teleop.tick()
            if now >= next_telemetry:
                socketio.emit("telemetry", telemetry(robot, teleop, camera, connected))
                next_telemetry = now + TELEMETRY_PERIOD_S
            time.sleep(TICK_S)
    except KeyboardInterrupt:
        pass
    finally:
        if teleop.pad is not None:
            teleop.pad.close()
        if camera is not None:
            camera.stop()
        teleop.cancel_maintenance()
        if not teleop.shutting_down and robot.is_connected:  # during maintenance, the port is the tool's
            if teleop.awake:
                teleop._lie_down()
            teleop.ring.stop()
            robot.disconnect(disable_torque=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
