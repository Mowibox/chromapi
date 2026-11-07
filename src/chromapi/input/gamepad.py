# SPDX-License-Identifier: Apache-2.0
"""Raw Linux joystick gamepad input - :class:`GamepadInput`.

No external dependency: reads ``/dev/input/js*`` directly
via the kernel joystick API, instead of a gamepad library. This also means Bluetooth "just
works" with no gamepad-specific Bluetooth code here at all: once a controller is paired at the
OS level (steps below), it shows up as an ordinary ``/dev/input/js0``, indistinguishable from
a wired one.

Pairing a controller over Bluetooth (Raspberry Pi or any Linux host)
----------------------------------------------------------------------
``chromapi gamepad pair`` does steps 1-2. By hand:

1. Put the controller in pairing mode (hold its pair button until its LED flashes rapidly).
2. ``bluetoothctl``, then: ``power on``, ``agent on``, ``scan on`` (note the MAC address once
   the controller shows up), ``scan off``, ``pair <MAC>``, ``trust <MAC>``, ``connect <MAC>``.
3. Once paired *and trusted*, it reconnects automatically on power-on - no repeat pairing.
4. With the ``joystick`` package installed  (``sudo apt install joystick``), you can use
   ``jstest /dev/input/js0``: this shows live axis/button numbers - may be useful if your 
   controller doesn't match :class:`GamepadAxisMap`'s defaults below.
"""

from __future__ import annotations

import fcntl
import logging
import os
import select
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_JS_EVENT = struct.Struct("=IhBB")
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80

#: Signed 16-bit axis full scale (raw joystick axis values are int16, +-32767).
_AXIS_FULL_SCALE = 32767.0

# evdev: motion sensors and rumble, which the joystick API does not carry
_INPUT_EVENT = struct.Struct("@llHHi")  # struct input_event: timeval, type, code, value
_EV_ABS, _EV_FF, _FF_RUMBLE = 0x03, 0x15, 0x50
# struct ff_effect (64-bit) with its rumble member (strong, weak)
_FF_EFFECT = struct.Struct("=HhHHHHH2xHH28x")
_EVIOCSFF = 0x40304580  # _IOW('E', 0x80, struct ff_effect)
RUMBLE_MS = 150  # [ms] one pulse
RUMBLE_GAP_MS = 120  # [ms] before each pulse
RUMBLE_LED_REFRESH_S = 0.3  # [s] some pads turn the light bar off to rumble
ACCEL_SMOOTHING = 0.05  # accelerometer low-pass
READ_PERIOD_S = 0.01  # [s] batched reads: one wake-up per report starves the control loop


@dataclass
class UserInput:
    """One tick of decoded input, as human intents; the caller maps them onto ``Chromapi.walk()``."""

    forward: float = 0.0  # [-1, 1] positive forward
    lateral: float = 0.0  # [-1, 1] positive right
    turn: float = 0.0  # [-1, 1] positive left
    vertical: float = 0.0  # [-1, 1] positive up
    pressed: Set[str] = field(default_factory=set)  # one-shot actions
    held: Set[str] = field(default_factory=set)  # hold-to-repeat actions


@dataclass
class GamepadAxisMap:
    """Which raw joystick axis/button numbers feed :class:`UserInput`, and the deadzone.

    Verify your own controller's numbering with ``jstest`` (see the module docstring) if
    movement doesn't respond, or responds on the wrong stick/axis.
    """

    left_x: int = 0
    left_y: int = 1
    right_x: int = 3
    right_y: int = 4
    dpad_x: int = 6
    dpad_y: int = 7
    deadzone: float = 0.12

class GamepadInput:
    """Reads a gamepad via the raw Linux joystick API.

    Runs its own background thread (started by :meth:`start`) parsing events as they arrive;
    :meth:`read` just returns the latest call every tick from a teleop loop.
    """

    def __init__(
        self, device: str = "/dev/input/js0", axis_map: Optional[GamepadAxisMap] = None
    ) -> None:
        """Build the input source - does not open the device until :meth:`start`.

        Args:
            device: Path to the joystick device node.
            axis_map: See :class:`GamepadAxisMap`.

        """
        self.device = device
        self.axis_map = axis_map or GamepadAxisMap()
        self._fd: Optional[int] = None
        self._axis_state: Dict[int, float] = {}
        self._pressed: Set[str] = set()
        self._held: Set[str] = set()
        self._hat_state: Dict[int, Optional[str]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._motion_fd: Optional[int] = None
        self._motion_path: Optional[str] = None
        self.motion = False  # read the motion sensors
        self._accel: List[Optional[float]] = [None, None, None]  # low-passed, raw units
        self._ff_fd: Optional[int] = None
        self._ff_id = -1
        self._led_files: List[Path] = []
        self._led_max = 255
        self._led_default: Optional[List[str]] = None
        self._rgb: Optional[Tuple[int, int, int]] = None
        self._led_refresh_at = float("inf")  # [s]
        self._battery: Optional[Path] = None

    def start(self) -> None:
        """Open the device and start the background reader thread."""
        try:
            self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No gamepad found at {self.device!r}. Is one connected/paired? Check with "
                f"'ls /dev/input/js*' - see chromapi.input.gamepad's module docstring for "
                f"Bluetooth pairing steps if nothing shows up there at all."
            ) from exc
        except PermissionError as exc:
            raise PermissionError(
                f"Found {self.device!r} but can't open it (permission denied) - check that "
                f"your user is in the 'input' group ('groups' to check, then 'sudo usermod "
                f"-aG input $USER' and re-login if not)."
            ) from exc
        self._find_extras()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="chromapi-gamepad", daemon=True
        )
        self._thread.start()

    def _find_extras(self) -> None:
        """Motion sensors, rumble, light bar and battery, when the pad has them."""
        try:
            pad = Path("/sys/class/input", Path(self.device).name, "device").resolve()
            hid = pad.parent.parent
            event = next(pad.glob("event*"), None)
            motion = next((next(d.glob("event*"), None) for d in hid.glob("input/input*")
                           if "Motion Sensors" in (d / "name").read_text()), None)
        except OSError:
            return
        if motion is not None:
            self._motion_path = f"/dev/input/{motion.name}"
        if event is not None:
            try:
                self._ff_fd = os.open(f"/dev/input/{event.name}", os.O_RDWR | os.O_NONBLOCK)
                effect = bytearray(_FF_EFFECT.pack(_FF_RUMBLE, -1, 0, 0, 0, RUMBLE_MS, RUMBLE_GAP_MS, 0xC000, 0x8000))
                fcntl.ioctl(self._ff_fd, _EVIOCSFF, effect, True)
                self._ff_id = _FF_EFFECT.unpack(effect)[1]
            except OSError:  # no rumble
                self._close_ff()
        rgb = next(hid.glob("leds/*:rgb:indicator"), None)
        triplet = [next(hid.glob(f"leds/*:{color}"), None) for color in ("red", "green", "blue")]
        leds: List[Path] = [rgb] if rgb is not None else [led for led in triplet if led is not None]
        if rgb is None and len(leds) != 3:
            leds = []
        try:
            self._led_max = int((leds[0] / "max_brightness").read_text()) if leds else 255
        except (OSError, ValueError):
            leds = []
        self._led_files = [rgb / "brightness", rgb / "multi_intensity"] if rgb is not None else [
            led / "brightness" for led in leds]
        self._battery = next(hid.glob("power_supply/*"), None)

    def _close_ff(self) -> None:
        if self._ff_fd is not None:
            os.close(self._ff_fd)
        self._ff_fd = None

    def tilt(self) -> Optional[Tuple[float, float, float]]:
        """Gravity in the pad's frame (raw units), None without motion sensors."""
        with self._lock:
            x, y, z = self._accel
        return None if x is None or y is None or z is None else (x, y, z)

    def rumble(self, pulses: int = 1) -> None:
        """Buzz ``pulses`` short pulses."""
        if self._ff_fd is None:
            return
        try:
            os.write(self._ff_fd, _INPUT_EVENT.pack(0, 0, _EV_FF, self._ff_id, pulses))
        except OSError:
            self._close_ff()
            return
        self._led_refresh_at = (time.monotonic() + pulses * (RUMBLE_MS + RUMBLE_GAP_MS) / 1000.0
                                + RUMBLE_LED_REFRESH_S)

    def set_color(self, rgb: Tuple[int, int, int]) -> None:
        """Light bar colour (needs ``deploy/99-chromapi-gamepad.rules``); :meth:`stop` restores the driver's."""
        self._rgb = rgb
        self._write_color(rgb)

    def _write_color(self, rgb: Tuple[int, int, int]) -> None:
        if not self._led_files:
            return
        levels = [str(round(min(max(c, 0), 255) * self._led_max / 255)) for c in rgb]
        values = [str(self._led_max), " ".join(levels)] if len(self._led_files) == 2 else levels
        try:
            if self._led_default is None:
                self._led_default = [path.read_text().strip() for path in self._led_files]
            for path, value in zip(self._led_files, values):
                path.write_text(value)
        except OSError as exc:
            logger.warning("Gamepad light bar: %s (install deploy/99-chromapi-gamepad.rules)", exc)
            self._led_files = []

    def battery(self) -> Optional[Tuple[int, bool]]:
        """(percent, charging) of the pad's battery, None if unknown."""
        if self._battery is None:
            return None
        try:
            percent = int((self._battery / "capacity").read_text())
            status = (self._battery / "status").read_text().strip()
        except (OSError, ValueError):
            return None
        return percent, status in ("Charging", "Full")

    def stop(self) -> None:
        """Stop the reader thread, restore the light bar, close the devices."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._led_default is not None:
            try:
                for path, value in zip(self._led_files, self._led_default):
                    path.write_text(value)
            except OSError:
                pass
        self._led_default = None
        self._close_ff()
        if self._motion_fd is not None:
            os.close(self._motion_fd)
            self._motion_fd = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                logger.warning("GamepadInput.stop(): closing %s failed (already disconnected?)", self.device)
            self._fd = None

    @property
    def connected(self) -> bool:
        """The reader thread runs (it ends when the device goes away)."""
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        """Background thread loop: read events from the device and update internal state."""
        while not self._stop_event.wait(READ_PERIOD_S):
            assert self._fd is not None
            self._open_motion(self.motion)
            fds = [self._fd] if self._motion_fd is None else [self._fd, self._motion_fd]
            readable, _, _ = select.select(fds, [], [], 0.1)
            if time.monotonic() >= self._led_refresh_at:
                self._refresh_color()
            if self._motion_fd in readable:
                self._read_motion()
            if self._fd not in readable:
                continue
            try:
                data = os.read(self._fd, _JS_EVENT.size * 256)
            except OSError:
                logger.warning("Gamepad %s disconnected", self.device)
                return
            n_complete = len(data) - (len(data) % _JS_EVENT.size)
            for offset in range(0, n_complete, _JS_EVENT.size):
                _, value, ev_type, number = _JS_EVENT.unpack_from(data, offset)
                self._handle_event(ev_type, number, value)

    def _open_motion(self, wanted: bool) -> None:
        """Open or close the motion sensors, from the reader thread."""
        if wanted == (self._motion_fd is not None) or self._motion_path is None:
            return
        if wanted:
            try:
                self._motion_fd = os.open(self._motion_path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as exc:
                logger.warning("Gamepad motion sensors: %s", exc)
                self._motion_path = None
            return
        os.close(self._motion_fd)  # type: ignore[arg-type]
        self._motion_fd = None
        with self._lock:
            self._accel = [None, None, None]

    def _refresh_color(self) -> None:
        """Send the colour again (the driver skips unchanged values)."""
        self._led_refresh_at = float("inf")
        if self._rgb is not None:
            r, g, b = self._rgb
            self._write_color((r + 1 if r < 255 else r - 1, g, b))
            self._write_color(self._rgb)

    def _read_motion(self) -> None:
        assert self._motion_fd is not None
        try:
            data = os.read(self._motion_fd, _INPUT_EVENT.size * 1024)
        except OSError:
            return
        for offset in range(0, len(data) - _INPUT_EVENT.size + 1, _INPUT_EVENT.size):
            _, _, ev_type, code, value = _INPUT_EVENT.unpack_from(data, offset)
            if ev_type == _EV_ABS and code < 3:  # accelerometer
                with self._lock:
                    last = self._accel[code]
                    self._accel[code] = float(value) if last is None else last + ACCEL_SMOOTHING * (value - last)

    def _handle_event(self, ev_type: int, number: int, value: int) -> None:
        """Update internal state from one decoded event."""
        kind = ev_type & ~_JS_EVENT_INIT
        is_init = bool(ev_type & _JS_EVENT_INIT)
        if kind == _JS_EVENT_AXIS and number == self.axis_map.dpad_x:
            self._update_hat(number, value, negative="dpad_left", positive="dpad_right")
        elif kind == _JS_EVENT_AXIS and number == self.axis_map.dpad_y:
            self._update_hat(number, value, negative="dpad_up", positive="dpad_down")
        elif kind == _JS_EVENT_AXIS:
            normalized = value / _AXIS_FULL_SCALE
            if abs(normalized) < self.axis_map.deadzone:
                normalized = 0.0
            with self._lock:
                self._axis_state[number] = normalized
        elif kind == _JS_EVENT_BUTTON:
            name = str(number)
            with self._lock:
                if value == 1:
                    self._held.add(name)
                    if not is_init:
                        self._pressed.add(name)
                else:
                    self._held.discard(name)

    def _update_hat(self, axis_number: int, value: int, negative: str, positive: str) -> None:
        """Decode one D-pad hat axis reading into a held name (or none)."""
        if value < -_AXIS_FULL_SCALE / 2:
            current: Optional[str] = negative
        elif value > _AXIS_FULL_SCALE / 2:
            current = positive
        else:
            current = None
        with self._lock:
            previous = self._hat_state.get(axis_number)
            if previous is not None:
                self._held.discard(previous)
            if current is not None:
                self._held.add(current)
            self._hat_state[axis_number] = current

    def read(self) -> UserInput:
        """Return the latest decoded input - non-blocking."""
        with self._lock:
            axes = dict(self._axis_state)
            pressed = set(self._pressed)
            held = set(self._held)
            self._pressed.clear()
        return UserInput(
            forward=-axes.get(self.axis_map.left_y, 0.0),
            lateral=axes.get(self.axis_map.left_x, 0.0),
            turn=-axes.get(self.axis_map.right_x, 0.0),
            vertical=-axes.get(self.axis_map.right_y, 0.0),
            pressed=pressed,
            held=held,
        )
