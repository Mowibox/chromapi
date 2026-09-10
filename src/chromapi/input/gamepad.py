"""Raw Linux joystick gamepad input - :class:`GamepadInput`.

No external dependency: reads ``/dev/input/js*`` directly
via the kernel joystick API, instead of a gamepad library. This also means Bluetooth "just
works" with no gamepad-specific Bluetooth code here at all: once a controller is paired at the
OS level (steps below), it shows up as an ordinary ``/dev/input/js0``, indistinguishable from
a wired one.

Pairing a controller over Bluetooth (Raspberry Pi or any Linux host)
----------------------------------------------------------------------
1. Put the controller in pairing mode (hold its pair button until its LED flashes rapidly).
2. ``bluetoothctl``, then: ``power on``, ``agent on``, ``scan on`` (note the MAC address once
   the controller shows up), ``scan off``, ``pair <MAC>``, ``trust <MAC>``, ``connect <MAC>``.
3. Once paired *and trusted*, it reconnects automatically on power-on - no repeat pairing.
4. With the ``joystick`` package installed  (``sudo apt install joystick``), you can use
   ``jstest /dev/input/js0``: this shows live axis/button numbers - may be useful if your 
   controller doesn't match :class:`GamepadAxisMap`'s defaults below.
"""

from __future__ import annotations

import logging
import os
import select
import struct
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Set

from chromapi.input.input_source import InputSource, UserInput

logger = logging.getLogger(__name__)

_JS_EVENT = struct.Struct("=IhBB")
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80

#: Signed 16-bit axis full scale (raw joystick axis values are int16, +-32767).
_AXIS_FULL_SCALE = 32767.0


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

class GamepadInput(InputSource):
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
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="chromapi-gamepad", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader thread and close the device."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                logger.warning("GamepadInput.stop(): closing %s failed (already disconnected?)", self.device)
            self._fd = None

    def _loop(self) -> None:
        """Background thread loop: read events from the device and update internal state."""
        while not self._stop_event.is_set():
            assert self._fd is not None
            readable, _, _ = select.select([self._fd], [], [], 0.1)
            if not readable:
                continue
            try:
                data = os.read(self._fd, _JS_EVENT.size * 32)
            except OSError:
                logger.warning("Gamepad %s disconnected", self.device)
                return
            n_complete = len(data) - (len(data) % _JS_EVENT.size)
            for offset in range(0, n_complete, _JS_EVENT.size):
                _, value, ev_type, number = _JS_EVENT.unpack_from(data, offset)
                self._handle_event(ev_type, number, value)

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
