# SPDX-License-Identifier: Apache-2.0
"""Motherboard Bridge Module for Chromapi.

Provides the UART communication interface between the Raspberry Pi and the STM32 motherboard
using a custom protocol.
"""

import binascii
import logging
import struct
import threading
import time
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import serial

BRIDGE_SYNC = b'\x55\xaa'
DEFAULT_BAUDRATE = 2_000_000

# Reply timeout for get_state(): a snapshot arrives every 5 ms, 50 ms = 10 missed cycles
_HOT_PATH_TIMEOUT_S = 0.05  # [s]
# Blocking granularity of a single serial read, set once on the port
_READ_POLL_S = 0.002  # [s]
# The STM32 releases every servo's torque after 100 ms without a valid frame: keepalive period
_LINK_KEEPALIVE_S = 0.05  # [s]

# seq, t_us, cycle_us, power, 12 x servo (pos, speed, load, temp, volt, current),
# imu acc/gyro/gyro_corr/quat, switches, servo fresh mask
_SNAPSHOT_FORMAT = '<BIH' + 'iii' + ('HhhBBh' * 12) + 'hhh' * 3 + 'hhhh' + 'BH'
_SNAPSHOT_LEN = struct.calcsize(_SNAPSHOT_FORMAT)  # 168 bytes
_SERVO_CURRENT_A_PER_UNIT = 0.0065  # STS3215 Present_Current resolution [A]


def _crc16(data: bytes) -> int:
    """Compute the CRC-16-CCITT checksum of the given data."""
    return binascii.crc_hqx(data, 0xFFFF)


#: Bridge protocol this SDK speaks - must match the firmware's BRIDGE_PROTOCOL_VERSION (bridge.h).
BRIDGE_PROTOCOL_VERSION = 1


class Command(IntEnum):
    """Available bridge commands."""

    PING_BRIDGE          = 0x01
    SET_POSITIONS        = 0x02
    SET_TORQUE           = 0x03
    PING_SERVO           = 0x04
    SET_SERVO_ID         = 0x06
    SET_LED_COLOR        = 0x07
    SET_LED_RING_BULK    = 0x08
    WRITE_SERVO_REGISTER = 0x0A
    READ_SERVO_REGISTER  = 0x0B
    GET_VERSION          = 0x0C

class Response(IntEnum):
    """Possible bridge responses."""

    OK              = 0x80
    ERROR           = 0x81
    STATE_SNAPSHOT  = 0x83
    SERVO_REG_VALUE = 0x84
    VERSION         = 0x85

class LedTarget(IntEnum):
    """LED target types for the SET_LED_COLOR command."""

    LED_RING_ALL    = 0x01
    LED_RING_SINGLE = 0x02

LED_RING_COUNT = 18

class BridgeClient:
    """Client class to interact with the STM32 Chromapi Motherboard."""

    def __init__(self, port: str = '/dev/ttyAMA0', baudrate: int = DEFAULT_BAUDRATE, timeout: float = 1.0) -> None:
        """Initialize the BridgeClient with serial port parameters.

        Args:
            port: Serial device of the RPi4 <-> STM32 UART bridge.
            baudrate: UART baud rate (must match the firmware).
            timeout: Default timeout for requests [s].

        """
        self.port: str = port
        self.baudrate: int = baudrate
        self.timeout: float = timeout  # [s]
        self.serial: Optional[serial.Serial] = None
        self.logger: logging.Logger = logging.getLogger("BridgeClient")
        self._rx = bytearray()
        self._latest: Optional[bytes] = None  # newest snapshot payload not yet returned
        # One frame exchange at a time: Chromapi's control loop and the caller's thread (wake_up,
        # rest, LEDs) share this port, and interleaved writes or RX parsing corrupt frames.
        self._io_lock = threading.Lock()

    def connect(self) -> bool:
        """Attempt to open the serial port connection."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=_READ_POLL_S)
            self.serial.reset_input_buffer()
            self.logger.info(f"Connected to bridge on {self.port} at {self.baudrate} baud.")
            return True
        except serial.SerialException as e:
            self.logger.error(f"Failed to open serial port: {e}")
            return False

    def close(self) -> None:
        """Close the serial port connection safely."""
        if self.serial and self.serial.is_open:
            self.serial.close()

    def _send_frame(self, cmd: int, payload: bytes = b'') -> None:
        """Wrap and send a command frame over UART."""
        if self.serial is None or not self.serial.is_open:
            raise ConnectionError("Serial port is not open.")
        body = bytes([len(payload) + 1, cmd]) + payload
        self.serial.write(BRIDGE_SYNC + body + struct.pack('<H', _crc16(body)))

    def _pop_frame(self) -> Optional[Tuple[int, bytes]]:
        """Extract the next valid frame from the RX buffer, or None if none is complete yet."""
        rx = self._rx
        while True:
            start = rx.find(BRIDGE_SYNC)
            if start < 0:
                del rx[:-1]  # keep a trailing 0x55, it may be the start of the next sync
                return None
            del rx[:start]
            if len(rx) < 3:
                return None
            n = rx[2]
            if len(rx) < n + 5:
                return None
            body = bytes(rx[2:3 + n])
            if n == 0 or rx[3 + n] | rx[4 + n] << 8 != _crc16(body):
                del rx[:1]  # bad frame: resync on the next sync pattern
                continue
            del rx[:n + 5]
            return body[1], body[2:]

    def _drain(self) -> None:
        """Parse everything already received without blocking, keeping the newest snapshot."""
        assert self.serial is not None
        self._rx += self.serial.read(self.serial.in_waiting)
        while (frame := self._pop_frame()) is not None:
            if frame[0] == Response.STATE_SNAPSHOT:
                self._latest = frame[1]

    def _wait_for(self, want: Tuple[int, ...], timeout: float) -> Tuple[int, bytes]:
        """Block until a frame whose cmd is in ``want`` arrives; snapshots seen meanwhile are kept."""
        assert self.serial is not None
        deadline = time.monotonic() + timeout  # [s]
        while True:
            frame = self._pop_frame()
            if frame is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Timeout waiting for a bridge frame.")
                self._rx += self.serial.read(max(1, self.serial.in_waiting))
            elif frame[0] in want:
                return frame
            elif frame[0] == Response.STATE_SNAPSHOT:
                self._latest = frame[1]

    def _transact(self, cmd: Command, payload: bytes = b'', want: Tuple[int, ...] = (Response.OK,)) -> Optional[Tuple[int, bytes]]:
        """Send a request/reply command; None on timeout/IO error. ERROR replies are returned too."""
        if self.serial is None or not self.serial.is_open:
            return None
        try:
            with self._io_lock:
                self._drain()  # drop stale replies so a late ACK can't answer this request
                self._send_frame(cmd, payload)
                return self._wait_for(want + (Response.ERROR,), self.timeout)
        except (TimeoutError, serial.SerialException) as e:
            self.logger.warning(f"{cmd.name} failed: {e}")
            return None

    def _ack(self, cmd: Command, payload: bytes = b'') -> bool:
        reply = self._transact(cmd, payload)
        return reply is not None and reply[0] == Response.OK

    def ping(self) -> bool:
        """Send a ping command to verify STM32 connectivity."""
        return self._ack(Command.PING_BRIDGE)

    def _led(self, cmd: Command, payload: bytes, wait: bool) -> bool:
        """``wait=False``: no ACK wait, so a lost ACK cannot hold the bus (and the control loop) for a timeout."""
        return self._ack(cmd, payload) if wait else self._send_nowait(cmd, payload)

    def set_led_color(self, index: int, r: int, g: int, b: int, wait: bool = True) -> bool:
        """Set a single ring LED (0-17)."""
        if not (0 <= index < LED_RING_COUNT):
            raise ValueError(f"index must be in [0, {LED_RING_COUNT - 1}]")
        payload = bytes([LedTarget.LED_RING_SINGLE, index & 0xFF, r & 0xFF, g & 0xFF, b & 0xFF])
        return self._led(Command.SET_LED_COLOR, payload, wait)

    def set_led_color_all(self, r: int, g: int, b: int, wait: bool = True) -> bool:
        """Set all 18 ring LEDs to a single color."""
        return self._led(Command.SET_LED_COLOR, bytes([LedTarget.LED_RING_ALL, r & 0xFF, g & 0xFF, b & 0xFF]), wait)

    def set_led_color_bulk(self, colors: List[Tuple[int, int, int]], wait: bool = True) -> bool:
        """Set all 18 ring LEDs individually in a single UART frame."""
        if len(colors) != LED_RING_COUNT:
            raise ValueError(f"Expected exactly {LED_RING_COUNT} (r,g,b) tuples.")
        payload = b''.join(bytes([r & 0xFF, g & 0xFF, b & 0xFF]) for r, g, b in colors)
        return self._led(Command.SET_LED_RING_BULK, payload, wait)

    @staticmethod
    def _decode_state_snapshot(payload: bytes) -> Dict[str, Any]:
        """Decode a STATE_SNAPSHOT payload.

        Units of the returned fields: ``t_us`` MCU time of the servo read [µs] (uint32, wraps),
        ``cycle_us`` bus cycle duration [µs]; ``voltage_V`` [V], ``current_A`` [A], ``power_W`` [W];
        per servo ``pos`` [step], ``speed`` [step.s⁻¹], ``load`` [0.1 %] of rated torque,
        ``temp_C`` [°C], ``volt_V`` [V], ``current_A`` [A] (magnitude), ``fresh`` (answered this
        cycle, else the values are from an earlier one); ``acc_mps2`` [m.s⁻²], ``gyro_rps`` (raw)
        and ``gyro_corr_rps`` (Mahony bias removed) [rad.s⁻¹], ``quat`` unit quaternion (w, x, y, z).
        """
        d = struct.unpack(_SNAPSHOT_FORMAT, payload)
        seq, t_us, cycle_us, bus_uV, curr_uA, power_uW = d[:6]
        servos, imu = d[6:78], d[78:91]
        switches, fresh = d[91], d[92]

        return {
            "seq": seq,
            "t_us": t_us,
            "cycle_us": cycle_us,
            "power": {
                "voltage_V": bus_uV / 1_000_000.0,
                "current_A": curr_uA / 1_000_000.0,
                "power_W":   power_uW / 1_000_000.0,
            },
            "servos": [
                {
                    "id":        i + 1,
                    "pos":       servos[i * 6],
                    "speed":     servos[i * 6 + 1],
                    "load":      servos[i * 6 + 2],
                    "temp_C":    servos[i * 6 + 3],
                    "volt_V":    servos[i * 6 + 4] / 10.0,
                    "current_A": servos[i * 6 + 5] * _SERVO_CURRENT_A_PER_UNIT,
                    "fresh":     bool(fresh >> i & 1),
                }
                for i in range(12)
            ],
            "imu": {
                "acc_mps2":      [v / 100.0 for v in imu[0:3]],
                "gyro_rps":      [v / 1000.0 for v in imu[3:6]],
                "gyro_corr_rps": [v / 1000.0 for v in imu[6:9]],
                "quat":          [v / 32767.0 for v in imu[9:13]],
            },
            "switches": {
                "TL": bool(switches & 0x01),
                "TR": bool(switches & 0x02),
                "BL": bool(switches & 0x04),
                "BR": bool(switches & 0x08),
            },
        }

    def _send_nowait(self, cmd: Command, payload: bytes) -> bool:
        """Send a command that gets no reply; False if the port is closed/broken."""
        try:
            with self._io_lock:
                self._send_frame(cmd, payload)
            return True
        except (ConnectionError, serial.SerialException) as e:
            self.logger.warning(f"{cmd.name} error: {e}")
            return False

    def sleep(self, duration: float) -> None:
        """``time.sleep()`` that keeps the STM32 link watchdog fed while torque is held.

        The firmware releases every servo after 100 ms without a valid frame. A script that holds
        torque while idle must wait with this instead of ``time.sleep()``. The pings expect no
        reply here: their ACKs are dropped by the next command's ``_drain()``.
        """
        end = time.monotonic() + duration  # [s]
        while (left := end - time.monotonic()) > 0:
            self._send_nowait(Command.PING_BRIDGE, b'')
            time.sleep(min(_LINK_KEEPALIVE_S, left))

    def set_positions(self, raw_steps: List[int]) -> bool:
        """Send target positions to all 12 servomotors."""
        if len(raw_steps) != 12:
            raise ValueError("Expected exactly 12 raw positions (pad with 0 if unused).")
        return self._send_nowait(Command.SET_POSITIONS, struct.pack('<12H', *raw_steps))

    def set_torque(self, enabled: bool, servo_ids: Optional[Iterable[int]] = None) -> bool:
        """Switch torque on/off for servo_ids (default: all 12) in one frame."""
        mask = 0
        for servo_id in (servo_ids if servo_ids is not None else range(1, 13)):
            if not 1 <= servo_id <= 12:
                raise ValueError(f"servo_id must be in [1, 12], got {servo_id}")
            mask |= 1 << (servo_id - 1)
        return self._send_nowait(Command.SET_TORQUE, struct.pack('<HH', mask, mask if enabled else 0))

    def get_state(self, timeout: float = _HOT_PATH_TIMEOUT_S) -> Optional[Dict[str, Any]]:
        """Retrieve the global hardware state snapshot."""
        if self.serial is None or not self.serial.is_open:
            return None
        try:
            with self._io_lock:
                self._drain()
                payload, self._latest = self._latest, None
                if payload is None:
                    _, payload = self._wait_for((Response.STATE_SNAPSHOT,), timeout)
        except (TimeoutError, serial.SerialException) as e:
            self.logger.error(f"get_state error: {e}")
            return None
        if len(payload) != _SNAPSHOT_LEN:
            self.logger.warning(f"Unexpected state snapshot size: {len(payload)} bytes (firmware/SDK protocol mismatch?)")
            return None
        return self._decode_state_snapshot(payload)

    def ping_servo(self, servo_id: int) -> bool:
        """Ping a specific servomotor on the bus."""
        return self._ack(Command.PING_SERVO, bytes([servo_id & 0xFF]))

    def set_servo_id(self, old_id: int, new_id: int) -> bool:
        """Change the ID of a Feetech servomotor."""
        return self._ack(Command.SET_SERVO_ID, bytes([old_id & 0xFF, new_id & 0xFF]))

    def write_servo_register(self, servo_id: int, reg: int, size: int, value: int) -> bool:
        """Write 1 or 2 bytes to a specific Feetech register via the STM32 bridge."""
        payload = bytes([servo_id & 0xFF, reg & 0xFF, size & 0xFF, value & 0xFF, (value >> 8) & 0xFF])
        return self._ack(Command.WRITE_SERVO_REGISTER, payload)

    def get_version(self) -> Optional[Tuple[int, Tuple[int, int, int]]]:
        """Return ``(protocol, (major, minor, patch))`` of the firmware, or None.

        None also means a firmware older than 2.0.0, which does not know this command.
        """
        reply = self._transact(Command.GET_VERSION, want=(Response.VERSION,))
        if reply is None or reply[0] != Response.VERSION or len(reply[1]) != 4:
            return None
        protocol, major, minor, patch = reply[1]
        return protocol, (major, minor, patch)

    def read_servo_register(self, servo_id: int, reg: int, size: int) -> Optional[int]:
        """Read 1 or 2 bytes from a specific Feetech register."""
        reply = self._transact(Command.READ_SERVO_REGISTER, bytes([servo_id & 0xFF, reg & 0xFF, size & 0xFF]),
                               want=(Response.SERVO_REG_VALUE,))
        if reply is None or reply[0] != Response.SERVO_REG_VALUE or len(reply[1]) != size or size not in (1, 2):
            return None
        return int.from_bytes(reply[1], 'little')
