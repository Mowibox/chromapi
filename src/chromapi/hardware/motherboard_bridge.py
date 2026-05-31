"""Motherboard Bridge Module for Chromapi.

Provides the UART communication interface between the Raspberry Pi and the STM32 motherboard
using a custom protocol.
"""

import logging
import struct
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import serial

BRIDGE_SYNC_1 = 0x55
BRIDGE_SYNC_2 = 0xAA


class Command(IntEnum):
    """Available bridge commands."""

    PING_BRIDGE    = 0x01
    SET_POSITIONS  = 0x02
    STATE_FEEDBACK = 0x03
    PING_SERVO     = 0x04
    GET_SERVO_INFO = 0x05
    SET_SERVO_ID   = 0x06
    SET_LED_COLOR  = 0x07
    GET_POWER      = 0x08


class Response(IntEnum):
    """Possible bridge responses."""

    OK             = 0x80
    ERROR          = 0x81
    POWER_READING  = 0x82
    STATE_SNAPSHOT = 0x83


class BridgeClient:
    """Client class to interact with the STM32 Chromapi Motherboard."""

    def __init__(self, port: str = '/dev/ttyS0', baudrate: int = 115200, timeout: float = 1.0) -> None:
        """Initialize the BridgeClient with serial port parameters."""
        self.port: str = port
        self.baudrate: int = baudrate
        self.timeout: float = timeout
        self.serial: Optional[serial.Serial] = None
        self.logger: logging.Logger = logging.getLogger("BridgeClient")

    def connect(self) -> bool:
        """Attempt to open the serial port connection."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
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

    def _calc_crc(self, data: bytes) -> int:
        """Calculate the XOR checksum for the given byte array."""
        crc = 0
        for b in data:
            crc ^= b
        return crc

    def _send_frame(self, cmd: Command, payload: bytes = b'') -> None:
        """Wrap and send a command frame over UART."""
        if self.serial is None or not self.serial.is_open:
            raise ConnectionError("Serial port is not open.")

        length_byte = len(payload) + 1
        crc_data = bytes([length_byte, cmd]) + payload
        frame = bytes([BRIDGE_SYNC_1, BRIDGE_SYNC_2]) + crc_data + bytes([self._calc_crc(crc_data)])

        self.serial.write(frame)
        self.serial.flush()

    def _read_reply(self) -> Tuple[int, bytes]:
        """Read and parse an incoming reply frame from UART."""
        if self.serial is None:
            raise ConnectionError("Serial port is not initialized.")

        sync_state = 0
        while sync_state < 2:
            b = self.serial.read(1)
            if not b:
                raise TimeoutError("Timeout waiting for reply header.")
            if sync_state == 0 and b[0] == BRIDGE_SYNC_1:
                sync_state = 1
            elif sync_state == 1 and b[0] == BRIDGE_SYNC_2:
                sync_state = 2
            else:
                sync_state = 0

        length_byte = self.serial.read(1)
        if not length_byte:
            raise TimeoutError("Timeout reading LEN.")

        rest = self.serial.read(length_byte[0] + 1)
        if len(rest) != length_byte[0] + 1:
            raise TimeoutError("Truncated frame.")

        cmd = rest[0]
        payload = rest[1:-1]

        if rest[-1] != self._calc_crc(length_byte + rest[:-1]):
            raise ValueError(f"CRC mismatch. Received: {rest[-1]}, Computed: {self._calc_crc(length_byte + rest[:-1])}")

        return cmd, payload

    def ping(self) -> bool:
        """Send a ping command to verify STM32 connectivity."""
        if self.serial is None:
            return False
        self.serial.reset_input_buffer()
        self._send_frame(Command.PING_BRIDGE)
        try:
            cmd, _ = self._read_reply()
            return cmd == Response.OK
        except Exception as e:
            self.logger.warning(f"Ping failed: {e}")
            return False

    def set_led_color(self, r: int, g: int, b: int) -> bool:
        """Set the RGB LEDs color on the motherboard."""
        self._send_frame(Command.SET_LED_COLOR, bytes([r & 0xFF, g & 0xFF, b & 0xFF]))
        try:
            cmd, _ = self._read_reply()
            return cmd == Response.OK
        except Exception:
            return False

    def get_power(self) -> Optional[Tuple[float, float, float]]:
        """Retrieve voltage, current, and power from the INA226 sensor."""
        if self.serial is None:
            return None
        self.serial.reset_input_buffer()
        self._send_frame(Command.GET_POWER)
        try:
            cmd, payload = self._read_reply()
            if cmd == Response.POWER_READING and len(payload) == 12:
                bus_uV, curr_uA, power_uW = struct.unpack('<iii', payload)
                return (bus_uV / 1_000_000.0, curr_uA / 1_000_000.0, power_uW / 1_000_000.0)
            return None
        except Exception as e:
            self.logger.warning(f"get_power error: {e}")
            return None

    def set_positions(self, raw_steps: List[int]) -> bool:
        """Send target positions to all 12 servomotors."""
        if len(raw_steps) != 12:
            raise ValueError("Expected exactly 12 raw positions (pad with 0 if unused).")
        self._send_frame(Command.SET_POSITIONS, struct.pack('<12H', *raw_steps))
        try:
            cmd, _ = self._read_reply()
            return cmd == Response.OK
        except Exception as e:
            self.logger.warning(f"set_positions error: {e}")
            return False

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Retrieve the global hardware state snapshot."""
        if self.serial is None:
            return None
        self.serial.reset_input_buffer()
        self._send_frame(Command.STATE_FEEDBACK)
        try:
            cmd, payload = self._read_reply()
            if cmd != Response.STATE_SNAPSHOT or len(payload) != 121:
                self.logger.warning(f"Unexpected state snapshot size: {len(payload)} bytes")
                return None

            data = struct.unpack('<iii' + ('HhhBB' * 12) + 'hhhhhhB', payload)

            return {
                "power": {
                    "voltage_V": data[0] / 1_000_000.0,
                    "current_A": data[1] / 1_000_000.0,
                    "power_W":   data[2] / 1_000_000.0,
                },
                "servos": [
                    {
                        "id":     i + 1,
                        "pos":    data[3 + i * 5],
                        "speed":  data[4 + i * 5],
                        "load":   data[5 + i * 5],
                        "temp_C": data[6 + i * 5],
                        "volt_V": data[7 + i * 5] / 10.0,
                    }
                    for i in range(12)
                ],
                "imu": {
                    "acc_mps2": [data[63] / 100.0,  data[64] / 100.0,  data[65] / 100.0],
                    "gyro_rps": [data[66] / 1000.0, data[67] / 1000.0, data[68] / 1000.0],
                },
                "switches": {
                    "TL": bool(data[69] & 0x01),
                    "TR": bool(data[69] & 0x02),
                    "BL": bool(data[69] & 0x04),
                    "BR": bool(data[69] & 0x08),
                },
            }

        except Exception as e:
            self.logger.error(f"get_state error: {e}")
            return None

    def ping_servo(self, servo_id: int) -> bool:
        """Ping a specific servomotor on the bus."""
        if self.serial is None:
            return False
        self.serial.reset_input_buffer()
        self._send_frame(Command.PING_SERVO, bytes([servo_id & 0xFF]))
        try:
            cmd, _ = self._read_reply()
            return cmd == Response.OK
        except Exception:
            return False

    def set_servo_id(self, old_id: int, new_id: int) -> bool:
        """Change the ID of a Feetech servomotor."""
        if self.serial is None:
            return False
        self.serial.reset_input_buffer()
        self._send_frame(Command.SET_SERVO_ID, bytes([old_id & 0xFF, new_id & 0xFF]))
        try:
            cmd, _ = self._read_reply()
            return cmd == Response.OK
        except Exception as e:
            self.logger.warning(f"set_servo_id error: {e}")
            return False