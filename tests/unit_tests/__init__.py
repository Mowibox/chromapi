import struct
import pytest
import serial
from unittest.mock import MagicMock, patch

from chromapi.hardware.motherboard_bridge import (
    BridgeClient, Command, Response, BRIDGE_SYNC_1, BRIDGE_SYNC_2
)


# =============================================================================
# Helpers
# =============================================================================

def build_frame(cmd: int, payload: bytes = b'') -> bytes:
    """Build a valid protocol frame as the STM32 would send it."""
    length_byte = len(payload) + 1
    crc_data = bytes([length_byte, cmd]) + payload
    crc = 0
    for b in crc_data:
        crc ^= b
    return bytes([BRIDGE_SYNC_1, BRIDGE_SYNC_2]) + crc_data + bytes([crc])


def make_reader(data: bytes):
    """
    Returns a read(n) callable that consumes bytes from a fixed buffer.
    Correctly handles the mixed read(1) / read(n) pattern in _read_reply.
    Returns b'' once the buffer is exhausted (simulates timeout).
    """
    buf = bytearray(data)
    pos = 0

    def read(n: int) -> bytes:
        nonlocal pos
        chunk = bytes(buf[pos:pos + n])
        pos += n
        return chunk

    return read


def make_state_payload(
    voltage_uV: int = 11_500_000,
    current_uA: int = 1_200_000,
    servo_pos: int = 2048,
    servo_speed: int = 0,
    servo_load: int = 100,
    servo_temp: int = 35,
    servo_volt: int = 74,   # raw value, divided by 10 → 7.4 V
    acc: tuple = (100, -200, 9800),
    gyro: tuple = (10, -5, 0),
    switches: int = 0b0101,
) -> bytes:
    """Build a valid 117-byte STATE_SNAPSHOT payload."""
    power  = struct.pack('<ii', voltage_uV, current_uA)
    servos = struct.pack('<HhhBB', servo_pos, servo_speed, servo_load, servo_temp, servo_volt) * 12
    imu    = struct.pack('<hhhhhh', *acc, *gyro)
    sw     = struct.pack('<B', switches)
    payload = power + servos + imu + sw
    assert len(payload) == 117, f"Bad payload size: {len(payload)}"
    return payload


# =============================================================================
# Fixture
# =============================================================================

@pytest.fixture
def bridge():
    """BridgeClient with a mocked, open serial port."""
    client = BridgeClient(port='/dev/ttyS0', baudrate=115200)
    client.serial = MagicMock(spec=serial.Serial)
    client.serial.is_open = True
    return client


# =============================================================================
# _calc_crc
# =============================================================================

class TestCalcCrc:
    def test_empty_returns_zero(self, bridge):
        assert bridge._calc_crc(b'') == 0

    def test_single_byte(self, bridge):
        assert bridge._calc_crc(b'\xAB') == 0xAB

    def test_identical_bytes_cancel(self, bridge):
        assert bridge._calc_crc(b'\xFF\xFF') == 0

    def test_ping_crc_data(self, bridge):
        # PING: crc_data = [LEN=0x01, CMD=0x01] → 0x01 ^ 0x01 = 0x00
        assert bridge._calc_crc(bytes([0x01, 0x01])) == 0x00

    def test_multi_byte_sequence(self, bridge):
        data = bytes([0x04, 0x07, 0xFF, 0x00, 0x80])
        expected = 0x04 ^ 0x07 ^ 0xFF ^ 0x00 ^ 0x80
        assert bridge._calc_crc(data) == expected


# =============================================================================
# _send_frame
# =============================================================================

class TestSendFrame:
    def test_ping_exact_bytes(self, bridge):
        # LEN=1, CMD=0x01, CRC = 1^1 = 0
        bridge._send_frame(Command.PING_BRIDGE)
        bridge.serial.write.assert_called_once_with(bytes([0x55, 0xAA, 0x01, 0x01, 0x00]))

    def test_frame_structure_with_payload(self, bridge):
        bridge._send_frame(Command.SET_LED_COLOR, bytes([255, 0, 128]))
        written = bridge.serial.write.call_args[0][0]

        assert written[0:2] == bytes([BRIDGE_SYNC_1, BRIDGE_SYNC_2])
        assert written[2] == 4                      # LEN = 3 payload + 1 CMD
        assert written[3] == Command.SET_LED_COLOR
        assert written[4:7] == bytes([255, 0, 128])
        assert written[7] == 4 ^ Command.SET_LED_COLOR ^ 255 ^ 0 ^ 128  # CRC

    def test_flush_is_called(self, bridge):
        bridge._send_frame(Command.PING_BRIDGE)
        bridge.serial.flush.assert_called_once()

    def test_raises_if_serial_is_none(self):
        client = BridgeClient()
        client.serial = None
        with pytest.raises(ConnectionError, match="not open"):
            client._send_frame(Command.PING_BRIDGE)

    def test_raises_if_port_closed(self, bridge):
        bridge.serial.is_open = False
        with pytest.raises(ConnectionError, match="not open"):
            bridge._send_frame(Command.PING_BRIDGE)


# =============================================================================
# _read_reply
# =============================================================================

class TestReadReply:
    def test_valid_ack_no_payload(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        cmd, payload = bridge._read_reply()
        assert cmd == Response.OK
        assert payload == b''

    def test_valid_reply_with_payload(self, bridge):
        data = struct.pack('<ii', 12_000_000, 500_000)
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.POWER_READING, data)
        )
        cmd, payload = bridge._read_reply()
        assert cmd == Response.POWER_READING
        assert payload == data

    def test_timeout_on_sync_header(self, bridge):
        bridge.serial.read.return_value = b''
        with pytest.raises(TimeoutError, match="header"):
            bridge._read_reply()

    def test_timeout_on_length_byte(self, bridge):
        frame = build_frame(Response.OK)
        # Deliver only SYNC1 + SYNC2, then nothing
        bridge.serial.read.side_effect = make_reader(frame[:2])
        with pytest.raises(TimeoutError, match="LEN"):
            bridge._read_reply()

    def test_truncated_frame(self, bridge):
        frame = build_frame(Response.OK)[:-1]  # strip last byte
        bridge.serial.read.side_effect = make_reader(frame)
        with pytest.raises(TimeoutError, match="Truncated"):
            bridge._read_reply()

    def test_crc_mismatch(self, bridge):
        frame = bytearray(build_frame(Response.OK))
        frame[-1] ^= 0xFF  # corrupt CRC
        bridge.serial.read.side_effect = make_reader(bytes(frame))
        with pytest.raises(ValueError, match="CRC mismatch"):
            bridge._read_reply()

    def test_resync_on_leading_noise(self, bridge):
        """Garbage bytes before a valid frame must be transparently skipped."""
        noise = bytes([0xFF, 0x00, 0x11])
        frame = build_frame(Response.OK)
        bridge.serial.read.side_effect = make_reader(noise + frame)
        cmd, _ = bridge._read_reply()
        assert cmd == Response.OK


# =============================================================================
# connect / close
# =============================================================================

class TestConnect:
    def test_success(self):
        with patch('chromapi.hardware.motherboard_bridge.serial.Serial') as mock_cls:
            client = BridgeClient(port='/dev/ttyS0', baudrate=115200)
            assert client.connect() is True
            mock_cls.assert_called_once_with('/dev/ttyS0', 115200, timeout=1.0)
            mock_cls.return_value.reset_input_buffer.assert_called_once()

    def test_serial_exception_returns_false(self):
        with patch(
            'chromapi.hardware.motherboard_bridge.serial.Serial',
            side_effect=serial.SerialException("port busy"),
        ):
            client = BridgeClient()
            assert client.connect() is False

    def test_close_when_open(self, bridge):
        bridge.close()
        bridge.serial.close.assert_called_once()

    def test_close_when_already_closed(self, bridge):
        bridge.serial.is_open = False
        bridge.close()
        bridge.serial.close.assert_not_called()


# =============================================================================
# ping
# =============================================================================

class TestPing:
    def test_ok(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        assert bridge.ping() is True

    def test_error_response(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.ERROR))
        assert bridge.ping() is False

    def test_timeout(self, bridge):
        bridge.serial.read.return_value = b''
        assert bridge.ping() is False

    def test_sends_correct_command(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        bridge.ping()
        written = bridge.serial.write.call_args[0][0]
        assert written[3] == Command.PING_BRIDGE


# =============================================================================
# set_led_color
# =============================================================================

class TestSetLedColor:
    def test_ok(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        assert bridge.set_led_color(255, 128, 0) is True

    def test_timeout(self, bridge):
        bridge.serial.read.return_value = b''
        assert bridge.set_led_color(0, 0, 0) is False

    def test_values_clamped_to_byte(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        bridge.set_led_color(256, 300, -1)
        written = bridge.serial.write.call_args[0][0]
        assert written[4] == 0    # 256 & 0xFF
        assert written[5] == 44   # 300 & 0xFF
        assert written[6] == 255  # -1  & 0xFF


# =============================================================================
# get_power
# =============================================================================

class TestGetPower:
    def _frame(self, voltage_uV: int, current_uA: int) -> bytes:
        return build_frame(Response.POWER_READING, struct.pack('<ii', voltage_uV, current_uA))

    def test_nominal_values(self, bridge):
        bridge.serial.read.side_effect = make_reader(self._frame(12_000_000, 500_000))
        v, a = bridge.get_power()
        assert v == pytest.approx(12.0)
        assert a == pytest.approx(0.5)

    def test_zero_values(self, bridge):
        bridge.serial.read.side_effect = make_reader(self._frame(0, 0))
        assert bridge.get_power() == (0.0, 0.0)

    def test_wrong_response_cmd(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.OK, struct.pack('<ii', 12_000_000, 0))
        )
        assert bridge.get_power() is None

    def test_wrong_payload_size(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.POWER_READING, b'\x00' * 4)
        )
        assert bridge.get_power() is None

    def test_timeout(self, bridge):
        bridge.serial.read.return_value = b''
        assert bridge.get_power() is None


# =============================================================================
# set_positions
# =============================================================================

class TestSetPositions:
    def test_ok(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        assert bridge.set_positions([2048] * 12) is True

    def test_timeout(self, bridge):
        bridge.serial.read.return_value = b''
        assert bridge.set_positions([2048] * 12) is False

    def test_too_few_positions(self, bridge):
        with pytest.raises(ValueError, match="12"):
            bridge.set_positions([2048] * 6)

    def test_too_many_positions(self, bridge):
        with pytest.raises(ValueError, match="12"):
            bridge.set_positions([2048] * 13)

    def test_payload_encoding(self, bridge):
        positions = list(range(12))
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        bridge.set_positions(positions)
        written = bridge.serial.write.call_args[0][0]
        payload = written[4:-1]  # skip SYNC1, SYNC2, LEN, CMD; strip CRC
        assert struct.unpack('<12H', payload) == tuple(positions)

    def test_mechanical_center(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        bridge.set_positions([2048] * 12)
        written = bridge.serial.write.call_args[0][0]
        assert all(v == 2048 for v in struct.unpack('<12H', written[4:-1]))


# =============================================================================
# get_state
# =============================================================================

class TestGetState:
    def test_power_fields(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload())
        )
        state = bridge.get_state()
        assert state["power"]["voltage_V"] == pytest.approx(11.5)
        assert state["power"]["current_A"] == pytest.approx(1.2)

    def test_servo_count(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload())
        )
        assert len(bridge.get_state()["servos"]) == 12

    def test_servo_ids_are_sequential(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload())
        )
        ids = [s["id"] for s in bridge.get_state()["servos"]]
        assert ids == list(range(1, 13))

    def test_servo_fields(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload())
        )
        s = bridge.get_state()["servos"][0]
        assert s["pos"]    == 2048
        assert s["speed"]  == 0
        assert s["load"]   == 100
        assert s["temp_C"] == 35
        assert s["volt_V"] == pytest.approx(7.4)

    def test_imu_fields(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload())
        )
        imu = bridge.get_state()["imu"]
        assert imu["acc"]  == [100, -200, 9800]
        assert imu["gyro"] == [10, -5, 0]

    def test_switches_partial(self, bridge):
        # 0b0101 → TL=True, TR=False, BL=True, BR=False
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload(switches=0b0101))
        )
        sw = bridge.get_state()["switches"]
        assert sw == {"TL": True, "TR": False, "BL": True, "BR": False}

    def test_switches_all_off(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload(switches=0x00))
        )
        assert not any(bridge.get_state()["switches"].values())

    def test_switches_all_on(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, make_state_payload(switches=0x0F))
        )
        assert all(bridge.get_state()["switches"].values())

    def test_wrong_response_cmd(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.OK, make_state_payload())
        )
        assert bridge.get_state() is None

    def test_wrong_payload_size(self, bridge):
        bridge.serial.read.side_effect = make_reader(
            build_frame(Response.STATE_SNAPSHOT, b'\x00' * 50)
        )
        assert bridge.get_state() is None

    def test_timeout(self, bridge):
        bridge.serial.read.return_value = b''
        assert bridge.get_state() is None


# =============================================================================
# ping_servo
# =============================================================================

class TestPingServo:
    def test_servo_present(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        assert bridge.ping_servo(3) is True

    def test_servo_absent(self, bridge):
        bridge.serial.read.return_value = b''
        assert bridge.ping_servo(99) is False

    def test_correct_id_in_payload(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        bridge.ping_servo(7)
        written = bridge.serial.write.call_args[0][0]
        assert written[4] == 7

    def test_broadcast_id(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        bridge.ping_servo(255)
        written = bridge.serial.write.call_args[0][0]
        assert written[4] == 255


# =============================================================================
# set_servo_id
# =============================================================================

class TestSetServoId:
    def test_ok(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        assert bridge.set_servo_id(254, 5) is True

    def test_timeout(self, bridge):
        bridge.serial.read.return_value = b''
        assert bridge.set_servo_id(254, 5) is False

    def test_error_response(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.ERROR))
        assert bridge.set_servo_id(1, 2) is False

    def test_payload_contains_both_ids(self, bridge):
        bridge.serial.read.side_effect = make_reader(build_frame(Response.OK))
        bridge.set_servo_id(old_id=3, new_id=7)
        written = bridge.serial.write.call_args[0][0]
        assert written[4] == 3  # old_id
        assert written[5] == 7  # new_id