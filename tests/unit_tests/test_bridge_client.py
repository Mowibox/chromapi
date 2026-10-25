# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the motherboard bridge client."""

import binascii
import struct
import threading
import time

import pytest

from chromapi.hardware.motherboard_bridge import BridgeClient, Command, Response


class DummySerial:
    """Byte-stream serial port: tests append to ``rx``, the client reads from it."""

    def __init__(self) -> None:
        self.is_open: bool = True
        self.written: list[bytes] = []
        self.rx = bytearray()

    @property
    def in_waiting(self) -> int:
        return len(self.rx)

    def reset_input_buffer(self) -> None:
        self.rx.clear()

    def close(self) -> None:
        self.is_open = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def read(self, size: int = 1) -> bytes:
        chunk = bytes(self.rx[:size])
        del self.rx[:size]
        return chunk


def _frame(cmd: int, payload: bytes = b'') -> bytes:
    body = bytes([len(payload) + 1, cmd]) + payload
    return b'\x55\xaa' + body + struct.pack('<H', binascii.crc_hqx(body, 0xFFFF))


def _snapshot_payload(seq=0, t_us=1000, gyro_corr=(0, 0, 0), servo_current=(0,) * 12, fresh=0x0FFF) -> bytes:
    """168-byte protocol v1 STATE_SNAPSHOT payload."""
    servos = []
    for i in range(12):
        servos += [100, 0, 0, 25, 60, servo_current[i]]  # pos, speed, load, temp_C, volt(raw->6.0V), current
    return struct.pack(
        '<BIH' + 'iii' + ('HhhBBh' * 12) + 'hhh' * 3 + 'hhhh' + 'BH',
        seq, t_us, 3100,
        7400000, 1500000, 11100000,
        *servos,
        0, 0, -981,  # acc
        12, -5, 3,   # raw gyro
        *gyro_corr,
        32767, 0, 0, 0,  # quat
        0,  # switches
        fresh,
    )


@pytest.fixture
def mock_serial(monkeypatch: pytest.MonkeyPatch) -> DummySerial:
    """Intercepts serial.Serial instantiation and injects DummySerial."""
    dummy = DummySerial()
    monkeypatch.setattr("serial.Serial", lambda *args, **kwargs: dummy)
    return dummy


@pytest.fixture
def bridge(mock_serial: DummySerial) -> BridgeClient:
    client = BridgeClient(port='/dev/dummy', timeout=0.02)
    client.connect()
    return client


def test_crc_matches_firmware_variant() -> None:
    """CRC-16/CCITT-FALSE check value - the firmware's crc16() must produce the same."""
    assert binascii.crc_hqx(b'123456789', 0xFFFF) == 0x29B1


def test_send_frame_format(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    mock_serial.rx += _frame(Response.OK)  # stale ACK from before the request: must be ignored
    assert not bridge.set_led_color_all(10, 20, 30)
    assert mock_serial.written[-1] == _frame(Command.SET_LED_COLOR, bytes([0x01, 10, 20, 30]))


def test_set_positions_is_fire_and_forget(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    assert bridge.set_positions(list(range(12))) is True
    assert mock_serial.written[-1] == _frame(Command.SET_POSITIONS, struct.pack('<12H', *range(12)))


def test_set_torque_frame(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    assert bridge.set_torque(True, [1, 12])
    assert mock_serial.written[-1] == _frame(Command.SET_TORQUE, struct.pack('<HH', 0x0801, 0x0801))
    assert bridge.set_torque(False)
    assert mock_serial.written[-1] == _frame(Command.SET_TORQUE, struct.pack('<HH', 0x0FFF, 0))
    assert bridge.set_torque(False, [3])
    assert mock_serial.written[-1] == _frame(Command.SET_TORQUE, struct.pack('<HH', 0x0004, 0))
    with pytest.raises(ValueError):
        bridge.set_torque(True, [13])


def test_sleep_feeds_link_watchdog(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    """A ping at least every 50 ms, well inside the firmware's 100 ms link timeout."""
    sent = []
    bridge._send_frame = lambda cmd, payload=b'': sent.append((time.monotonic(), cmd))
    t0 = time.monotonic()
    bridge.sleep(0.3)
    assert time.monotonic() - t0 >= 0.3
    times = [t0] + [t for t, cmd in sent if cmd == Command.PING_BRIDGE] + [time.monotonic()]
    assert len(times) >= 8 and max(b - a for a, b in zip(times, times[1:])) < 0.08


def test_command_reply_skips_streamed_snapshots(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    """An ACK interleaved with the stream is found, and the snapshot seen meanwhile is kept."""
    bridge._send_frame = lambda *a, **k: mock_serial.rx.extend(  # reply arrives after the request
        _frame(Response.STATE_SNAPSHOT, _snapshot_payload(seq=7)) + _frame(Response.OK))
    assert bridge.ping()
    state = bridge.get_state(timeout=0.0)
    assert state is not None and state["seq"] == 7


def test_get_state_returns_freshest(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    for seq in (1, 2, 3):
        mock_serial.rx += _frame(Response.STATE_SNAPSHOT, _snapshot_payload(seq=seq))
    state = bridge.get_state()
    assert state is not None and state["seq"] == 3
    assert bridge.get_state(timeout=0.01) is None  # nothing new since: never returns a frame twice


def test_resync_after_garbage_and_bad_crc(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    bad = bytearray(_frame(Response.STATE_SNAPSHOT, _snapshot_payload(seq=1)))
    bad[10] ^= 0xFF
    mock_serial.rx += b'\x00\x55\x13' + bad + _frame(Response.STATE_SNAPSHOT, _snapshot_payload(seq=2))
    state = bridge.get_state()
    assert state is not None and state["seq"] == 2


def test_get_state_timeout(bridge: BridgeClient) -> None:
    assert bridge.get_state(timeout=0.01) is None


def test_snapshot_decoding(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    currents = [100, -100] + [0] * 10
    mock_serial.rx += _frame(Response.STATE_SNAPSHOT, _snapshot_payload(
        t_us=123456, gyro_corr=(2, -1, 0), servo_current=currents, fresh=0b101))
    state = bridge.get_state()

    assert state is not None
    assert state["t_us"] == 123456 and state["cycle_us"] == 3100
    assert state["power"]["voltage_V"] == 7.4
    assert state["servos"][0]["pos"] == 100
    assert state["servos"][0]["volt_V"] == 6.0
    assert state["servos"][0]["current_A"] == pytest.approx(0.65)
    assert state["servos"][1]["current_A"] == pytest.approx(-0.65)
    assert [s["fresh"] for s in state["servos"][:3]] == [True, False, True]
    assert state["imu"]["gyro_rps"] == pytest.approx([0.012, -0.005, 0.003])
    assert state["imu"]["gyro_corr_rps"] == pytest.approx([0.002, -0.001, 0.0])
    assert state["imu"]["quat"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_read_servo_register(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    bridge._send_frame = lambda *a, **k: mock_serial.rx.extend(_frame(Response.SERVO_REG_VALUE, b'\x34\x12'))
    assert bridge.read_servo_register(1, 0x38, 2) == 0x1234


def test_command_error_and_timeout(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    bridge._send_frame = lambda *a, **k: mock_serial.rx.extend(_frame(Response.ERROR))
    assert bridge.ping_servo(3) is False
    bridge._send_frame = lambda *a, **k: None
    assert bridge.ping_servo(3) is False


def test_control_loop_and_caller_threads_share_the_port(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    """A control-loop thread polling snapshots must never swallow the reply another thread waits for."""
    def reply(data: bytes) -> None:  # the firmware: a command gets its ACK, every frame a snapshot
        if data[3] != Command.SET_POSITIONS:
            mock_serial.rx += _frame(Response.OK)
        mock_serial.rx += _frame(Response.STATE_SNAPSHOT, _snapshot_payload())

    def write(data: bytes) -> None:  # the reply comes back over the wire a little later
        threading.Timer(0.001, reply, (data,)).start()

    def read(size: int = 1) -> bytes:  # a real port blocks briefly when nothing has arrived
        if not mock_serial.rx:
            time.sleep(0.0005)
        chunk = bytes(mock_serial.rx[:size])
        del mock_serial.rx[:size]
        return chunk

    mock_serial.write = write  # type: ignore[method-assign]
    mock_serial.read = read  # type: ignore[method-assign]
    bridge.timeout = 0.2  # [s]
    stop = threading.Event()

    def control_loop() -> None:
        while not stop.is_set():
            bridge.set_positions([2048] * 12)
            bridge.get_state()

    loop = threading.Thread(target=control_loop)
    loop.start()
    try:
        acks = [bridge.set_led_color_all(10, 0, 10) for _ in range(50)]
    finally:
        stop.set()
        loop.join()
    assert all(acks)


def test_get_version(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    sent = []

    def firmware(cmd: int, payload: bytes = b'') -> None:
        sent.append(cmd)
        mock_serial.rx.extend(_frame(Response.VERSION, bytes([1, 1, 1, 0])))

    bridge._send_frame = firmware
    assert bridge.get_version() == (1, (1, 1, 0))
    assert sent == [Command.GET_VERSION]


def test_get_version_on_old_firmware(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    bridge._send_frame = lambda *a, **k: mock_serial.rx.extend(_frame(Response.ERROR))  # 1.0.0: unknown command
    assert bridge.get_version() is None
