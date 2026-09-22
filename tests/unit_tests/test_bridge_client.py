import pytest
import struct
from chromapi.hardware.motherboard_bridge import BridgeClient, Command, Response

class DummySerial:
    """Minimalist serial port."""
    def __init__(self) -> None:
        self.is_open: bool = True
        self.written_data: bytes = b''
        self.rx_buffer: list[bytes] = []
        self.reset_input_buffer_calls: int = 0

    def reset_input_buffer(self) -> None:
        self.reset_input_buffer_calls += 1
        
    def flush(self) -> None: 
        pass
        
    def close(self) -> None: 
        self.is_open = False

    def write(self, data: bytes) -> None:
        self.written_data = data

    def read(self, size: int = 1) -> bytes:
        if not self.rx_buffer:
            return b''
        
        chunk = self.rx_buffer[0][:size]
        self.rx_buffer[0] = self.rx_buffer[0][size:]

        if not self.rx_buffer[0]:
            self.rx_buffer.pop(0)
            
        return chunk

@pytest.fixture
def mock_serial(monkeypatch: pytest.MonkeyPatch) -> DummySerial:
    """Intercepts serial.Serial instantiation and injects DummySerial."""
    dummy = DummySerial()
    monkeypatch.setattr("serial.Serial", lambda *args, **kwargs: dummy)
    return dummy

@pytest.fixture
def bridge(mock_serial: DummySerial) -> BridgeClient:
    """Provides an isolated BridgeClient instance for each test."""
    client = BridgeClient(port='/dev/dummy', timeout=0.1)
    client.connect()
    return client

def test_calc_crc(bridge: BridgeClient) -> None:
    """Test CRC calculation (successive XOR)."""
    data = bytes([0x03, 0x01, 0xFF])
    assert bridge._calc_crc(data) == 0xFD

def test_send_frame_format(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    """Verifies that _send_frame generates the exact byte sequence expected."""
    bridge.set_led_color_all(10, 20, 30)
    
    crc_data = bytes([0x05, 0x07, 0x01, 10, 20, 30])
    expected_crc = bridge._calc_crc(crc_data)
    expected_frame = bytes([0x55, 0xAA]) + crc_data + bytes([expected_crc])
    
    assert mock_serial.written_data == expected_frame

def test_read_reply_timeout(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    """Verifies that a TimeoutError is raised if the serial port does not respond."""
    mock_serial.rx_buffer = []
    with pytest.raises(TimeoutError, match="Timeout waiting for reply header"):
        bridge._read_reply()

def test_read_reply_crc_mismatch(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    """Verifies the rejection of a corrupted frame (invalid CRC)."""
    mock_serial.rx_buffer = [b'\x55\xAA\x01\x80\x00']
    with pytest.raises(ValueError, match="CRC mismatch"):
        bridge._read_reply()

def test_set_positions_resets_input_buffer(bridge: BridgeClient, mock_serial: DummySerial) -> None:
    """Verifies that set_positions() calls reset_input_buffer() before sending the command."""
    mock_serial.rx_buffer = []  # no reply queued - set_positions() will time out, that's fine here
    calls_before = mock_serial.reset_input_buffer_calls  # connect() itself already made one call
    bridge.set_positions([0] * 12)
    assert mock_serial.reset_input_buffer_calls == calls_before + 1


def test_set_positions_decodes_state_snapshot(bridge: BridgeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that set_positions() correctly decodes a STATE_SNAPSHOT reply into a structured dict."""
    servo_fields = [100, 0, 0, 25, 60] * 12  # pos, speed, load, temp_C, volt_V(raw, ->6.0V)
    payload = struct.pack(
        '<iii' + ('HhhBB' * 12) + 'hhhhhhhhhhB',
        7400000, 1500000, 11100000,
        *servo_fields,
        0, 0, 981, 0, 0, 0, 32767, 0, 0, 0,  # acc(x,y,z), gyro(x,y,z), quat(w,x,y,z)
        0,  # switches_mask
    )
    monkeypatch.setattr(bridge, '_send_frame', lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, '_read_reply', lambda timeout=None: (Response.STATE_SNAPSHOT, payload))

    state = bridge.set_positions([0] * 12)

    assert state is not None
    assert state["power"]["voltage_V"] == 7.4
    assert state["servos"][0]["pos"] == 100
    assert state["servos"][0]["volt_V"] == 6.0
    assert state["imu"]["quat"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_set_positions_rejects_unexpected_reply(bridge: BridgeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that set_positions() returns None if the reply is not a STATE_SNAPSHOT."""
    monkeypatch.setattr(bridge, '_send_frame', lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, '_read_reply', lambda timeout=None: (Response.OK, b''))

    assert bridge.set_positions([0] * 12) is None


def test_get_power_parsing(bridge: BridgeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests struct parsing without relying on low-level _read_reply logic."""
    simulated_payload = struct.pack('<iii', 7400000, 1500000, 11100000)
    
    monkeypatch.setattr(bridge, '_send_frame', lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, '_read_reply', lambda: (Response.POWER_READING, simulated_payload))
    
    power = bridge.get_power()
    
    assert power is not None
    assert power[0] == 7.4   # Voltage in Volts
    assert power[1] == 1.5   # Current in Amperes
    assert power[2] == 11.1  # Power in Watts