# SPDX-License-Identifier: Apache-2.0
"""Unit tests for chromapi.input.gamepad - event parsing only, no real device needed."""

import pytest

from chromapi.input.gamepad import (
    _JS_EVENT_AXIS,
    _JS_EVENT_BUTTON,
    _JS_EVENT_INIT,
    GamepadInput,
)


@pytest.fixture
def gamepad() -> GamepadInput:
    """Provide a GamepadInput with no device opened - _handle_event is tested directly."""
    return GamepadInput()


def test_left_stick_up_reads_as_positive_forward(gamepad: GamepadInput) -> None:
    """Raw axis 1 (left Y) at full "up" (-32767 on a typical controller) means forward=+1."""
    gamepad._handle_event(_JS_EVENT_AXIS, 1, -32767)
    assert gamepad.read().forward == pytest.approx(1.0, abs=0.01)


def test_left_stick_right_reads_as_positive_lateral(gamepad: GamepadInput) -> None:
    """Raw axis 0 (left X) at full "right" (+32767) means lateral=+1."""
    gamepad._handle_event(_JS_EVENT_AXIS, 0, 32767)
    assert gamepad.read().lateral == pytest.approx(1.0, abs=0.01)


def test_right_stick_right_reads_as_negative_turn(gamepad: GamepadInput) -> None:
    """Turn follows Chromapi.walk()'s wz convention (positive = left) - see UserInput."""
    gamepad._handle_event(_JS_EVENT_AXIS, 3, 32767)
    assert gamepad.read().turn == pytest.approx(-1.0, abs=0.01)


def test_right_stick_up_reads_as_positive_vertical(gamepad: GamepadInput) -> None:
    """Raw axis 4 (right Y, the default GamepadAxisMap.right_y) at "up" means vertical=+1."""
    gamepad._handle_event(_JS_EVENT_AXIS, 4, -32767)
    assert gamepad.read().vertical == pytest.approx(1.0, abs=0.01)


def test_small_axis_values_are_rejected_by_the_deadzone(gamepad: GamepadInput) -> None:
    """A stick value well under the default 0.12 deadzone reads as exactly 0, not near-0."""
    small_value = int(0.05 * 32767)
    gamepad._handle_event(_JS_EVENT_AXIS, 1, small_value)
    assert gamepad.read().forward == 0.0


def test_button_press_is_edge_triggered_and_consumed_on_read(gamepad: GamepadInput) -> None:
    """A press shows up once, on the next read(), then is cleared."""
    gamepad._handle_event(_JS_EVENT_BUTTON, 0, 1)
    assert "0" in gamepad.read().pressed
    assert gamepad.read().pressed == set()  # already consumed


def test_init_events_are_not_counted_as_button_presses(gamepad: GamepadInput) -> None:
    """Every joystick device sends a synthetic INIT event per axis/button on open.

    These must not look like the user pressing every button at once.
    """
    gamepad._handle_event(_JS_EVENT_BUTTON | _JS_EVENT_INIT, 0, 1)
    assert gamepad.read().pressed == set()


def test_button_release_is_not_a_press(gamepad: GamepadInput) -> None:
    """value=0 is a release, not a press - must not appear in pressed."""
    gamepad._handle_event(_JS_EVENT_BUTTON, 0, 0)
    assert gamepad.read().pressed == set()


def test_button_held_is_level_triggered_and_not_consumed_on_read(gamepad: GamepadInput) -> None:
    """Unlike pressed, held stays set across multiple read() calls until the button is released."""
    gamepad._handle_event(_JS_EVENT_BUTTON, 5, 1)
    assert "5" in gamepad.read().held
    assert "5" in gamepad.read().held  # still held, not cleared by read()
    gamepad._handle_event(_JS_EVENT_BUTTON, 5, 0)
    assert "5" not in gamepad.read().held


def test_init_button_down_counts_as_held_but_not_pressed(gamepad: GamepadInput) -> None:
    """A button resting "down" at device-open time is genuinely held."""
    gamepad._handle_event(_JS_EVENT_BUTTON | _JS_EVENT_INIT, 0, 1)
    result = gamepad.read()
    assert "0" in result.held
    assert result.pressed == set()


def test_dpad_hat_axis_decodes_to_held_direction_names(gamepad: GamepadInput) -> None:
    """The D-pad's hat axes decode into held direction names, not raw axis values.

    Default dpad_x=6, dpad_y=7 - see GamepadAxisMap.dpad_x's docstring.
    """
    gamepad._handle_event(_JS_EVENT_AXIS, 6, 32767)  # full right
    assert gamepad.read().held == {"dpad_right"}
    gamepad._handle_event(_JS_EVENT_AXIS, 7, -32767)  # full up
    assert gamepad.read().held == {"dpad_right", "dpad_up"}


def test_dpad_hat_axis_release_clears_the_held_direction(gamepad: GamepadInput) -> None:
    """Releasing a hat axis back to 0 clears its held direction, and only its direction."""
    gamepad._handle_event(_JS_EVENT_AXIS, 6, 32767)  # full right
    gamepad._handle_event(_JS_EVENT_AXIS, 6, 0)  # released
    assert gamepad.read().held == set()


def test_dpad_hat_axis_flipping_directly_swaps_held_direction(gamepad: GamepadInput) -> None:
    """A hat jumping straight from one extreme to the other must not leave a stuck direction.

    Skipping 0 (going straight from full-left to full-right) is something real hardware can
    do - the old direction must still be cleared from held.
    """
    gamepad._handle_event(_JS_EVENT_AXIS, 6, 32767)  # full right
    gamepad._handle_event(_JS_EVENT_AXIS, 6, -32767)  # full left, no pass through 0
    assert gamepad.read().held == {"dpad_left"}


def test_pairing_checks_each_step_really_happened(monkeypatch, capsys):
    """Only an unpaired gamepad is paired; a step counts once ``info`` shows it, not when bluetoothctl returns."""
    import sys

    from chromapi.input import pairing

    pad = "CC:CC:CC:CC:CC:CC"
    state = {}
    kinds = {"AA:AA:AA:AA:AA:AA": "audio-card", "BB:BB:BB:BB:BB:BB": "input-gaming", pad: "input-gaming"}
    calls = []

    def ctl(*args, timeout_s=0):
        args = [a for a in args if a not in ("--agent", "NoInputNoOutput", "--timeout") and not a.isdigit()]
        calls.append(tuple(args))
        command = args[0]
        if command == "devices":
            return True, "Device AA:AA:AA:AA:AA:AA Speaker\nDevice BB:BB:BB:BB:BB:BB Old pad\nDevice CC:CC:CC:CC:CC:CC Wireless Controller"
        if command == "info":
            mac = args[1]
            paired = "yes" if mac == "BB:BB:BB:BB:BB:BB" or state.get("Paired") and mac == pad else "no"
            fields = {f: "yes" if state.get(f) and mac == pad else "no" for f in ("Trusted", "Connected")}
            return True, f"Icon: {kinds[mac]}\nPaired: {paired}\nBonded: {paired}\n" + "\n".join(f"{k}: {v}" for k, v in fields.items())
        if command in effects:
            state[effects[command]] = True
        return True, "Attempting to " + command  # bluetoothctl's answer whether it then works or not

    monkeypatch.setattr(pairing, "_ctl", ctl)
    monkeypatch.setattr(pairing, "CHECK_S", 0.0)
    monkeypatch.setattr(sys, "argv", ["pair"])
    effects = {"trust": "Trusted", "connect": "Connected"}
    assert pairing.main() == 1 and "Could not pair Wireless Controller: not Paired afterwards" in capsys.readouterr().out
    effects["pair"] = "Paired"
    calls.clear()
    assert pairing.main() == 0
    assert [c for c in calls if c[0] in ("pair", "trust", "connect")] == [("trust", pad), ("pair", pad), ("connect", pad)]
    assert pairing.main() == 1 and "No controller found" in capsys.readouterr().out  # paired now: not picked again


def test_motion_sensor_events_low_pass_the_gravity():
    """The kernel structs have their 64-bit sizes; accelerometer events (not the gyro's) feed a smoothed gravity."""
    import os

    from chromapi.input import gamepad

    assert gamepad._INPUT_EVENT.size == 24 and gamepad._FF_EFFECT.size == 48
    pad = gamepad.GamepadInput("/dev/null")
    assert pad.tilt() is None
    read_fd, write_fd = os.pipe()
    pad._motion_fd = read_fd
    events = [(gamepad._EV_ABS, 0, 0), (gamepad._EV_ABS, 1, 8192), (gamepad._EV_ABS, 2, 0),
              (gamepad._EV_ABS, 3, 999),  # gyroscope: ignored
              (gamepad._EV_ABS, 2, 8192)]
    os.write(write_fd, b"".join(gamepad._INPUT_EVENT.pack(0, 0, *e) for e in events))
    pad._read_motion()
    assert pad.tilt() == (0.0, 8192.0, gamepad.ACCEL_SMOOTHING * 8192)
    pad._motion_path = "/dev/input/event-motion"
    pad._open_motion(False)  # the tilt no longer serves: closed, and no stale gravity for next time
    assert pad._motion_fd is None and pad.tilt() is None
    os.close(write_fd)


def test_light_bar_writes_either_led_layout_and_puts_the_colour_back(tmp_path):
    """hid_playstation's one RGB LED, or hid_sony's red, green and blue LEDs (seen on the robot's controller)."""
    from chromapi.input import gamepad

    triplet = []
    for color, level in (("red", "0"), ("green", "0"), ("blue", "64")):
        path = tmp_path / f"input3:{color}" / "brightness"
        path.parent.mkdir()
        path.write_text(level)
        triplet.append(path)
    pad = gamepad.GamepadInput("/dev/null")
    pad._led_files = triplet
    pad.set_color((63, 1, 88))
    assert [p.read_text() for p in triplet] == ["63", "1", "88"]
    writes = []
    pad._write_color = lambda rgb: writes.append(rgb)
    pad._refresh_color()  # after a rumble: one step off and back, or the driver would send nothing
    assert writes == [(64, 1, 88), (63, 1, 88)]
    del pad._write_color
    pad.stop()
    assert [p.read_text() for p in triplet] == ["0", "0", "64"]  # the driver's colour back

    rgb = [tmp_path / "brightness", tmp_path / "multi_intensity"]
    rgb[0].write_text("0")
    rgb[1].write_text("0 0 255")
    pad = gamepad.GamepadInput("/dev/null")
    pad._led_files, pad._led_max = rgb, 255
    pad.set_color((255, 0, 0))
    assert [p.read_text() for p in rgb] == ["255", "255 0 0"]
    pad.stop()
    assert [p.read_text() for p in rgb] == ["0", "0 0 255"]
