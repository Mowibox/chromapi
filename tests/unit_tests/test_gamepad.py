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
    """Turn follows Chromapi.walk()'s wz convention (positive = left) - see UserInput's docstring."""
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
