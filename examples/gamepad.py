# SPDX-License-Identifier: Apache-2.0
"""Drive Chromapi with a gamepad: left stick to walk, right stick to turn, button 0 to stand up / lie down.

Pair the controller first (see :mod:`chromapi.input.gamepad`). If a stick moves the wrong way,
check your controller's axis numbers with ``jstest /dev/input/js0`` and adjust
:class:`~chromapi.input.gamepad.GamepadAxisMap`.

Usage::

    python examples/gamepad.py                   # on the robot
    python examples/gamepad.py --sim             # in MuJoCo, with a gamepad plugged into the PC
    python examples/gamepad.py --sim --follow    # in MuJoCo, the camera following the robot
"""

import argparse
import time

import numpy as np

from chromapi.chromapi import IDLE_LED_RGB, Chromapi
from chromapi.input.gamepad import GamepadInput

MAX_LINEAR_MPS = 0.10  # [m.s⁻¹] at full stick deflection
MAX_ANGULAR_RAD_S = np.radians(40.0)  # [rad.s⁻¹]
TOGGLE_BUTTON = "0"  # A on Xbox layouts, cross on PlayStation ones


def main() -> None:
    """Teleoperate until Ctrl-C."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", action="store_true", help="run in MuJoCo instead of on the robot")
    parser.add_argument("--follow", action="store_true", help="in MuJoCo, the camera follows the robot")
    parser.add_argument("--device", default="/dev/input/js0", help="joystick device")
    args = parser.parse_args()

    gamepad = GamepadInput(args.device)
    try:
        gamepad.start()
    except (FileNotFoundError, PermissionError) as exc:
        raise SystemExit(f"[gamepad] {exc}") from exc

    with Chromapi(backend="mujoco" if args.sim else "hardware", launch_viewer=args.sim, follow_robot=args.follow) as robot:
        awake = False
        print(f"Press button {TOGGLE_BUTTON} to stand up, Ctrl-C to quit.")
        try:
            while True:
                user = gamepad.read()
                if TOGGLE_BUTTON in user.pressed:
                    if awake:
                        robot.stop_walk()
                        if not args.sim:
                            robot.set_led_all(0, 0, 0)
                        robot.rest()
                    else:
                        robot.wake_up()
                        time.sleep(robot.wake_up_duration_s + 0.3)
                        if not args.sim:
                            robot.set_led_all(*IDLE_LED_RGB)
                    awake = not awake
                elif awake:
                    # UserInput.lateral is positive to the right, walk()'s vy positive to the left.
                    robot.walk(
                        vx=user.forward * MAX_LINEAR_MPS,
                        vy=-user.lateral * MAX_LINEAR_MPS,
                        wz=user.turn * MAX_ANGULAR_RAD_S,
                    )
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            gamepad.stop()
            if awake:
                robot.stop_walk()
                time.sleep(0.3)
                if not args.sim:
                    robot.set_led_all(0, 0, 0)
                robot.rest()


if __name__ == "__main__":
    main()
