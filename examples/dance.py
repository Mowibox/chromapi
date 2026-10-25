# SPDX-License-Identifier: Apache-2.0
"""Chromapi dances on the spot: feet planted, the trunk tilts around in a circle under a spinning rainbow.

Shows how to write a :class:`~chromapi.chromapi.Move`: a behavior that updates the joint
targets on every control-loop tick, here with the whole-body IK (:meth:`Kromatics.body_ik`).

Usage::

    python examples/dance.py          # on the robot
    python examples/dance.py --sim    # in MuJoCo simulation
"""

import argparse
import colorsys
import time

import numpy as np

from chromapi.chromapi import (
    DEFAULT_CHROMAPI_SCREAM_PATH,
    Chromapi,
    MotorCommand,
    Move,
    RobotState,
)
from chromapi.kinematics.kromatics import Kromatics
from chromapi.locomotion.gait import WALK_POSTURE
from chromapi.media.audio import play_wav

TILT_RAD = np.radians(8.0)  # [rad] roll/pitch amplitude of the circle
CIRCLE_S = 2.0  # [s] per full circle, for the trunk and the rainbow alike
LED_COUNT = 18
LED_BRIGHTNESS = 0.4  # [0, 1]


class TiltCircle(Move):
    """Roll and pitch trace a circle, eased in over the first circle."""

    def __init__(self, kinematics: Kromatics) -> None:
        """Dance on the walking footprint."""
        self.kinematics = kinematics
        x_reach, y_reach, self.height = WALK_POSTURE
        self.footprint = kinematics.stance_targets(self.height, x_reach, y_reach)
        self.t = 0.0  # [s]

    def on_start(self) -> None:
        """Restart the dance from the beginning."""
        self.t = 0.0

    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """Solve this tick's trunk orientation."""
        self.t += dt
        amplitude = TILT_RAD * min(1.0, self.t / CIRCLE_S)
        theta = 2.0 * np.pi * self.t / CIRCLE_S
        rpy = (amplitude * np.cos(theta), amplitude * np.sin(theta), 0.0)
        q, _ = self.kinematics.body_ik(
            np.zeros(3), np.array(rpy), foot_targets_world=self.footprint,
            reference_height=self.height, q_init=command.target_angles,
        )
        command.target_angles.update(q)


def rainbow(turns: float) -> list:
    """Return the ring's colors: a full hue wheel, rotated by ``turns`` (1.0 = one full turn)."""
    return [
        tuple(round(255 * c) for c in colorsys.hsv_to_rgb((i / LED_COUNT + turns) % 1.0, 1.0, LED_BRIGHTNESS))
        for i in range(LED_COUNT)
    ]


def main() -> None:
    """Stand up, dance, lie down."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", action="store_true", help="run in MuJoCo instead of on the robot")
    parser.add_argument("--duration", type=float, default=8.0, help="dance length [s]")
    args = parser.parse_args()

    with Chromapi(backend="mujoco" if args.sim else "hardware", launch_viewer=args.sim) as robot:
        robot.wake_up()
        time.sleep(robot.wake_up_duration_s + 0.3)

        x_reach, y_reach, height = WALK_POSTURE
        robot.go_to_pose(duration=1.0, x_reach=x_reach, y_reach=y_reach, reference_height=height)
        time.sleep(1.5)

        dance = TiltCircle(robot.kinematics)
        play_wav(DEFAULT_CHROMAPI_SCREAM_PATH, wait=False)
        robot.play_move(dance)
        while dance.t < args.duration:
            if not args.sim:
                robot.set_led_bulk(rainbow(dance.t / CIRCLE_S))  # one ring turn per trunk circle
            time.sleep(0.05)

        robot.go_to_pose(duration=0.5, x_reach=x_reach, y_reach=y_reach, reference_height=height)
        time.sleep(0.8)
        if not args.sim:
            robot.set_led_all(0, 0, 0)
        robot.rest()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl-C, or the simulation window closed
        pass
