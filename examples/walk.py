# SPDX-License-Identifier: Apache-2.0
"""Chromapi walks there and back: forward and back, left and right, a quarter turn and back.

Usage::

    python examples/walk.py                    # on the robot
    python examples/walk.py --sim              # in MuJoCo
    python examples/walk.py --sim --follow     # in MuJoCo, the camera following the robot
    python examples/walk.py --pattern crawl    # slower, statically stable gait
"""

import argparse
import math
import time

from chromapi.chromapi import AUDIO_DIR, IDLE_LED_RGB, Chromapi
from chromapi.media.audio import play_wav


def main() -> None:
    """Stand up, walk a short there-and-back sequence, lie down."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", action="store_true", help="run in MuJoCo instead of on the robot")
    parser.add_argument("--follow", action="store_true", help="in MuJoCo, the camera follows the robot")
    parser.add_argument("--pattern", choices=["trot", "crawl"], default="trot", help="gait pattern")
    args = parser.parse_args()

    with Chromapi(backend="mujoco" if args.sim else "hardware", launch_viewer=args.sim, follow_robot=args.follow) as robot:
        robot.wake_up()
        time.sleep(robot.wake_up_duration_s + 0.3)
        start_yaw = robot.get_imu_yaw()  # [rad]
        if not args.sim:
            robot.set_led_all(*IDLE_LED_RGB)
        play_wav(AUDIO_DIR / "squeak.wav", wait=False)

        for distance in (0.3, -0.3):  # [m] forward, then back
            robot.walk_distance(distance, pattern=args.pattern)
        for distance in (0.1, -0.1):  # [m] left, then right
            robot.walk_distance(distance, lateral=True, pattern=args.pattern)
        for angle in (90.0, -90.0):  # [deg] left, then right
            robot.turn(angle, pattern=args.pattern)
        drift = (robot.get_imu_yaw() - start_yaw + math.pi) % (2.0 * math.pi) - math.pi  # [rad]
        robot.turn(-math.degrees(drift), pattern=args.pattern)
        time.sleep(1.0)  # let the feet settle on the stance footprint

        if not args.sim:
            robot.set_led_all(0, 0, 0)
        robot.rest()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl-C, or the simulation window closed
        pass
