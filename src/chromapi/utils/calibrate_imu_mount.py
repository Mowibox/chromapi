# SPDX-License-Identifier: Apache-2.0
"""IMU Mount Calibration Tool.

Put the robot on a flat floor (rest pose or standing, chassis level) and don't touch it.

Usage::

    chromapi-calibrate-imu
    chromapi-calibrate-imu --duration 10 --dry-run
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

from chromapi.chromapi import USER_CALIBRATION_PATH, save_calibration
from chromapi.estimation.attitude import ImuMount, roll_pitch
from chromapi.hardware.motherboard_bridge import BridgeClient

STILL_GYRO_RPS = 0.05  # [rad.s⁻¹] samples above this are dropped (robot moving)
MIN_STILL_FRACTION = 0.9
SAMPLE_PERIOD_S = 0.01  # [s]


def _record(bridge: BridgeClient, duration_s: float) -> List[Sequence[float]]:
    """Quaternions read while the gyro says the robot is still."""
    quats: List[Sequence[float]] = []
    n_total = 0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        state = bridge.get_state()
        if state is not None:
            n_total += 1
            if np.linalg.norm(state["imu"]["gyro_rps"]) < STILL_GYRO_RPS:
                quats.append(state["imu"]["quat"])
        time.sleep(SAMPLE_PERIOD_S)
    if n_total == 0 or len(quats) < MIN_STILL_FRACTION * n_total:
        raise RuntimeError(f"robot not still enough ({len(quats)}/{n_total} still samples)")
    return quats


def main() -> None:
    """Record, compute and write the IMU mount offset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyAMA0", help="UART bridge serial device")
    parser.add_argument("--calibration", type=Path, default=USER_CALIBRATION_PATH, help="calibration file to update")
    parser.add_argument("--duration", type=float, default=5.0, help="recording time [s]")
    parser.add_argument("--dry-run", action="store_true", help="measure and print, but never write the calibration")
    args = parser.parse_args()

    bridge = BridgeClient(port=args.port)
    if not bridge.connect():
        sys.exit("[!] Cannot open the UART bridge.")
    try:
        print(f"[*] Recording {args.duration:.0f} s - keep the robot still on a level floor...")
        quats = _record(bridge, args.duration)
    except RuntimeError as exc:
        sys.exit(f"[!] {exc}")
    finally:
        bridge.close()

    mount = ImuMount.from_level_samples(quats)
    roll, pitch = roll_pitch(mount.offset)
    print(f"[*] {len(quats)} samples - board offset: roll {math.degrees(roll):+.2f} deg, "
          f"pitch {math.degrees(pitch):+.2f} deg")
    print(f"    mount_offset_wxyz: [{', '.join(f'{v:.6f}' for v in mount.offset)}]")
    if args.dry_run:
        print("(--dry-run: calibration left untouched)")
        return
    path = save_calibration({"imu": {"mount_offset_wxyz": [float(v) for v in mount.offset]}}, args.calibration)
    print(f"[SUCCESS] Written to {path}")


if __name__ == "__main__":
    main()
