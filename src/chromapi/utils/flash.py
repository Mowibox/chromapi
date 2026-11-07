    # SPDX-License-Identifier: Apache-2.0
"""Flash the motherboard firmware (STM32) from the Raspberry Pi, through its ROM bootloader.

The servos lose their torque while the STM32 resets: make sure Chromapi is laid down.

Usage::

    chromapi flash                      # the firmware shipped with this SDK
    chromapi flash path/to/firmware.hex
"""

import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from chromapi.chromapi import load_config
from chromapi.hardware.battery import Battery
from chromapi.hardware.motherboard_bridge import BridgeClient
from chromapi.utils.health import port_users

FIRMWARE_DIR = Path(__file__).resolve().parents[1] / "assets" / "firmware"
FLASH_SCRIPT = FIRMWARE_DIR / "flash.sh"
BOOT_TIMEOUT_S = 8.0  # [s] for the flashed firmware to answer (IMU calibration first)
SHIPPED_HEX_RE = re.compile(r"chromapi_stm32_core_v(\d+\.\d+\.\d+)\.hex")


def _version_key(version: str) -> Tuple[int, ...]:
    return tuple(map(int, version.split(".")))


def shipped_firmware() -> Optional[Tuple[Path, str]]:
    """``(hex, version)`` of the newest firmware shipped with this SDK, None if none."""
    found = [(path, match.group(1)) for path in FIRMWARE_DIR.glob("*.hex")
             if (match := SHIPPED_HEX_RE.fullmatch(path.name))]
    return max(found, key=lambda hex_version: _version_key(hex_version[1]), default=None)


def shipped_version() -> Optional[str]:
    """Version of the firmware shipped with this SDK; None if this install has none."""
    shipped = shipped_firmware()
    return shipped[1] if shipped else None


def update_available(installed: Optional[str], shipped: Optional[str]) -> bool:
    """Whether ``shipped`` is newer than ``installed``."""
    if shipped is None:
        return False
    if installed is None:
        return True
    return _version_key(shipped) > _version_key(installed)


def read_firmware(port: str) -> Optional[str]:
    """Firmware version string; None if the STM32 does not answer (or predates GET_VERSION)."""
    bridge = BridgeClient(port=port)
    if not bridge.connect():
        return None
    try:
        version = bridge.get_version()
    finally:
        bridge.close()
    return None if version is None else ".".join(map(str, version[1]))


def battery_percent(port: str) -> Optional[float]:
    """Charge estimate from ~1 s of readings; None if the STM32 does not answer."""
    battery = Battery(load_config()["battery"])
    bridge = BridgeClient(port=port)
    if not bridge.connect():
        return None
    try:
        for i in range(5):
            snap = bridge.get_state(timeout=0.2)
            if snap is not None:
                battery.update(snap["power"]["voltage_V"], snap["power"]["current_A"], 0.2 * i)
    finally:
        bridge.close()
    return battery.percent


def main() -> int:
    """Check the port and the battery, confirm, flash, then read the new version back."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    shipped = shipped_firmware()
    parser.add_argument("hex", nargs="?", type=Path, default=shipped[0] if shipped else None,
                        help="firmware file (.hex); default: the one shipped with this SDK")
    parser.add_argument("--port", default="/dev/ttyAMA0", help="serial port to the STM32")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    if args.hex is None or not args.hex.is_file():
        print(f"[!] {args.hex} not found: give the path of a firmware .hex")
        return 1
    users = port_users(args.port)
    if users:
        print(f"[!] {args.port} is open by {', '.join(users)}: stop it first (robot program, web app)")
        return 1

    before = read_firmware(args.port)
    percent = battery_percent(args.port)
    critical = load_config()["battery"]["critical_percent"]
    if percent is not None and percent < critical:
        print(f"[!] Battery at {percent:.0f} %: charge it before flashing")
        return 1
    print(f"Installed firmware: {before or 'unknown (not answering, or older than 2.0.0)'}")
    print(f"Battery: {'unknown' if percent is None else f'{percent:.0f} %'}")
    print(f"New firmware: {args.hex}")
    if not args.yes and input("Lay Chromapi down: the servos go limp. Flash now? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    for attempt in (1, 2):
        if subprocess.run(["bash", str(FLASH_SCRIPT), str(args.hex)], check=False).returncode == 0:
            break
        if attempt == 1:
            print("[!] The bootloader did not answer: trying again")
    else:
        print("[!] Flash failed twice. The STM32 bootloader cannot be erased: check the wiring and the UART, then "
              "run 'chromapi flash' again.")
        return 1

    # The firmware calibrates its IMU at boot before answering: its timeouts are expected, not warnings.
    logging.getLogger("BridgeClient").setLevel(logging.ERROR)
    deadline = time.monotonic() + BOOT_TIMEOUT_S  # [s]
    while time.monotonic() < deadline:
        after = read_firmware(args.port)
        if after is not None:
            print(f"Done: firmware {after} (was {before or 'unknown'})")
            return 0
        time.sleep(0.5)
    print("[!] Flashed, but the STM32 does not answer: power-cycle the robot and run 'chromapi version'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
