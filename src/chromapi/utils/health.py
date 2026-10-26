# SPDX-License-Identifier: Apache-2.0
"""Chromapi health check: what is wrong, named plainly, with what to do about it.

Read-only: nothing moves. Exits with code 1 if any check fails, so scripts and the web app can use it too.

Usage::

    chromapi health
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from chromapi.chromapi import USER_CALIBRATION_PATH, load_config
from chromapi.hardware.battery import CRITICAL, EMPTY, LOW, Battery
from chromapi.hardware.motherboard_bridge import BRIDGE_PROTOCOL_VERSION, BridgeClient

OK, WARN, FAIL = "OK", "WARN", "FAIL"
Result = Tuple[str, str, str]  # (level, check, detail)

SNAPSHOTS = 5  # a servo missing one bus cycle is not a fault: it must answer in at least one snapshot
SERVO_HOT_C = 60  # [°C]
GRAVITY_TOLERANCE = 1.5  # [m.s⁻²] |acc| away from g by more than this: robot moving, or IMU wrong
SERVO_ID_RANGE = range(1, 253)
SOUND_CARD = "sndrpigooglevoi"  # see deploy/asound.conf
AUDIO_KEEPALIVE = "chromapi-audio-keepalive"
BOOT_CMDLINES = (Path("/boot/firmware/cmdline.txt"), Path("/boot/cmdline.txt"))


def port_users(port: str) -> List[str]:
    """Return ``"<comm> (PID <pid>)"`` for every other process holding ``port`` open."""
    target = os.path.realpath(port)
    users = []
    for proc in Path("/proc").glob("[0-9]*"):
        if int(proc.name) == os.getpid():
            continue
        try:
            if any(os.path.realpath(fd) == target for fd in (proc / "fd").iterdir()):
                users.append(f"{(proc / 'comm').read_text().strip()} (PID {proc.name})")
        except OSError:  # process gone, or another user's
            continue
    return users


def check_system(port: str) -> List[Result]:
    """Checks that need no bridge: serial console, port ownership, audio."""
    results = []
    for cmdline in BOOT_CMDLINES:
        if cmdline.exists():
            if any(c in cmdline.read_text() for c in ("console=serial0", "console=ttyAMA0")):
                results.append((FAIL, "serial console", f"Linux console is on the bridge port ({cmdline}): "
                                "run 'sudo raspi-config nonint do_serial_cons 1' and reboot"))
            else:
                results.append((OK, "serial console", "off"))
            break

    users = port_users(port)
    if users:
        results.append((FAIL, "bridge port", f"{port} already open by {', '.join(users)}: stop it first"))

    if shutil.which("aplay") is None:
        results.append((FAIL, "sound card", "aplay not found: sudo apt install alsa-utils"))
    elif SOUND_CARD in subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=False).stdout:
        results.append((OK, "sound card", SOUND_CARD))
    else:
        results.append((FAIL, "sound card", f"{SOUND_CARD} not found by ALSA: check "
                        "'dtoverlay=googlevoicehat-soundcard' in config.txt and the I2S wiring"))

    if shutil.which("systemctl"):
        state = subprocess.run(["systemctl", "is-active", AUDIO_KEEPALIVE],
                               capture_output=True, text=True, check=False).stdout.strip()
        if state == "active":
            results.append((OK, "audio keep-alive", "active"))
        else:
            results.append((WARN, "audio keep-alive", f"{state or 'missing'}: the speaker will click, "
                            f"install deploy/{AUDIO_KEEPALIVE}.service"))
    return results


def check_calibration(path: Path) -> List[Result]:
    """Is each calibration measured on this robot, rather than the package defaults?"""
    try:
        calibration: Dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        calibration = {}
    results = []
    for section, key, label, command in (
        ("servos", "zero_offset_steps", "servo centers", "chromapi calibrate servos --all"),
        ("imu", "mount_offset_wxyz", "IMU mount", "chromapi calibrate imu"),
    ):
        if key in (calibration.get(section) or {}):
            results.append((OK, label, f"calibrated ({path})"))
        else:
            results.append((WARN, label, f"using package defaults: run '{command}'"))
    return results


def check_robot(bridge: Any, config: Dict[str, Any]) -> List[Result]:
    """Checks through the bridge. ``bridge`` must be connected; nothing here moves the robot."""
    id_map: Dict[str, int] = config["servos"]["id_map"]
    if not bridge.ping():
        return [(FAIL, "bridge", "STM32 not answering: check the motherboard power and the UART wires")]
    results = [(OK, "bridge", "STM32 answering")]

    version = bridge.get_version()
    if version is None:
        results.append((FAIL, "firmware", "older than 1.1.0 (no version command): update it"))
    elif version[0] != BRIDGE_PROTOCOL_VERSION:
        results.append((FAIL, "firmware", f"{'.'.join(map(str, version[1]))} speaks bridge protocol {version[0]}, "
                        f"this SDK speaks {BRIDGE_PROTOCOL_VERSION}: update one of them"))
    else:
        results.append((OK, "firmware", f"{'.'.join(map(str, version[1]))}, bridge protocol {version[0]}"))

    snapshots = [s for s in (bridge.get_state(timeout=0.2) for _ in range(SNAPSHOTS)) if s is not None]
    if not snapshots:
        results.append((FAIL, "state", "no state snapshot: firmware and SDK out of sync?"))
        return results
    last = snapshots[-1]

    battery = Battery(config["battery"])
    for i, snap in enumerate(snapshots):
        battery.update(snap["power"]["voltage_V"], snap["power"]["current_A"], 0.2 * i)
    power = last["power"]
    reading = f"{power['voltage_V']:.2f} V, {power['current_A']:.2f} A"
    if battery.percent is None:
        results.append((FAIL, "battery", "power monitor reads 0 V: INA226 / I2C problem"))
    elif battery.level in (EMPTY, CRITICAL):
        results.append((FAIL, "battery", f"{battery.percent:.0f} % ({reading}): too low to walk, need to charge"))
    elif battery.level == LOW:
        results.append((WARN, "battery", f"{battery.percent:.0f} % ({reading}): consider charging soon"))
    else:
        results.append((OK, "battery", f"{battery.percent:.0f} % ({reading})"))

    acc = last["imu"]["acc_mps2"]  # [m.s⁻²]
    norm = math.sqrt(sum(a * a for a in acc))  # [m.s⁻²]
    if norm == 0.0:
        results.append((FAIL, "IMU", "accelerometer reads zeros: SPI link to the BMI088 dead?"))
    elif abs(norm - 9.81) > GRAVITY_TOLERANCE:
        results.append((WARN, "IMU", f"gravity reads {norm:.2f} m/s²: keep the robot still and run again"))
    else:
        results.append((OK, "IMU", f"gravity {norm:.2f} m/s²"))

    answering = {s["id"] for snap in snapshots for s in snap["servos"] if s["fresh"]}
    missing = {joint: sid for joint, sid in id_map.items() if sid not in answering}
    for joint, sid in missing.items():
        results.append((FAIL, f"servo {joint}", f"id {sid} not answering: check its cable"))
    if missing:
        # The usual cause: a new servo still on its factory id, or two servos sharing one.
        expected = set(id_map.values())
        strays = [i for i in SERVO_ID_RANGE if i not in expected and bridge.ping_servo(i)]
        for i in strays:
            results.append((FAIL, "servo bus", f"a servo answers on unexpected id {i}: "
                            "give it its id with 'chromapi motors setup'"))
    else:
        results.append((OK, "servos", f"all {len(id_map)} answering"))
    for s in last["servos"]:
        if s["fresh"] and s["temp_C"] > SERVO_HOT_C:
            results.append((WARN, f"servo id {s['id']}", f"{s['temp_C']} °C: let it cool down"))
    return results


def run_checks(port: str, calibration_path: Path) -> List[Result]:
    """Run every check and return the results, in order."""
    results = check_system(port) + check_calibration(calibration_path)
    bridge = BridgeClient(port=port)
    if not bridge.connect():
        return results + [(FAIL, "bridge port", f"cannot open {port}: is the UART enabled in config.txt?")]
    try:
        return results + check_robot(bridge, load_config(calibration_path=calibration_path))
    finally:
        bridge.close()


def main() -> int:
    """Print the health report; return 1 if any check failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyAMA0", help="serial port to the STM32 bridge")
    parser.add_argument("--calibration", type=Path, default=USER_CALIBRATION_PATH, help="calibration file")
    args = parser.parse_args()

    results = run_checks(args.port, args.calibration)
    for level, check, detail in results:
        print(f"[{level:^4}] {check}: {detail}")
    failed = sum(level == FAIL for level, _, _ in results)
    warned = sum(level == WARN for level, _, _ in results)
    print(f"\n{'healthy' if not failed else f'{failed} problem(s)'}"
          + (f", {warned} warning(s)" if warned else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
