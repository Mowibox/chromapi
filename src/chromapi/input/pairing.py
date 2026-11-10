# SPDX-License-Identifier: Apache-2.0
"""Pair a Bluetooth gamepad with ``bluetoothctl``: the first unpaired one found in pairing mode.

Usage::

    chromapi gamepad pair
"""

import argparse
import re
import subprocess
import time
from typing import List, Tuple

SCAN_S = 20  # [s]
STEP_TIMEOUT_S = 30  # [s]
# (command, fields `info` must show, --timeout [s]); trusted first, or a controller pad is refused right after pairing
STEPS = (("trust", ("Trusted",), 0.0), ("pair", ("Paired", "Bonded"), 15.0), ("connect", ("Connected",), 10.0))
CHECK_S = 5.0  # [s] for a step's state to show in `info`
_DEVICE = re.compile(r"^Device ([0-9A-F:]{17}) (.*)$")
_COLOUR = re.compile(r"\x1b\[[0-9;]*m|\x01|\x02")  # bluetoothctl colours even into a pipe


def _ctl(*args: str, timeout_s: float = STEP_TIMEOUT_S) -> Tuple[bool, str]:
    """Run one ``bluetoothctl`` command: (succeeded, output)."""
    try:
        done = subprocess.run(["bluetoothctl", *args], capture_output=True, text=True, timeout=timeout_s, check=False)
    except FileNotFoundError:
        return False, "bluetoothctl not found: sudo apt install bluez"
    except subprocess.TimeoutExpired:
        return False, f"bluetoothctl {' '.join(args)}: no answer after {timeout_s:.0f} s"
    output = _COLOUR.sub("", done.stdout + done.stderr).strip()
    return done.returncode == 0 and "Failed" not in output and "not available" not in output, output


def _reached(mac: str, fields: Tuple[str, ...]) -> bool:
    """``info`` shows ``<field>: yes`` for each field it lists, within :data:`CHECK_S`."""
    deadline = time.monotonic() + CHECK_S
    while True:
        info = _ctl("info", mac)[1]
        if all(f"{field}: yes" in info or f"{field}:" not in info for field in fields):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def unpaired_gamepads() -> List[Tuple[str, str]]:
    """(MAC, name) of the gamepads BlueZ knows of and has not paired yet."""
    found = []
    for line in _ctl("devices")[1].splitlines():
        match = _DEVICE.match(line.strip())
        if match:
            info = _ctl("info", match[1])[1]
            if "Icon: input-gaming" in info and "Paired: no" in info:
                found.append((match[1], match[2]))
    return found


def main() -> int:
    """Pair the first gamepad found; the adapter is pairable only meanwhile."""
    try:
        return _pair()
    finally:
        _ctl("pairable", "off")


def _pair() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scan", type=float, default=SCAN_S, help="[s] how long to look for the controller")
    args = parser.parse_args()
    ok, output = _ctl("power", "on")
    if ok:  # not pairable: paired without a bond, then refused
        ok, output = _ctl("pairable", "on")
    if not ok:
        print(f"Bluetooth does not start: {output.splitlines()[-1] if output else 'no adapter'}")
        return 1
    print(f"Looking for a controller in pairing mode for {args.scan:.0f} s...", flush=True)
    _ctl("--timeout", f"{args.scan:.0f}", "scan", "on", timeout_s=args.scan + STEP_TIMEOUT_S)
    gamepads = unpaired_gamepads()
    if not gamepads:
        print("No controller found: hold its pair button until its light flashes fast, then try again")
        return 1
    mac, name = gamepads[0]
    for step, fields, open_s in STEPS:
        print(f"{step} {name} ({mac})", flush=True)
        keep_open = ("--agent", "NoInputNoOutput", "--timeout", f"{open_s:.0f}") if open_s else ()
        ok, output = _ctl(*keep_open, step, mac, timeout_s=open_s + STEP_TIMEOUT_S)
        if not ok or not _reached(mac, fields):
            failed = [line for line in output.splitlines() if "Failed" in line]
            print(f"Could not {step} {name}: {failed[-1] if failed else f'not {fields[0]} afterwards'}")
            return 1
    print(f"Paired {name}: it reconnects on its own when switched on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
