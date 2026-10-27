# SPDX-License-Identifier: Apache-2.0
"""Support report: one text file with what helps to understand a problem, to post on forums.

Versions, health, calibration, services, recent logs. No passwords and no network configuration.

Usage::

    chromapi report
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from chromapi.chromapi import USER_CALIBRATION_PATH

BRIDGE_COMMANDS: List[List[str]] = [  # they open the serial port: not while a robot program holds it
    [sys.executable, "-m", "chromapi", "version"],
    [sys.executable, "-m", "chromapi", "health"],
]
COMMANDS: List[List[str]] = [
    ["cat", "/proc/device-tree/model"],
    ["uname", "-a"],
    [sys.executable, "--version"],
    ["uptime"],
    ["vcgencmd", "get_throttled"],  # under-voltage and throttling since boot
    ["vcgencmd", "measure_temp"],
    ["cat", str(USER_CALIBRATION_PATH)],
    ["systemctl", "status", "--no-pager", "chromapi", "chromapi-audio-keepalive", "chromapi-fan@*"],
    ["journalctl", "--no-pager", "-n", "300", "-u", "chromapi"],
    [sys.executable, "-m", "pip", "list"],
]


def run(command: List[str]) -> str:
    """Output of ``command`` (stdout and stderr), or why it could not run."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"[not run: {e}]\n"
    return (done.stdout + done.stderr).replace("\0", "")  # /proc/device-tree/model ends with a NUL


def write_report(path: Path, use_bridge: bool = True, extra: str = "") -> Path:
    """Write the report to ``path``; ``extra`` (e.g. the live robot state) goes at the end."""
    with open(path, "w") as f:
        for command in (BRIDGE_COMMANDS if use_bridge else []) + COMMANDS:
            f.write(f"$ {' '.join(command)}\n{run(command)}\n")
        f.write(extra)
    return path


def main() -> int:
    """Write the report in the home folder and print its path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="report file (default: ~/chromapi_report_<date>.txt)")
    args = parser.parse_args()

    path = args.output or Path.home() / f"chromapi_report_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    write_report(path)
    print(f"Report written to {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
