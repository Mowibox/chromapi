# SPDX-License-Identifier: Apache-2.0
"""Chromapi command line: one ``chromapi`` command for every robot tool.

Usage::

    chromapi run
    chromapi version
    chromapi calibrate servos --all
    chromapi calibrate imu --dry-run
    chromapi motors setup
    chromapi health
    chromapi flash
    chromapi report
    chromapi <command> --help
"""

import argparse
import importlib
import sys
from typing import Any, Dict, List, Optional

from chromapi.chromapi import sdk_version
from chromapi.hardware.motherboard_bridge import BRIDGE_PROTOCOL_VERSION, BridgeClient

#: ``chromapi <group> <tool>`` -> (``module:function`` of a tool that parses its own arguments, help).
TOOLS: Dict[str, Any] = {
    "calibrate": {
        "servos": ("chromapi.utils.calibrate_servo_centers:main", "find each servo's center from its mechanical stops"),
        "imu": ("chromapi.utils.calibrate_imu_mount:main", "measure the IMU board tilt on a level floor"),
    },
    "motors": {
        "setup": ("chromapi.utils.setup_motors:main", "set each servo's ID and registers (--all: the 12 already assembled)"),
    },
    "run": ("chromapi.web.app:main", "start the robot and its web app (http://<robot-ip>:5000)"),
    "health": ("chromapi.utils.health:main", "what is wrong with the robot, and what to do about it"),
    "flash": ("chromapi.utils.flash:main", "update the motherboard firmware (STM32)"),
    "report": ("chromapi.utils.report:main", "write a support report to post on Discord"),
}


def version(port: str) -> int:
    """Print the SDK, firmware and bridge protocol versions; return 1 if they can't work together."""
    print(f"chromapi SDK {sdk_version()} · bridge protocol {BRIDGE_PROTOCOL_VERSION}")
    bridge = BridgeClient(port=port)
    if not bridge.connect():
        print(f"firmware     unknown (cannot open {port})")
        return 1
    try:
        fw = bridge.get_version()
    finally:
        bridge.close()
    if fw is None:
        print("firmware     older than 1.1.0, or not answering - update it")
        return 1
    protocol, (major, minor, patch) = fw
    print(f"firmware     {major}.{minor}.{patch} · bridge protocol {protocol}")
    if protocol != BRIDGE_PROTOCOL_VERSION:
        print(f"[!] protocol mismatch: firmware {protocol}, SDK {BRIDGE_PROTOCOL_VERSION} - update one of them")
        return 1
    return 0


def _run_tool(name: str, target: str, args: List[str]) -> None:
    """Run a tool's own CLI on ``args``, under its ``chromapi ...`` name."""
    module, function = target.split(":")
    sys.argv = [name, *args]  # the tool parses sys.argv itself
    sys.exit(getattr(importlib.import_module(module), function)())


def main(argv: Optional[List[str]] = None) -> None:
    """Dispatch ``chromapi <command>``."""
    parser = argparse.ArgumentParser(prog="chromapi", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    version_parser = commands.add_parser("version", help="SDK, firmware and bridge protocol versions")
    version_parser.add_argument("--port", default="/dev/ttyAMA0", help="serial port to the STM32 bridge")
    for name, entry in TOOLS.items():
        if isinstance(entry, tuple):
            tool = commands.add_parser(name, help=entry[1], add_help=False)
            tool.set_defaults(target=entry[0], name=f"chromapi {name}")
            continue
        group = commands.add_parser(name, help=", ".join(entry)).add_subparsers(required=True, metavar="<tool>")
        for tool_name, (target, help_text) in entry.items():
            tool = group.add_parser(tool_name, help=help_text, add_help=False)
            tool.set_defaults(target=target, name=f"chromapi {name} {tool_name}")

    # Unknown arguments belong to the tool (argparse.REMAINDER would reject a leading --help).
    args, tool_args = parser.parse_known_args(argv)
    if args.command == "version":
        if tool_args:
            parser.error(f"unrecognized arguments: {' '.join(tool_args)}")
        sys.exit(version(args.port))
    _run_tool(args.name, args.target, tool_args)


if __name__ == "__main__":
    main()
