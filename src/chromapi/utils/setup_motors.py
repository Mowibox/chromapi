# SPDX-License-Identifier: Apache-2.0
"""Motor Setup and Full Provisioning Tool.

Usage::

    chromapi motors setup          # new servos: give each its ID, one servo on the bus at a time
    chromapi motors setup --all    # assembled robot: rewrite the registers of all 12, IDs unchanged
    chromapi motors setup --id N   # the new servo (ID 1) becomes N, no prompt
"""

import argparse
import time
from typing import Dict, List, Optional

from chromapi.chromapi import load_config
from chromapi.hardware.motherboard_bridge import BridgeClient

FACTORY_ID = 1
# Feetech STS3215 Register Map
REG_ID = 5
REG_MAX_VOLT = 14
REG_MIN_VOLT = 15
REG_UNLOAD_COND = 19
REG_P = 21
REG_D = 22
REG_I = 23
REG_MODE = 33
REG_ACCEL = 41
REG_LOCK = 55
# Undocumented: acceleration cap of the internal motion profile.
REG_MAX_ACCEL = 85
MAX_ACCEL = 254
# What configure_servo() writes, as read_servo_config() reads it back.
EXPECTED = {"P": 32, "I": 0, "D": 0, "Acceleration": 0, "Max Accel": MAX_ACCEL, "Mode": 0,
            "Max Voltage": 85, "Min Voltage": 60, "Unload Cond": 44}

def read_servo_config(bridge: BridgeClient, servo_id: int) -> Dict[str, Optional[int]]:
    """Read back current configuration registers directly from the hardware."""
    return {
        "P": bridge.read_servo_register(servo_id, REG_P, 1),
        "I": bridge.read_servo_register(servo_id, REG_I, 1),
        "D": bridge.read_servo_register(servo_id, REG_D, 1),
        "Acceleration": bridge.read_servo_register(servo_id, REG_ACCEL, 1),
        "Max Accel": bridge.read_servo_register(servo_id, REG_MAX_ACCEL, 1),
        "Mode": bridge.read_servo_register(servo_id, REG_MODE, 1),
        "Max Voltage": bridge.read_servo_register(servo_id, REG_MAX_VOLT, 1),
        "Min Voltage": bridge.read_servo_register(servo_id, REG_MIN_VOLT, 1),
        "Unload Cond": bridge.read_servo_register(servo_id, REG_UNLOAD_COND, 1),
    }

def configure_servo(bridge: BridgeClient, target_id: int, new_id: int) -> Dict[str, Optional[int]]:
    """Unlock, configure, assign ID, and verify parameters; return the registers that read back wrong."""
    print(f"[*] Unlocking EEPROM for ID {target_id}...")
    bridge.write_servo_register(target_id, REG_LOCK, 1, 0)
    time.sleep(0.05)

    print("[*] Setting Voltage Limits: Min = 6.0V (60), Max = 8.5V (85)")
    bridge.write_servo_register(target_id, REG_MIN_VOLT, 1, 60)
    bridge.write_servo_register(target_id, REG_MAX_VOLT, 1, 85)

    print("[*] Setting Unload Condition = 44 (no torque cut on voltage)")
    bridge.write_servo_register(target_id, REG_UNLOAD_COND, 1, 44)

    print(f"[*] Setting Mode = 0 (Position), Acceleration = 0, Max Accel = {MAX_ACCEL}")
    bridge.write_servo_register(target_id, REG_MODE, 1, 0)
    bridge.write_servo_register(target_id, REG_ACCEL, 1, 0)
    bridge.write_servo_register(target_id, REG_MAX_ACCEL, 1, MAX_ACCEL)

    print("[*] Setting PID = (32, 0, 0)")
    bridge.write_servo_register(target_id, REG_P, 1, 32)
    bridge.write_servo_register(target_id, REG_D, 1, 0)
    bridge.write_servo_register(target_id, REG_I, 1, 0)

    if target_id != new_id:
        print(f"[*] Changing ID to {new_id}...")
        bridge.write_servo_register(target_id, REG_ID, 1, new_id)
        time.sleep(0.05)

    print("[*] Locking EEPROM...")
    bridge.write_servo_register(new_id, REG_LOCK, 1, 1)
    
    time.sleep(0.1)

    print("\n" + "="*40)
    print(f" VERIFYING MOTOR CONFIGURATION FOR ID {new_id}")
    print("="*40)
    
    cfg = read_servo_config(bridge, new_id)
    wrong = {key: val for key, val in cfg.items() if key in EXPECTED and val != EXPECTED[key]}
    
    for key, val in cfg.items():
        if val is None:
            print(f" {key:12} : [ERROR] Hardware read timeout")
        else:
            print(f" {key:12} : {val}")
    print("="*40)
    print(" Max Accel only applies after a servo power cycle.")
    return wrong

def setup_one(bridge: BridgeClient, new_id: int, ids: List[int]) -> int:
    """Assembly step: the new servo (ID 1) becomes ``new_id``; steps go from the highest ID down, one servo at a time."""
    if new_id not in ids:
        print(f"[!] ID {new_id} is not one of the robot's servos {ids} (config servos.id_map).")
        return 1
    missing = [i for i in ids if i > new_id and not bridge.ping_servo(i)]
    if missing:
        print(f"[!] Servos set up before not answering: {missing}. Check the chain, or set them up again.")
        return 1
    if new_id != FACTORY_ID and bridge.ping_servo(new_id):
        print(f"[*] A servo already answers at ID {new_id}: rewriting its registers.")
        target = new_id
    elif bridge.ping_servo(FACTORY_ID):
        target = FACTORY_ID
    else:
        print(f"[!] No servo answers at the factory ID {FACTORY_ID}: plug in the new servo.")
        return 1
    wrong = configure_servo(bridge, target, new_id)
    if wrong:
        print(f"[!] Read back different from what was written (lost frame?): {wrong}. Run again.")
        return 1
    print(f"[SUCCESS] Servo {new_id} set up" + (": power-cycle the robot so that Max Accel applies."
                                                if new_id == min(ids) else "."))
    return 0


def main() -> int:
    """Run the interactive prompt for servo provisioning, or reconfigure every servo with ``--all``."""
    parser = argparse.ArgumentParser(description="Set each servo's ID and registers.")
    parser.add_argument("--all", action="store_true",
                        help="rewrite the registers of every servo of the robot, keeping their IDs")
    parser.add_argument("--id", type=int, help="assembly step without a prompt: the new servo at ID 1 gets this ID")
    args = parser.parse_args()

    bridge = BridgeClient()
    if not bridge.connect():
        return 1

    if args.id is not None:
        try:
            return setup_one(bridge, args.id, sorted(load_config()["servos"]["id_map"].values()))
        finally:
            bridge.close()

    if args.all:
        try:
            missing, wrong = [], {}
            for servo_id in sorted(load_config()["servos"]["id_map"].values()):
                if bridge.ping_servo(servo_id):
                    wrong_regs = configure_servo(bridge, servo_id, servo_id)
                    if wrong_regs:
                        wrong[servo_id] = wrong_regs
                else:
                    missing.append(servo_id)
        finally:
            bridge.close()
        if missing:
            print(f"[!] Not answering, not configured: {missing}. Check their cables and run again.")
        if wrong:
            print(f"[!] Read back different from what was written (lost frame?): {wrong}. Run again.")
        if missing or wrong:
            return 1
        print("\n[*] All servos configured. Power-cycle the robot so that Max Accel applies.")
        return 0

    try:
        while True:
            print("\n--------------------------------------------------")
            target_id_str = input("Enter the NEW desired ID (or 'q' to quit): ")
            if target_id_str.lower() == 'q':
                break

            new_id = int(target_id_str)
            if new_id < 1 or new_id > 253:
                print("The ID must be between 1 and 253.")
                continue
            print(f"[*] New target ID: {new_id}")
            input("-> Connect ONLY the target servo, then press ENTER...")
            configure_servo(bridge, 254, new_id)
            print("-> Disconnect the servo before continuing.")
            time.sleep(1)

    except ValueError:
        print("[!] Invalid input. Program aborted.")
    finally:
        bridge.close()
    return 0

if __name__ == "__main__":
    main()