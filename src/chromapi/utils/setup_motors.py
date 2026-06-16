"""Motor Setup and Full Provisioning Tool."""

import time
from typing import Dict, Optional

from chromapi.hardware.motherboard_bridge import BridgeClient

# Feetech STS3215 Register Map
REG_ID = 5
REG_MAX_VOLT = 14
REG_MIN_VOLT = 15
REG_P = 21
REG_D = 22
REG_I = 23
REG_MODE = 33
REG_ACCEL = 41
REG_LOCK = 55

def read_servo_config(bridge: BridgeClient, servo_id: int) -> Dict[str, Optional[int]]:
    """Read back current configuration registers directly from the hardware."""
    return {
        "P": bridge.read_servo_register(servo_id, REG_P, 1),
        "I": bridge.read_servo_register(servo_id, REG_I, 1),
        "D": bridge.read_servo_register(servo_id, REG_D, 1),
        "Acceleration": bridge.read_servo_register(servo_id, REG_ACCEL, 1),
        "Mode": bridge.read_servo_register(servo_id, REG_MODE, 1),
        "Max Voltage": bridge.read_servo_register(servo_id, REG_MAX_VOLT, 1),
        "Min Voltage": bridge.read_servo_register(servo_id, REG_MIN_VOLT, 1)
    }

def configure_servo(bridge: BridgeClient, target_id: int, new_id: int) -> None:
    """Unlock, configure, assign ID, and verify parameters."""
    print(f"[*] Unlocking EEPROM for ID {target_id}...")
    bridge.write_servo_register(target_id, REG_LOCK, 1, 0)
    time.sleep(0.05)

    print("[*] Setting Voltage Limits: Min = 6.6V (66), Max = 8.4V (84)")
    bridge.write_servo_register(target_id, REG_MIN_VOLT, 1, 66)
    bridge.write_servo_register(target_id, REG_MAX_VOLT, 1, 84)

    print("[*] Setting Mode = 0 (Position), Acceleration = 0 (Instant)")
    bridge.write_servo_register(target_id, REG_MODE, 1, 0)
    bridge.write_servo_register(target_id, REG_ACCEL, 1, 0)

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
    
    for key, val in cfg.items():
        if val is None:
            print(f" {key:12} : [ERROR] Hardware read timeout")
        else:
            print(f" {key:12} : {val}")
    print("="*40)

def main() -> None:
    """Run the interactive prompt for servo provisioning."""
    bridge = BridgeClient()
    if not bridge.connect():
        return

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

if __name__ == "__main__":
    main()