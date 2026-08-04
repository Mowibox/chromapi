"""Hardware Diagnostic Tool.

Provides CLI utilities to test the UART bridge, power monitor, IMU, audio subsystem,
and servomotors sequentially.
"""

import logging
import os

from chromapi.hardware.motherboard_bridge import BridgeClient


def run_diagnostics() -> None:
    """Execute the full suite of hardware diagnostic checks."""
    print("====================================================")
    print(" CHROMAPI HARDWARE DIAGNOSTIC TOOL")
    print("====================================================\n")

    bridge = BridgeClient(port='/dev/ttyS0', baudrate=1000000)
    
    print("[*] Testing UART Bridge Connection...")
    if not bridge.connect():
        print("    [FAIL] Cannot open serial port /dev/ttyS0.")
        return
    
    if bridge.ping():
        print("    [PASS] STM32 Bridge is ONLINE.")
    else:
        print("    [FAIL] STM32 Bridge timeout. Check RX/TX lines and power.")
        return

    print("\n[*] Testing Power Monitor (INA226)...")
    power = bridge.get_power()
    if power:
        print(f"    [PASS] Bus Voltage: {power[0]:.2f} V | Current: {power[1]:.2f} A")
        if power[0] < 6.0:
            print("    [WARN] Battery voltage is critically low (< 6.0V)!")
    else:
        print("    [FAIL] INA226 communication failed. Check I2C bus on PCB.")

    print("\n[*] Testing IMU & Switches via Feedback Stream...")
    state = bridge.get_state()
    if state:
        acc = state['imu']['acc_mps2']
        if acc == [0.0, 0.0, 0.0]:
            print("    [FAIL] IMU Accel reads all zeros. SPI communication dead?")
        else:
            print(f"    [PASS] IMU active. Z-Accel: {acc[2]:.2f} m/s^2")
        print(f"    [INFO] Switches state: {state['switches']}")
    else:
        print("    [FAIL] Could not retrieve global state frame (121 bytes).")

    print("\n[*] Testing Servomotor Bus (Feetech STS3215)...")
    if state:
        active_servos = [s for s in state['servos'] if s['volt_V'] > 0.1]
        active_ids = [s['id'] for s in active_servos]
        missing_ids = [i for i in range(1, 13) if i not in active_ids]
        
        print(f"    [INFO] Detected {len(active_servos)}/12 servos.")
        if missing_ids:
            print(f"    [FAIL] Missing Servo IDs: {missing_ids}. Check cables or ID assignment.")
        else:
            print("    [PASS] All 12 servos connected and responding.")
            
        for s in active_servos:
            if s['temp_C'] > 60:
                print(f"    [WARN] Servo ID {s['id']:02d} is overheating ({s['temp_C']}°C)!")

    print("\n[*] Testing Audio Subsystem (I2S/ALSA)...")
    aplay_output = os.popen("aplay -l").read()
    if "sndrpigooglevoi" in aplay_output.lower():
        print("    [PASS] Custom audio soundcard detected by ALSA.")
    else:
        print("    [FAIL] Audio soundcard NOT detected. Check Device Tree Overlay (.dtbo) and I2S wiring.")

    bridge.set_led_color_all(0, 0, 0)
    bridge.close()
    print("\n====================================================")
    print(" DIAGNOSTIC COMPLETE")
    print("====================================================")

if __name__ == "__main__":
    run_diagnostics()