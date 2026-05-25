import time
from chromapi.hardware.motherboard_bridge import BridgeClient

def main():
    print("==================================================")
    print(" FEETECH STS3215 ID CONFIGURATION TOOL")
    print("==================================================\n")
    print("[WARNING] Connect ONLY ONE servo at a time!")

    bridge = BridgeClient()
    if not bridge.connect():
        return

    while True:
        try:
            print("\n--------------------------------------------------")
            target_id_str = input("Enter the NEW desired ID or 'q' to quit: ")
            if target_id_str.lower() == 'q':
                break

            new_id = int(target_id_str)
            if new_id < 1 or new_id > 253:
                print("The ID must be between 1 and 253.")
                continue

            input("-> Connect ONLY the target servo, then press ENTER...")
            print(f"Writing ID {new_id}...")
            success = bridge.set_servo_id(254, new_id)

            if success:
                print(f" [SUCCESS] The servo now responds to ID {new_id}.")
                print(" You can disconnect it and move to the next one.")
            else:
                print(" [FAILURE] Timeout or error. Check the power supply and data cable.")

            time.sleep(0.5)

        except ValueError:
            print("Invalid input.")

    bridge.close()

if __name__ == "__main__":
    main()