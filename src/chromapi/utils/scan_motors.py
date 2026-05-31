"""Motor Scanning Utility.

Pings the servo bus to discover and list all connected Feetech servomotors.
"""

from chromapi.hardware.motherboard_bridge import BridgeClient


def scan_bus() -> None:
    """Scan the bus and print the IDs of responding servomotors."""
    print("Scanning motors on the bus...")
    bridge = BridgeClient()
    
    if not bridge.connect():
        return
        
    found = 0
    for i in range(1, 253):
        if bridge.ping_servo(i):
            print(f" [✓] Found servo with ID: {i}")
            found += 1
            
    print(f"Scan completed. {found} servo(s) found.")
    bridge.close()

if __name__ == "__main__":
    scan_bus()