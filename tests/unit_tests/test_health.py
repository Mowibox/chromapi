# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chromapi health``."""

from chromapi.hardware.motherboard_bridge import BRIDGE_PROTOCOL_VERSION
from chromapi.utils.health import FAIL, OK, WARN, check_calibration, check_robot

ID_MAP = {"tl_1": 1, "tl_2": 2, "tl_3": 3}


class FakeBridge:
    """Connected bridge whose servo bus holds ``bus_ids``."""

    def __init__(self, bus_ids, voltage=7.8, acc=(0.0, 0.0, 9.8)):
        """Robot at rest on a charged battery unless told otherwise."""
        self.bus_ids, self.voltage, self.acc = set(bus_ids), voltage, list(acc)

    def ping(self):
        """STM32 always answers."""
        return True

    def get_version(self):
        """Current firmware."""
        return BRIDGE_PROTOCOL_VERSION, (1, 1, 0)

    def get_state(self, timeout):
        """Snapshot where only the servos on the bus answer."""
        return {
            "power": {"voltage_V": self.voltage, "current_A": 0.5},
            "imu": {"acc_mps2": self.acc},
            "servos": [{"id": i, "fresh": i in self.bus_ids, "temp_C": 30} for i in range(1, 13)],
        }

    def ping_servo(self, servo_id):
        """Answer for ids on the bus."""
        return servo_id in self.bus_ids


def levels(results):
    """Map each check to its level."""
    return {check: level for level, check, _ in results}


def test_healthy_robot():
    """All servos, good battery, still IMU: nothing to report."""
    assert set(levels(check_robot(FakeBridge({1, 2, 3}), ID_MAP)).values()) == {OK}


def test_servo_on_factory_id_is_named():
    """A missing servo triggers the bus scan, which finds it on another id."""
    results = check_robot(FakeBridge({1, 3, 200}), ID_MAP)
    assert levels(results)["servo tl_2"] == FAIL
    assert any(check == "servo bus" and "unexpected id 200" in detail for _, check, detail in results)


def test_low_battery_and_moving_imu():
    """Low battery fails; gravity off only warns (the robot may be moving)."""
    found = levels(check_robot(FakeBridge({1, 2, 3}, voltage=5.5, acc=(0.0, 0.0, 14.0)), ID_MAP))
    assert found["battery"] == FAIL
    assert found["IMU"] == WARN


def test_calibration_defaults_warn(tmp_path):
    """Each calibration not measured yet is a warning."""
    path = tmp_path / "calibration.yaml"
    assert set(levels(check_calibration(path)).values()) == {WARN}
    path.write_text("imu:\n  mount_offset_wxyz: [1, 0, 0, 0]\n")
    assert levels(check_calibration(path)) == {"servo centers": WARN, "IMU mount": OK}
