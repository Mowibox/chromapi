# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the battery charge estimate."""

import pytest

from chromapi.hardware.battery import CRITICAL, EMPTY, LOW, OK, Battery

CONFIG = {
    "ocv_table": [[8.0, 100], [7.0, 0]],
    "resistance_ohm": 0.25,
    "smoothing_s": 10.0,
    "low_percent": 30.0,
    "critical_percent": 15.0,
}


def test_unread_until_first_reading():
    """No reading (or the INA226's 0 V before its first conversion) is not 0 %."""
    battery = Battery(CONFIG)
    battery.update(0.0, 0.0, 0.0)
    assert battery.percent is None and battery.level is None


def test_load_drop_compensated():
    """7.25 V under 1 A reads as 7.5 V without load: 50 %, whatever the current."""
    battery = Battery(CONFIG)
    battery.update(7.25, 1.0, 0.0)
    assert battery.percent == pytest.approx(50.0)
    battery.update(7.5, 0.0, 1.0)
    assert battery.percent == pytest.approx(50.0)


def test_smoothing_and_levels():
    """A step moves the estimate by dt/smoothing_s per reading; table ends clamp to 0-100 %."""
    battery = Battery(CONFIG)
    battery.update(7.9, 0.0, 0.0)
    assert battery.level == OK
    battery.update(6.9, 0.0, 1.0)  # 1 s of a 10 s smoothing: 10 % of the step
    assert battery.ocv_v == pytest.approx(7.8)
    for ocv, level in ((7.2, LOW), (7.1, CRITICAL), (6.5, EMPTY)):
        battery.update(ocv, 0.0, 100.0 + ocv)
        battery.update(ocv, 0.0, 200.0 + ocv)  # > smoothing_s later: settled
        assert battery.level == level
    assert battery.percent == 0.0
