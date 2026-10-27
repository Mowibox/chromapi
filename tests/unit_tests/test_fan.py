# SPDX-License-Identifier: Apache-2.0
"""Fan service: the automatic mode's PI step and the speed requests, off the Pi (no PWM)."""

from pathlib import Path

import pytest

from chromapi.hardware import fan
from chromapi.hardware.fan import MIN_RUN_PERCENT, PERIOD_S, TARGET_TEMP_C, Fan


def test_auto_percent() -> None:
    f = Fan(chip=999)  # no such PWM chip: no write
    assert not f.available
    assert f.auto_percent(TARGET_TEMP_C - 10.0, PERIOD_S) == 0.0  # cool: off
    hot = [f.auto_percent(TARGET_TEMP_C + 7.0, PERIOD_S) for _ in range(30)]
    assert MIN_RUN_PERCENT <= hot[0] < hot[-1] <= 100.0  # hot: starts, then speeds up while it stays hot
    f.percent, f._integral = MIN_RUN_PERCENT, 20.0  # running, near the target for a while
    assert f.auto_percent(TARGET_TEMP_C + 1.0, PERIOD_S) == MIN_RUN_PERCENT  # held at the minimum
    assert f.auto_percent(TARGET_TEMP_C - 5.0, PERIOD_S) == 0.0  # well below: stops


def test_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fan, "OVERRIDE_PATH", tmp_path / "missing" / "fan")
    assert not fan.request(40.0)  # service not running
    monkeypatch.setattr(fan, "OVERRIDE_PATH", tmp_path / "fan")
    assert fan.requested() is None  # nothing requested: automatic
    assert fan.request(140.0) and fan.requested() == 100.0
    assert fan.request(None) and fan.requested() is None
