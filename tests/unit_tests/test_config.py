# SPDX-License-Identifier: Apache-2.0
"""Package defaults + per-robot calibration file."""

import yaml

from chromapi.chromapi import DEFAULT_CONFIG_PATH, load_config, save_calibration


def test_calibration_overrides_only_what_was_measured(tmp_path):
    calib = tmp_path / "chromapi" / "calibration.yaml"
    defaults_before = DEFAULT_CONFIG_PATH.read_text()

    assert load_config(calibration_path=calib)["servos"]["zero_offset_steps"]["tl_2"] == 0  # no file yet

    save_calibration({"servos": {"zero_offset_steps": {"tl_2": 17}}}, calib)
    save_calibration({"imu": {"mount_offset_wxyz": [0.9, 0.1, 0.0, 0.0]}}, calib)
    save_calibration({"servos": {"zero_offset_steps": {"br_3": -4}}}, calib)

    config = load_config(calibration_path=calib)
    offsets = config["servos"]["zero_offset_steps"]
    assert (offsets["tl_2"], offsets["br_3"], offsets["tl_1"]) == (17, -4, 0)
    assert config["imu"]["mount_offset_wxyz"] == [0.9, 0.1, 0.0, 0.0]
    assert config["servos"]["id_map"]["tl_1"] == 1  # untouched defaults still come through

    assert yaml.safe_load(calib.read_text()) == {  # the file holds the measurements and nothing else
        "servos": {"zero_offset_steps": {"tl_2": 17, "br_3": -4}},
        "imu": {"mount_offset_wxyz": [0.9, 0.1, 0.0, 0.0]},
    }
    assert DEFAULT_CONFIG_PATH.read_text() == defaults_before
    assert not list(calib.parent.glob("*.tmp"))
