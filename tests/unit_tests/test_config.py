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


def test_tibia_folds_with_the_femur_direction(monkeypatch):
    """Raw servo direction x config sign = femur direction; lowering femur -> tibia folded down."""
    from chromapi.utils import calibrate_servo_centers as cal

    moves = []
    monkeypatch.setattr(cal, "_move_to_pose", lambda bridge, servos, pose: moves.append(pose))
    servos = {"sign": {"tl_2": -1, "tr_2": 1}}
    for femur, raw_direction, expected in [
        ("tl_2", -1, cal.TIBIA_FEMUR_UP_RAD),  # sign -1: raw steps down = femur angle up
        ("tl_2", +1, cal.TIBIA_FEMUR_DOWN_RAD),
        ("tr_2", -1, cal.TIBIA_FEMUR_DOWN_RAD),
    ]:
        cal._tibia_follows_femur(None, servos, femur)(raw_direction)
        assert moves[-1] == {femur[:-1] + "3": expected}
