# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shipped firmware and the update check."""

import re

from chromapi.utils import flash


def test_the_sdk_ships_a_firmware_with_its_version():
    """An Intel HEX image (every record ':' + hex, ends with the EOF record) and a major.minor.patch version."""
    lines = flash.SHIPPED_HEX.read_text().split()
    assert all(re.fullmatch(r":[0-9A-F]+", line) for line in lines) and lines[-1] == ":00000001FF"
    assert re.fullmatch(r"\d+\.\d+\.\d+", flash.shipped_version() or "")


def test_update_available():
    """Numeric comparison (1.10 > 1.9); no answer to GET_VERSION means older than 2.0.0; nothing shipped, nothing offered."""
    assert flash.update_available("1.1.0", "2.0.0") and flash.update_available("2.9.0", "2.10.0")
    assert flash.update_available(None, "2.0.0")
    assert not flash.update_available("2.0.0", "2.0.0") and not flash.update_available("2.1.0", "2.0.0")
    assert not flash.update_available("1.0.0", None)
