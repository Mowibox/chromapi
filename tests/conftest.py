"""
Root conftest.py — pytest configuration for the chromapi test suite.

Hardware tests are guarded by the `hardware` marker.
They are excluded from the default testpaths and must be run explicitly:

    # On the Raspberry Pi with the robot assembled:
    pytest tests/hardware_tests/ -v

    # Run a single hardware test:
    pytest tests/hardware_tests/test_05_imu_data.py -v

    # CI never runs hardware tests — they are not in testpaths.
"""

import pytest

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "hardware: requires physical hardware (RPi + peripherals). Never runs in CI."
    )