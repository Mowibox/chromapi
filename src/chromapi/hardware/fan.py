# SPDX-License-Identifier: Apache-2.0
"""Cooling fan on the Raspberry Pi's hardware PWM, through ``/sys/class/pwm``.

``deploy/chromapi-fan@.service`` runs :func:`main` for as long as the Pi is on, whatever else runs: it holds the
CPU near :data:`TARGET_TEMP_C`. Any process imposes a speed with
:func:`request`, and gives it back to the automatic mode with ``request(None)``.

Usage::

    python3 -m chromapi.hardware.fan    # what the service runs
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

PWM_CHIP = 0
PWM_CHANNEL = 1
PWM_FREQUENCY_HZ = 25_000  # [Hz]
TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
OVERRIDE_PATH = Path("/run/chromapi/fan")  # the service's RuntimeDirectory: gone when it stops
# [°C] the Pi 4 throttles from 80 °C. Measured (contact_runs, 2026-10-25/26): ~55 °C standing without the
# fan, 42-46 °C trotting 37 min with fan, LEDs, audio and camera at full fan; the fan costs no measurable power.
TARGET_TEMP_C = 50.0
FULL_SPEED_TEMP_C = 75.0  # [°C] full speed whatever was requested
KP = 5.0  # [%.°C⁻¹]
KI = 0.1  # [%.°C⁻¹.s⁻¹]
MIN_RUN_PERCENT = 30.0  # [%] slower, the fan may stall: the automatic mode stops it instead
PERIOD_S = 1.0  # [s] also the delay before a request applies
EXPORT_TIMEOUT_S = 2.0  # [s] for udev to give the freshly exported channel to the gpio group


def _pwm_dir(chip: int = PWM_CHIP, channel: int = PWM_CHANNEL) -> Path:
    return Path(f"/sys/class/pwm/pwmchip{chip}/pwm{channel}")


def cpu_temp_c() -> Optional[float]:
    """CPU temperature [°C], None off a Pi."""
    try:
        return int(TEMP_PATH.read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def request(percent: Optional[float]) -> bool:
    """Impose a fan speed [%] on the fan service, or None for the automatic mode. False if the service is off."""
    try:
        OVERRIDE_PATH.write_text("auto" if percent is None else f"{min(max(float(percent), 0.0), 100.0):g}")
    except OSError:
        logger.warning("Fan service not running (no %s): request ignored", OVERRIDE_PATH.parent)
        return False
    return True


def requested() -> Optional[float]:
    """The speed [%] imposed through :func:`request`, None in the automatic mode."""
    try:
        return float(OVERRIDE_PATH.read_text())
    except (OSError, ValueError):  # no file, or "auto"
        return None


def status() -> Optional[Tuple[float, bool]]:
    """(speed [%], automatic) as the service runs the fan, None if it is not running."""
    if not OVERRIDE_PATH.parent.is_dir():
        return None
    try:
        duty, period = (int((_pwm_dir() / name).read_text()) for name in ("duty_cycle", "period"))
    except (OSError, ValueError):
        return None
    return 100.0 * duty / period, requested() is None


class Fan:
    """One PWM channel driving the fan, and the automatic mode's state."""

    def __init__(self, chip: int = PWM_CHIP, channel: int = PWM_CHANNEL, frequency_hz: int = PWM_FREQUENCY_HZ,
                 target_c: float = TARGET_TEMP_C) -> None:
        """Export and configure the channel; :attr:`available` tells whether it worked."""
        self._pwm = _pwm_dir(chip, channel)
        self._period_ns = int(1e9 / frequency_hz)  # [ns]
        self.available = False
        self.percent = 0.0  # [%]
        self.target_c = target_c  # [°C]
        self._integral = 0.0  # [%]
        deadline = time.monotonic() + EXPORT_TIMEOUT_S
        while True:
            try:
                if not self._pwm.exists():
                    (self._pwm.parent / "export").write_text(str(channel))
                (self._pwm / "period").write_text(str(self._period_ns))
                (self._pwm / "duty_cycle").write_text("0")
                (self._pwm / "enable").write_text("1")
                break
            except OSError as exc:
                # Right after the export, udev has not yet handed the new channel's files to the gpio group.
                if not isinstance(exc, PermissionError) or time.monotonic() > deadline:
                    logger.warning("Fan PWM unavailable (%s): no fan control", exc)
                    return
                time.sleep(0.1)
        self.available = True

    def set_percent(self, percent: float) -> None:
        """Fan speed [%], 0 to 100."""
        if not self.available:
            return
        self.percent = max(0.0, min(100.0, float(percent)))
        try:
            (self._pwm / "duty_cycle").write_text(str(int(self._period_ns * self.percent / 100.0)))
        except OSError:
            logger.exception("Fan PWM write failed: no fan control from now on")
            self.available = False

    def resume_auto(self) -> None:
        """Start the automatic mode again from the current speed (no jump)."""
        self._integral = self.percent

    def auto_percent(self, temp_c: float, dt: float) -> float:
        """PI step on the CPU temperature: the speed [%] for ``temp_c`` [°C], ``dt`` [s] after the previous step."""
        error = temp_c - self.target_c  # [°C]
        self._integral = min(max(self._integral + KI * error * dt, 0.0), 100.0)  # clamped: no windup
        percent = min(max(KP * error + self._integral, 0.0), 100.0)
        if percent < MIN_RUN_PERCENT:  # running: hold the minimum down to half of it, then stop
            percent = MIN_RUN_PERCENT if self.percent > 0.0 and percent > MIN_RUN_PERCENT / 2 else 0.0
        return percent


def main() -> int:
    """Run the fan until the service stops; 1 if the PWM cannot be driven (systemd retries)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fan = Fan()
    manual: Optional[float] = None
    while fan.available:
        temp = cpu_temp_c()
        wanted = requested()
        if wanted != manual:
            logger.info("Fan %s", "automatic" if wanted is None else f"at {wanted:g} % (requested)")
            if wanted is None:
                fan.resume_auto()
            manual = wanted
        if temp is None or temp >= FULL_SPEED_TEMP_C:
            fan.set_percent(100.0)
        elif manual is not None:
            fan.set_percent(manual)
        else:
            fan.set_percent(fan.auto_percent(temp, PERIOD_S))
        time.sleep(PERIOD_S)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
