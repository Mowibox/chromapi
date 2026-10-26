# SPDX-License-Identifier: Apache-2.0
"""Battery charge estimate from the INA226 reading: ``V + I·R`` smoothed, voltage -> % table conversion."""

from typing import Any, Dict, Optional

import numpy as np

OK, LOW, CRITICAL, EMPTY = "ok", "low", "critical", "empty"


class Battery:
    """Charge estimate, updated with every power reading."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Build from the battery section of ``config.yaml``."""
        table = np.array(sorted(config["ocv_table"]), dtype=float)  # rows (V_ocv [V], charge [%])
        self._ocv_v, self._ocv_percent = table[:, 0], table[:, 1]
        self.resistance_ohm: float = float(config["resistance_ohm"])  # [Ω]
        self.smoothing_s: float = float(config["smoothing_s"])  # [s]
        self.low_percent: float = float(config["low_percent"])  # [%]
        self.critical_percent: float = float(config["critical_percent"])  # [%]
        self.ocv_v: Optional[float] = None  # [V] smoothed V + I·R; None until the first reading
        self._last_t: Optional[float] = None  # [s]

    def update(self, voltage_v: float, current_a: float, t: float) -> None:
        """Add a reading taken at time ``t`` [s]."""
        if voltage_v <= 0.0:
            return
        ocv = voltage_v + current_a * self.resistance_ohm  # [V]
        if self.ocv_v is None or self._last_t is None:
            self.ocv_v = ocv
        else:
            k = min(1.0, max(0.0, t - self._last_t) / self.smoothing_s)  # low-pass filter
            self.ocv_v += k * (ocv - self.ocv_v)
        self._last_t = t

    @property
    def percent(self) -> Optional[float]:
        """Charge left in %."""
        if self.ocv_v is None:
            return None
        return float(np.interp(self.ocv_v, self._ocv_v, self._ocv_percent))

    @property
    def level(self) -> Optional[str]:
        """``"ok"``, ``"low"`` (yellow), ``"critical"`` (red: no walking), ``"empty"`` (sit down); None if unread."""
        percent = self.percent
        if percent is None:
            return None
        if percent <= 0.0:
            return EMPTY
        if percent < self.critical_percent:
            return CRITICAL
        return LOW if percent < self.low_percent else OK
