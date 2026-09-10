"""Abstract human input source - :class:`InputSource` + :class:`UserInput`.

One small interface so a teleop loop can swap gamepad for
keyboard or anything else without touching the rest of the script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set


@dataclass
class UserInput:
    """One tick's worth of decoded human input - robot-frame-agnostic on purpose.

    ``forward``/``lateral``/``turn`` are plain human-intuitive intents, not
    ``Chromapi.walk()``'s own ``vx``/``vy``/``wz`` (whose sign/axis convention is empirically
    quirky, per that method's own docstring) - the caller maps these onto ``walk()``'s actual
    arguments and picks the speed scale, this class only decodes "what does the human want".
    """

    forward: float = 0.0
    """Normalized [-1, 1] - positive means "move forward"."""
    lateral: float = 0.0
    """Normalized [-1, 1] - positive means "strafe right"."""
    turn: float = 0.0
    """Normalized [-1, 1] - positive means "turn left."""
    vertical: float = 0.0
    """Normalized [-1, 1] - positive means "raise"."""
    pressed: Set[str] = field(default_factory=set)
    """Used for one-shot actions (toggle a mode, fire a sound, wake up/rest)."""
    held: Set[str] = field(default_factory=set)
    """Use this for continuous hold-to-repeat actions (raise/lower height, adjust a speed scale)."""


class InputSource(ABC):
    """Abstract interface for human input."""

    def start(self) -> None:
        """Start the input source (e.g. launch a background thread). No-op by default."""

    def stop(self) -> None:
        """Stop the input source and release resources. No-op by default."""

    @abstractmethod
    def read(self) -> UserInput:
        """Return the current input state. Must be non-blocking."""
