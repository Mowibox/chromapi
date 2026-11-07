# SPDX-License-Identifier: Apache-2.0
"""LED ring animations and moods: each mood is a ring animation played with a sound.

:class:`LedRing` renders at ~20 fps in a background thread: a persistent animation (off, idle breathing),
a temporary flash over it (a mood), and a torch (plain white, over everything) for the camera. Works in
simulation too, where the ring material takes the mean colour.

Usage::

    ring = LedRing(robot)
    ring.start()
    ring.set_persistent(render_idle)
    ring.play_mood("happy")       # sound + animation
"""

import colorsys
import logging
import math
import random
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from chromapi.chromapi import AUDIO_DIR
from chromapi.hardware.motherboard_bridge import LED_RING_COUNT

logger = logging.getLogger(__name__)

Color = Tuple[int, int, int]
RenderFn = Callable[[Any, float], None]  # (robot, t [s]) -> sets the ring

BRIGHTNESS = 0.35  # global scale: full brightness is blinding and draws ~1 A from the 5 V rail
IDLE = (182, 5, 252)  # violet-pink, Chromapi's default idle colour
_idle: Color = IDLE
_rainbow = False
RAINBOW_IDLE_S = 4.0  # [s] per turn
TORCH_LEVEL = 200  # white colour for the torch
FPS = 20.0  # [Hz] the animations stay smooth; each frame costs the shared serial bus
TOP_HALF = {14, 15, 16, 17, 0, 1, 2, 3, 4}  # LED 0 is at the top of the ring
RING_CLOCKWISE = True  # LED indices clockwise, seen from the front
BOTTOM_HALF = {5, 6, 7, 8, 9, 10, 11, 12, 13}


def _dim(r: float, g: float, b: float) -> Color:
    return (int(r * BRIGHTNESS), int(g * BRIGHTNESS), int(b * BRIGHTNESS))


def dimmed(color: Color) -> Color:
    """``color`` at the ring's brightness."""
    return _dim(*color)


def idle_color() -> Color:
    """The idle colour."""
    return _rainbow_rgb(2 * math.pi * time.monotonic() / RAINBOW_IDLE_S) if _rainbow else _idle


def _rainbow_rgb(a: float) -> Color:
    """Rainbow colour at angle ``a`` [rad]."""
    return tuple(int(127.5 * (1 + math.sin(a + phase))) for phase in (0.0, 2.094, 4.188))  # type: ignore[return-value]


def _idle_leds(t: float) -> List[Color]:
    """The idle colour of each LED at ``t`` [s]."""
    if _rainbow:
        return [_rainbow_rgb(2 * math.pi * (t / RAINBOW_IDLE_S + i / LED_RING_COUNT)) for i in range(LED_RING_COUNT)]
    return [_idle] * LED_RING_COUNT


def idle_rainbow() -> bool:
    """The idle colour is the rainbow."""
    return _rainbow


def set_idle_color(color: Optional[Color] = None) -> None:
    """Set :func:`idle_color` (None: :data:`IDLE`), ending the rainbow."""
    global _idle, _rainbow
    _idle = IDLE if color is None else tuple(int(min(max(c, 0), 255)) for c in color)  # type: ignore[assignment]
    _rainbow = False


def set_idle_rainbow() -> None:
    """Make the idle colour a slowly turning rainbow."""
    global _rainbow
    _rainbow = True


def accent_color(color: Color) -> Color:
    """A pale neighbour of ``color``, 50 degrees round the hue wheel, on the brighter side."""
    h, _, _ = colorsys.rgb_to_hsv(*(c / 255.0 for c in color))
    candidates = [colorsys.hsv_to_rgb((h + shift) % 1.0, 0.6, 1.0) for shift in (50 / 360, -50 / 360)]
    best = max(candidates, key=lambda rgb: 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    return tuple(int(round(c * 255)) for c in best)  # type: ignore[return-value]


def _from_top(i: int) -> int:
    """Distance of LED ``i`` from the top of the ring, in LEDs."""
    return min(i, LED_RING_COUNT - i)


def render_off(robot: Any, t: float) -> None:
    """All off."""
    robot.set_led_all(0, 0, 0)


LISTENING_LEDS_PER_S = 18.0  # [LED/s] one turn a second


def render_listening(robot: Any, t: float) -> None:
    """Someone listens through the microphone: an accent comet over the dimmed idle colour."""
    head = int(t * LISTENING_LEDS_PER_S) % LED_RING_COUNT
    colors = []
    for i, base in enumerate(_idle_leds(t)):
        d = (head - i) % LED_RING_COUNT
        k = 1.0 - d / 4 * 0.75 if d < 4 else 0.0  # head and a 3-LED tail
        colors.append(_dim(*(m * k + c * 0.3 * (1.0 - k) for m, c in zip(accent_color(base), base))))
    robot.set_led_bulk(colors)


def render_torch(robot: Any, t: float) -> None:
    """Plain white, to light the camera's view."""
    robot.set_led_all(TORCH_LEVEL, TORCH_LEVEL, TORCH_LEVEL)


def render_idle(robot: Any, t: float) -> None:
    """Slow breathing in the idle colour, or the spinning rainbow."""
    if _rainbow:
        robot.set_led_bulk([_dim(*c) for c in _idle_leds(t)])
        return
    k = 0.3 + 0.7 * (math.sin(t * math.pi) + 1.0) / 2.0
    robot.set_led_all(*_dim(*(c * k for c in _idle)))


def render_rainbow(robot: Any, t: float) -> None:
    """Spinning rainbow."""
    robot.set_led_bulk([_dim(*_rainbow_rgb(t * 6.0 + 2 * math.pi * i / LED_RING_COUNT)) for i in range(LED_RING_COUNT)])


def render_sad(robot: Any, t: float) -> None:
    """Blue to cyan towards the bottom, slow pulse."""
    k = 0.5 + 0.5 * (math.sin(t * 0.4 * math.pi) + 1.0) / 2.0
    robot.set_led_bulk([_dim(0, 255 * _from_top(i) / (LED_RING_COUNT / 2) * k, 255 * k) for i in range(LED_RING_COUNT)])


def render_angry(robot: Any, t: float) -> None:
    """Red to orange, slow pulse, a frowning brow (LEDs off) at the top."""
    k = 0.5 + 0.5 * (math.sin(t * 0.4 * math.pi) + 1.0) / 2.0
    brow = {16, 17, 0, 1, 2}
    depth = max(_from_top(i) for i in brow) + 1
    colors: List[Color] = []
    for i in range(LED_RING_COUNT):
        if i in brow:
            colors.append((0, 0, 0))
            continue
        x = (_from_top(i) - depth) / (LED_RING_COUNT / 2 - depth)
        colors.append(_dim(255 * k, 140 * x * k, 0))
    robot.set_led_bulk(colors)


def _pulse_wave(robot: Any, t: float, rgb: Color, every_other: bool = False) -> None:
    k = 0.4 + 0.6 * abs(math.sin(t * 4.0 * math.pi))
    colors: List[Color] = []
    for i in range(LED_RING_COUNT):
        w = k * (0.8 + 0.2 * math.sin(i + t * 10.0))
        colors.append((0, 0, 0) if every_other and i % 2 else _dim(rgb[0] * w, rgb[1] * w, rgb[2] * w))
    robot.set_led_bulk(colors)


def render_yes(robot: Any, t: float) -> None:
    """Green fast pulsing wave."""
    _pulse_wave(robot, t, (0, 255, 0))


def render_no(robot: Any, t: float) -> None:
    """Red fast pulsing wave."""
    _pulse_wave(robot, t, (255, 0, 0))


def render_surprised(robot: Any, t: float) -> None:
    """White fast pulsing wave, every other LED."""
    _pulse_wave(robot, t, (255, 255, 255), every_other=True)


def render_alert(robot: Any, t: float) -> None:
    """Whole ring blinking red."""
    robot.set_led_all(*(_dim(255, 0, 0) if math.sin(t * 6 * math.pi) > 0 else (0, 0, 0)))


def render_scared(robot: Any, t: float) -> None:
    """Pale blue, even and odd LEDs alternating fast."""
    even_on = int(t * 9.0) % 2 == 0
    robot.set_led_bulk([_dim(200, 230, 255) if (i % 2 == 0) == even_on else (0, 0, 0) for i in range(LED_RING_COUNT)])


THINKING_CYCLE_S = 2.2  # [s] per lap
THINKING_DOTS = 4
THINKING_GAP_S = 0.4  # [s] between dots
THINKING_EASE = 0.55  # 0: even speed, 1: stops at the top


def render_thinking(robot: Any, t: float) -> None:
    """A loading spinner: blue to indigo dots chasing round the ring, slower at the top."""
    lit = [0.0] * LED_RING_COUNT
    colors = [(0.0, 0.0, 0.0)] * LED_RING_COUNT
    for j in range(THINKING_DOTS):
        s = ((t - j * THINKING_GAP_S) / THINKING_CYCLE_S) % 1.0
        lap = s - THINKING_EASE * math.sin(2 * math.pi * s) / (2 * math.pi)
        pos = lap * LED_RING_COUNT
        mix = j / (THINKING_DOTS - 1)
        rgb = (75 * mix, 0.0, 255 * (1 - mix) + 130 * mix)
        for i in range(LED_RING_COUNT):
            d = abs((pos - i + LED_RING_COUNT / 2) % LED_RING_COUNT - LED_RING_COUNT / 2)
            k = max(0.0, 1.0 - d)  # glide between two LEDs
            if k > lit[i]:
                lit[i], colors[i] = k, (rgb[0] * k, rgb[1] * k, rgb[2] * k)
    robot.set_led_bulk([_dim(*c) for c in colors])


def render_purring(robot: Any, t: float) -> None:
    """Top half in the idle colour, with a fast purr."""
    k = 0.85 + 0.15 * math.sin(t * 14.0 * math.pi)
    robot.set_led_bulk([_dim(*(c * k for c in base)) if i in TOP_HALF else (0, 0, 0)
                        for i, base in enumerate(_idle_leds(t))])


def render_excited(robot: Any, t: float) -> None:
    """Bright yellow strobe."""
    k = 0.4 + 0.6 * abs(math.sin(t * 10.0 * math.pi))
    robot.set_led_all(*_dim(255 * k, 255 * k, 0))


QUESTION_HOOK = range(-3, 7)  # LED steps clockwise from the top
QUESTION_DOT = 9
QUESTION_WRITE_S = 0.7  # [s]


def _led_at(step: int) -> int:
    """The LED ``step`` places clockwise from the top (LED 0)."""
    return (step if RING_CLOCKWISE else -step) % LED_RING_COUNT


def render_confused(robot: Any, t: float) -> None:
    """A cyan question mark, written then glowing."""
    glow = 0.75 + 0.25 * math.cos(max(t - QUESTION_WRITE_S - 0.2, 0.0) * 2.0 * math.pi)
    colors = [(0, 0, 0)] * LED_RING_COUNT
    for n, step in enumerate(QUESTION_HOOK):
        k = min(max(t / QUESTION_WRITE_S * len(QUESTION_HOOK) - n, 0.0), 1.0)
        colors[_led_at(step)] = _dim(0, 255 * k * glow, 255 * k * glow)
    if t >= QUESTION_WRITE_S + 0.2:
        colors[_led_at(QUESTION_DOT)] = _dim(0, 255 * glow, 255 * glow)
    robot.set_led_bulk(colors)


def render_squeak(robot: Any, t: float) -> None:
    """Idle colour to its accent, 4 beats per squeak."""
    mix = 0.5 + 0.5 * math.sin((t % 0.342) / 0.342 * 4 * 2 * math.pi)
    robot.set_led_bulk([_dim(*(a + (b - a) * mix for a, b in zip(base, accent_color(base)))) for base in _idle_leds(t)])


def render_joy(robot: Any, t: float) -> None:
    """A pink smile: the top half lit."""
    robot.set_led_bulk([_dim(220, 0, 220) if i in TOP_HALF else (0, 0, 0) for i in range(LED_RING_COUNT)])


def render_dizzy(robot: Any, t: float) -> None:
    """A spinning yellow comet, 0.4 turn/s like the chassis in the dizzy gesture."""
    head = int(t * 0.4 * LED_RING_COUNT) % LED_RING_COUNT
    colors = []
    for i in range(LED_RING_COUNT):
        d = (head - i) % LED_RING_COUNT
        k = 1.0 - d / 5 * 0.8 if d < 5 else 0.0
        colors.append(_dim(255 * k, 255 * k, 0))
    robot.set_led_bulk(colors)


def render_tired(robot: Any, t: float) -> None:
    """Low orange eyelids (bottom half), blinking now and then."""
    blink = random.random() < 0.05
    robot.set_led_bulk([_dim(255, 100, 0) if i in BOTTOM_HALF and not blink else (0, 0, 0) for i in range(LED_RING_COUNT)])


BATTERY_FLASH_S = 3.0  # [s] how long the ring shows the charge


def battery_color(percent: float) -> Color:
    """The motherboard status LED's colour for ``percent`` (firmware ``Battery_Led_Update``, same thresholds).

    Red (10 %) -> orange (20 %) -> yellow (30 %, "low") -> green (60 %) -> green-cyan (100 %), on a 0-50 scale there.
    """
    p = min(max(percent, 0.0), 100.0)
    if p >= 60.0:
        rgb = (0.0, 50.0, 50.0 * (p - 60.0) / 40.0)
    elif p >= 30.0:
        rgb = (50.0 * (60.0 - p) / 30.0, 50.0, 0.0)
    elif p >= 20.0:
        rgb = (50.0, 15.0 + 35.0 * (p - 20.0) / 10.0, 0.0)
    else:
        rgb = (50.0, 15.0 * max(p - 10.0, 0.0) / 10.0, 0.0)
    return _dim(*(c * 255.0 / 50.0 for c in rgb))


def render_battery(percent: float) -> RenderFn:
    """The charge on the ring: as many LEDs as the charge (from the top), in the status LED's colour.

    Under 10 % they blink at 1 Hz, as the status LED does. At least one LED stays lit, so 0 % still shows something.
    """
    lit = max(1, round(LED_RING_COUNT * min(max(percent, 0.0), 100.0) / 100.0))
    color = battery_color(percent)

    def render(robot: Any, t: float) -> None:
        on = percent >= 10.0 or int(2.0 * t) % 2 == 0
        robot.set_led_bulk([color if on and i < lit else (0, 0, 0) for i in range(LED_RING_COUNT)])

    return render


@dataclass(frozen=True)
class Mood:
    """A sound played with a ring animation."""

    name: str
    render: RenderFn
    sound: str  # file in AUDIO_DIR
    emoji: str  # for the web app
    min_s: float = 0.0  # [s] minimum animation length
    gain: float = 1.0  # sound volume factor, for a sound quieter on the small speaker than the others


MOODS: List[Mood] = [
    Mood("scream", render_rainbow, "chromapi_scream.wav", "📣"),
    Mood("happy", render_rainbow, "happy.wav", "😄"),
    Mood("sad", render_sad, "sad.wav", "😢"),
    Mood("angry", render_angry, "upset.wav", "😠"),
    Mood("surprised", render_surprised, "surprised.wav", "😮"),
    Mood("joy", render_joy, "greeting.wav", "🥰"),
    Mood("dizzy", render_dizzy, "dizzy.wav", "😵"),
    Mood("tired", render_tired, "sleepy.wav", "😴"),
    Mood("yes", render_yes, "yes.wav", "👍"),
    Mood("no", render_no, "no.wav", "👎"),
    Mood("alert", render_alert, "alert.wav", "🚨"),
    Mood("scared", render_scared, "scared.wav", "😨"),
    Mood("thinking", render_thinking, "thinking.wav", "🤔", min_s=2 * THINKING_CYCLE_S + 0.4),
    Mood("purring", render_purring, "purring.wav", "😺", gain=3.5),
    Mood("excited", render_excited, "excited.wav", "🤩"),
    Mood("confused", render_confused, "confused.wav", "❓", min_s=2.0),
    Mood("squeak", render_squeak, "squeak.wav", "🐭"),
]
MOODS_BY_NAME: Dict[str, Mood] = {mood.name: mood for mood in MOODS}
PAD_MOOD_COLORS: Dict[str, Color] = {
    "scream": (255, 0, 80), "happy": (255, 210, 0), "sad": (0, 100, 255), "angry": (255, 140, 0),
    "surprised": (255, 255, 255), "joy": (220, 0, 220), "dizzy": (255, 255, 0), "tired": (255, 120, 0),
    "yes": (0, 255, 0), "no": (255, 0, 0), "alert": (255, 51, 0), "scared": (200, 230, 255),
    "thinking": (75, 0, 130), "purring": (255, 105, 180), "excited": (255, 255, 0), "confused": (0, 229, 255),
    "squeak": (180, 180, 180),
}
MIN_FLASH_S = 1.2  # [s] a short sound still gets a visible animation


def sound_duration_s(path: Path) -> float:
    """Length of a .wav [s], 0 if it cannot be read."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (OSError, wave.Error):
        return 0.0


class _OnChange:
    """Forwards a frame to the robot only when it differs from the last one: a still ring costs nothing."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self.last: Any = None

    def set_led_all(self, r: int, g: int, b: int) -> None:
        if self.last != (r, g, b):
            self.robot.set_led_all(r, g, b)
            self.last = (r, g, b)

    def set_led_bulk(self, colors: List[Color]) -> None:
        if self.last != colors:
            self.robot.set_led_bulk(colors)
            self.last = colors


class LedRing:
    """Background ring renderer: persistent animation, temporary flash over it, the listening sign over that, torch over everything."""

    def __init__(self, robot: Any) -> None:
        """No thread until :meth:`start`."""
        self.robot = robot
        self._out = _OnChange(robot)
        self.torch = False
        self.listening = False  # Chromapi's microphone is streamed to a page: shown, whatever else plays
        self._persistent: RenderFn = render_off
        self._flash: Optional[Tuple[RenderFn, float]] = None  # (animation, until [s])
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start rendering."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="chromapi-led", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop rendering and turn the ring off."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        render_off(self.robot, 0.0)  # always sent, whatever the last frame was

    def set_persistent(self, render: RenderFn) -> None:
        """The animation shown when nothing else plays."""
        self._persistent = render

    def flash(self, render: RenderFn, duration_s: float) -> None:
        """Show ``render`` for ``duration_s``, then go back to the persistent animation."""
        self._flash = (render, time.monotonic() + duration_s)

    def play_mood(self, name: str, duration_s: float = 0.0) -> None:
        """A mood's sound and its animation, for as long as the sound, or ``duration_s`` [s] (a gesture) if longer."""
        mood = MOODS_BY_NAME[name]
        path = AUDIO_DIR / mood.sound
        self.flash(mood.render, max(MIN_FLASH_S, mood.min_s, sound_duration_s(path), duration_s))
        self.robot.play_sound(path, gain=mood.gain)

    def _loop(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            flash = self._flash
            if flash is not None and now >= flash[1]:
                self._flash = flash = None
            render = (render_torch if self.torch else render_listening if self.listening
                      else flash[0] if flash else self._persistent)
            try:
                render(self._out, now - t0)
            except Exception:  # a lost frame on the bus: the next one will do
                logger.debug("LED frame failed", exc_info=True)
            self._stop.wait(1.0 / FPS)


#: The ring after a crash (``chromapi led error``).
ERROR_RGB: Color = (255, 0, 0)


def error_main() -> int:
    """Light the ring in :data:`ERROR_RGB`."""
    from chromapi.hardware.motherboard_bridge import BridgeClient

    bridge = BridgeClient()
    if not bridge.connect():
        return 1
    try:
        return 0 if bridge.set_led_color_all(*_dim(*ERROR_RGB)) else 1
    finally:
        bridge.close()
