# SPDX-License-Identifier: Apache-2.0
"""Recorded moves: trajectories stored as files and played back.

Each sample of a recorded move holds either:

- ``body``: the chassis pose over the planted feet (:class:`RecordedMove`);
- ``leg``: offsets of one front leg's 3 joints from where it starts (:class:`RecordedLegMove`), played after
  shifting the weight onto the other three feet.

A library is a folder (the bundled ``assets/moves`` by default) or a Hugging Face dataset (``"<account>/<name>"``,
``pip install 'chromapi[hub]'``), downloaded once then read from the cache, so it can be used offline.

Usage::

    body = BalanceMove(dt=1.0 / robot.loop_hz)
    moves = RecordedMoves()  # or RecordedMoves("Mowibox/chromaposes")
    robot.play_move(moves.get("yes", body))
    robot.play_move(moves.get("greeting", body, leg="tr"))
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from chromapi.chromapi import MotorCommand, Move, RobotState
from chromapi.locomotion.balance import BalanceMove

logger = logging.getLogger(__name__)

MOVES_DIR = Path(__file__).resolve().parent.parent / "assets" / "moves"
SETTLE_S = 0.4  # [s] back to the start pose before done

# What check_move lets through (the bundled moves stay well within: 3 s, 0.22 m/s, 1.4 rad/s, leg 4.7 rad/s)
MAX_FILE_BYTES = 256_000
MAX_DURATION_S = 10.0  # [s]
MIN_STEP_S = 0.005  # [s] between samples
MAX_SHIFT_M = 0.03  # [m] |x|, |y|, |z| of a body move
MAX_TILT_RAD = 0.2  # [rad] |roll|, |pitch|, |yaw|
MAX_SHIFT_SPEED_MPS = 0.3  # [m.s⁻¹]
MAX_TILT_SPEED_RAD_S = 2.0  # [rad.s⁻¹]
MAX_LEG_RAD = 1.2  # [rad] leg joint offsets; the joint limits are applied on top while playing
MAX_LEG_SPEED_RAD_S = 6.0  # [rad.s⁻¹]
_KEYS = {"description", "time", "set_target_data", "default_leg", "mood"}

# Weight shift onto the 3 other feet, for a leg move
EMOTE_COM_MARGIN_M = 0.03  # [m] CoM inside the support triangle
EMOTE_MAX_SHIFT_M = 0.05  # [m] foot workspace limit
EMOTE_SHIFT_S = 0.5  # [s]
EMOTE_LAND_S = 0.3  # [s] 4 feet down before shifting back
EMOTE_MAX_TILT_DEG = 8.0  # [deg] steeper than this, no move starts

_DIAGONAL_LEG = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}

# Feet frame: +X forward, +Y left, +Z up [m]; roll > 0 right, pitch > 0 nose down, yaw > 0 left [rad].
Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float]]  # (xyz, rpy)


def _ease(t01: float) -> float:
    """Cosine easing on [0, 1], zero speed at both ends."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t01, 0.0), 1.0))


def _numbers(values: Any, what: str) -> np.ndarray:
    """``values`` as a float array, if it is a list of plain finite numbers."""
    if not isinstance(values, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise ValueError(f"{what}: not a list of numbers")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what}: not finite")
    return array


def check_move(move: Any) -> None:
    """Raise ValueError unless ``move`` (a parsed JSON) is a known, bounded move: see the MAX_* limits."""
    if not isinstance(move, dict) or set(move) - _KEYS:
        raise ValueError(f"unknown keys {sorted(set(move) - _KEYS) if isinstance(move, dict) else move!r}")
    if not isinstance(move.get("description"), str) or not 0 < len(move["description"]) <= 500:
        raise ValueError("description: a text of 1 to 500 characters")
    if "mood" in move and not isinstance(move["mood"], str):
        raise ValueError("mood: a mood name")
    t = _numbers(move.get("time"), "time")  # [s]
    data = move.get("set_target_data")
    if not isinstance(data, list) or len(data) != len(t) or len(t) < 2:
        raise ValueError("set_target_data: one sample per time, at least 2")
    if t[0] != 0.0 or t[-1] > MAX_DURATION_S or np.any(np.diff(t) < MIN_STEP_S):
        raise ValueError(f"time: from 0, increasing by {MIN_STEP_S} s at least, up to {MAX_DURATION_S} s")
    key = next(iter(data[0])) if isinstance(data[0], dict) and len(data[0]) == 1 else None
    if key not in ("body", "leg") or not all(isinstance(d, dict) and list(d) == [key] for d in data):
        raise ValueError('set_target_data: every sample {"body": [6]} or every sample {"leg": [3]}')
    size = 6 if key == "body" else 3
    values = np.asarray([_numbers(d[key], key) for d in data if isinstance(d[key], list) and len(d[key]) == size])  # (N, size)
    if len(values) != len(t):
        raise ValueError(f"{key}: {size} values per sample")
    if not np.allclose(values[[0, -1]], 0.0, atol=1e-6):
        raise ValueError(f"{key}: zero at the first and last sample")
    speed = np.abs(np.diff(values, axis=0)) / np.diff(t)[:, None]  # per second
    if key == "body":
        if np.abs(values[:, :3]).max() > MAX_SHIFT_M + 1e-9 or np.abs(values[:, 3:]).max() > MAX_TILT_RAD + 1e-9:
            raise ValueError(f"body: beyond {MAX_SHIFT_M} m or {MAX_TILT_RAD} rad")
        if speed[:, :3].max() > MAX_SHIFT_SPEED_MPS or speed[:, 3:].max() > MAX_TILT_SPEED_RAD_S:
            raise ValueError(f"body: faster than {MAX_SHIFT_SPEED_MPS} m/s or {MAX_TILT_SPEED_RAD_S} rad/s")
    else:
        if move.get("default_leg") not in ("tl", "tr"):
            raise ValueError('default_leg: "tl" or "tr" (a front leg)')
        if np.abs(values).max() > MAX_LEG_RAD or speed.max() > MAX_LEG_SPEED_RAD_S:
            raise ValueError(f"leg: beyond {MAX_LEG_RAD} rad or faster than {MAX_LEG_SPEED_RAD_S} rad/s")


def hub_snapshot(repo_id: str) -> Path:
    """Folder of the Hugging Face dataset ``repo_id`` (its JSON files only): downloaded, or the cache when offline."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("moves from the Hub need huggingface_hub: pip install 'chromapi[hub]'") from exc
    kwargs: Dict[str, Any] = {"repo_id": repo_id, "repo_type": "dataset", "allow_patterns": ["*.json"]}
    try:
        return Path(snapshot_download(**kwargs))
    except Exception as exc:  # no network, Hub down: the copy downloaded last time
        logger.warning("Moves %s not downloaded (%s): using the cache", repo_id, exc)
        return Path(snapshot_download(**kwargs, local_files_only=True))


def leg_joints(leg: str) -> Tuple[str, str, str]:
    """Coxa, femur and tibia joint names of ``leg``."""
    return (f"{leg}_1", f"{leg}_2", f"{leg}_3")


class _Trajectory:
    """The samples of ``key`` in a move, linearly interpolated (held at both ends)."""

    def __init__(self, move: Dict[str, Any], key: str) -> None:
        self.description: str = move["description"]
        self._time = np.asarray(move["time"], dtype=float)  # [s]
        self._values = np.asarray([sample[key] for sample in move["set_target_data"]], dtype=float)  # (N, k)
        self.duration = float(self._time[-1])  # [s]

    def _at(self, t: float) -> List[float]:
        return [float(np.interp(t, self._time, self._values[:, i])) for i in range(self._values.shape[1])]


class RecordedMove(Move, _Trajectory):
    """The chassis follows the trajectory over the planted feet, then comes back to the start pose."""

    def __init__(self, move: Dict[str, Any], body: BalanceMove) -> None:
        """``move``: the parsed JSON; ``body``: the BalanceMove that holds the feet."""
        _Trajectory.__init__(self, move, "body")
        self._body = body
        self._elapsed = 0.0  # [s]
        self.done = False

    def evaluate(self, t: float) -> Pose:
        """Chassis pose at ``t`` [s]."""
        v = self._at(t)
        return (v[0], v[1], v[2]), (v[3], v[4], v[5])

    def on_start(self) -> None:
        """Restart, from wherever the robot stands."""
        self._elapsed, self.done = 0.0, False
        self._body.on_start()

    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """This tick's pose, then the start pose again."""
        self._elapsed += dt
        if self._elapsed < self.duration:
            self._body.set_body_pose(*self.evaluate(self._elapsed))
        else:
            self._body.set_body_pose()
            self.done = self._elapsed >= self.duration + SETTLE_S
        self._body.step(state, command, dt)


class RecordedLegMove(Move, _Trajectory):
    """Shift the weight onto the 3 other feet, play the leg trajectory with ``leg``, put it down, shift back."""

    def __init__(self, move: Dict[str, Any], body: BalanceMove, leg: str) -> None:
        """``move``: the parsed JSON; ``body``: holds the other 9 joints; ``leg``: the front leg that moves."""
        _Trajectory.__init__(self, move, "leg")
        self._body, self._leg = body, leg
        self._reset()

    def _reset(self) -> None:
        """Back to the shift phase, nothing measured yet."""
        self._phase = "shift"
        self._elapsed = 0.0  # [s] in the current phase
        self._shift: Optional[np.ndarray] = None  # [m] CoM offset, feet frame
        self._start: Optional[List[float]] = None  # [rad] leg joints when the trajectory starts
        self.done = False

    def evaluate(self, t: float) -> Tuple[float, float, float]:
        """Coxa, femur, tibia offsets from the start pose at ``t`` [s] of the trajectory."""
        v = self._at(t)
        return (v[0], v[1], v[2])

    def on_start(self) -> None:
        """Restart from the shift phase."""
        self._reset()
        self._body.on_start()

    def _compute_shift(self) -> np.ndarray:
        """CoM offset that puts the centre of mass EMOTE_COM_MARGIN_M inside the support triangle."""
        feet, com = self._body.feet, self._body.com0
        opposite = _DIAGONAL_LEG[self._leg]
        a, b = (feet[leg][:2] for leg in ("tl", "tr", "bl", "br") if leg not in (self._leg, opposite))
        edge = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-edge[1], edge[0]])
        if (feet[opposite][:2] - a) @ normal < 0.0:
            normal = -normal  # inward
        shift_xy = max(0.0, EMOTE_COM_MARGIN_M - float((com[:2] - a) @ normal)) * normal
        norm = float(np.linalg.norm(shift_xy))
        if norm > EMOTE_MAX_SHIFT_M:
            shift_xy *= EMOTE_MAX_SHIFT_M / norm
        return np.array([shift_xy[0], shift_xy[1], 0.0])

    def step(self, state: RobotState, command: MotorCommand, dt: float) -> None:
        """Shift, leg trajectory, land, shift back."""
        if self._shift is None:
            self._body.step(state, command, dt)  # anchors the feet
            self._shift = self._compute_shift()
            return
        self._elapsed += dt
        if self._phase == "shift":
            self._body.set_body_pose(_ease(self._elapsed / EMOTE_SHIFT_S) * self._shift)
            if self._elapsed >= EMOTE_SHIFT_S:
                self._phase, self._elapsed = "leg", 0.0
        elif self._phase == "leg":
            joints = leg_joints(self._leg)
            if self._start is None:
                self._start = [command.target_angles[j] for j in joints]
            offsets = (0.0, 0.0, 0.0) if self._elapsed >= self.duration else self.evaluate(self._elapsed)
            self._body.lift(self._leg, {j: q0 + dq for j, q0, dq in zip(joints, self._start, offsets)})
            if self._elapsed >= self.duration:
                self._phase, self._elapsed = "land", 0.0
        elif self._phase == "land":
            self._body.lift(None)
            if self._elapsed >= EMOTE_LAND_S:
                self._phase, self._elapsed = "unshift", 0.0
        elif self._elapsed < EMOTE_SHIFT_S:
            self._body.set_body_pose((1.0 - _ease(self._elapsed / EMOTE_SHIFT_S)) * self._shift)
        else:
            self._body.set_body_pose()
            self.done = True
        self._body.step(state, command, dt)


class RecordedMoves:
    """A library of recorded moves: every valid ``<name>.json`` of a folder or of a Hugging Face dataset."""

    def __init__(self, source: Union[Path, str] = MOVES_DIR) -> None:
        """Read every move of ``source`` once: a folder, or a dataset id ``"<account>/<name>"`` (see :func:`hub_snapshot`)."""
        directory = Path(source)
        if isinstance(source, str) and not directory.is_dir():
            directory = hub_snapshot(source)
        self._moves: Dict[str, Dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError(f"over {MAX_FILE_BYTES} bytes")
                move = json.loads(path.read_text())
                check_move(move)
            except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
                logger.warning("Move %s left out: %s", path.name, exc)
                continue
            self._moves[path.stem] = move

    def update(self, other: "RecordedMoves") -> None:
        """Add ``other``'s moves; same name: ``other``'s wins."""
        self._moves = {**self._moves, **other._moves}

    def is_leg_move(self, name: str) -> bool:
        """Whether ``name`` lifts a leg (greeting, kick) rather than posing the chassis."""
        return "leg" in self._moves[name]["set_target_data"][0]

    def mood(self, name: str) -> Optional[str]:
        """The mood (ring and sound) the file asks for, if any."""
        return self._moves[name].get("mood")

    def get(self, name: str, body: BalanceMove, leg: Optional[str] = None) -> Union[RecordedMove, RecordedLegMove]:
        """A fresh player of move ``name``, balanced by ``body``; ``leg`` for a leg move (default: the file's)."""
        move = self._moves[name]
        if self.is_leg_move(name):
            return RecordedLegMove(move, body, leg or move["default_leg"])
        return RecordedMove(move, body)

    def list_moves(self) -> List[str]:
        """Names of the moves, sorted."""
        return list(self._moves)
