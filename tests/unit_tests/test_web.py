# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the web app (``chromapi run``) and the recorded moves it plays."""

import math
import time
from pathlib import Path

import pytest

from chromapi.chromapi import MotorCommand, RobotState
from chromapi.locomotion.recorded_move import (
    MAX_DURATION_S,
    SETTLE_S,
    RecordedLegMove,
    RecordedMove,
    RecordedMoves,
)
from chromapi.web import app as web


class StubTeleop:
    """Records the actions the page asks for."""

    def __init__(self):
        """No action yet."""
        self.actions, self.speed_scale, self.max_speed_scale = [], 1.0, web.MAX_SPEED_SCALE

    def request(self, action):
        """Store the action."""
        self.actions.append(action)

    def listen_stop(self):
        """The pilot left: nothing to close here."""

    def extra_moves(self):
        """A dataset move, listed after the moods."""
        return ["wiggle"]


def test_deadman_centres_the_sticks(monkeypatch):
    """When the page goes silent, the sticks read zero: the robot stops but is not released."""
    monkeypatch.setattr(web, "INPUT_TIMEOUT_S", 0.05)
    web_input = web.WebInput()
    web_input.set_axes({"ly": 1.0, "lx": 0.05})  # lx inside the deadzone
    assert web_input.read() == {"lx": 0.0, "ly": 1.0, "rx": 0.0, "ry": 0.0}
    time.sleep(0.1)
    assert web_input.read() == {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}


def test_server_serves_the_page_offline_and_relays_intentions(monkeypatch, tmp_path):
    """The page and its socket.io client come from the robot; page events reach the teleop."""
    teleop, web_input = StubTeleop(), web.WebInput()
    written = []
    monkeypatch.setattr(web, "write_report", lambda path, use_bridge, extra: _fake_report(path, use_bridge, extra, written))
    access = web.Access(tmp_path / "access.json")
    app, socketio = web.build_server(teleop, web_input, None, tmp_path, lambda: {"battery": {"percent": 42}}, access)
    http = app.test_client()
    page = http.get("/").get_data(as_text=True)
    assert '<script src="/static/socket.io.min.js">' in page and "cdnjs" not in page
    for asset in ("/static/socket.io.min.js", "/static/logo.png", "/static/motherboard_front.png", "/static/fonts/varela-round.woff2", "/favicon.ico"):
        assert http.get(asset).status_code == 200, asset
    # Signed out: the page and its sign-in screen only.
    for private in ("/video_feed", "/gallery", "/report", "/recordings/x.jpg", "/moods"):
        assert http.get(private).status_code == 401, private
    assert not socketio.test_client(app, flask_test_client=http).is_connected()
    assert http.get("/auth/state").get_json() == {"configured": False, "signed_in": False, "min_length": 6}
    assert http.post("/auth/setup", json={"code": "12345"}).status_code == 400  # too short
    assert http.post("/auth/setup", json={"code": "123456"}).status_code == 200  # the first visitor chooses
    assert http.get("/auth/state").get_json()["signed_in"]
    other = app.test_client()
    assert other.post("/auth/setup", json={"code": "999999"}).status_code == 409  # only once
    assert other.post("/auth/login", json={"code": "999999"}).status_code == 403
    assert other.post("/auth/login", json={"code": "123456"}).status_code == 200
    assert other.post("/auth/change", json={"code": "abcdef"}).status_code == 403  # the current code first
    assert other.post("/auth/change", json={"code": "abcdef", "current": "999999"}).status_code == 403
    assert other.post("/auth/change", json={"code": "abcdef", "current": "123456"}).status_code == 200
    assert http.get("/gallery").status_code == 401  # a new code signs the other browsers out
    assert http.post("/auth/login", json={"code": "abcdef"}).status_code == 200
    assert http.get("/photo").status_code == 404  # no camera
    (tmp_path / "20261027-101010_photo.jpg").write_bytes(b"jpg")
    (tmp_path / "20261027-101011_camera.avi").write_bytes(b"avi")
    assert [f["kind"] for f in http.get("/gallery").get_json()["files"]] == ["photo", "video"]
    assert http.get("/recordings/20261027-101010_photo.jpg").data == b"jpg"
    (tmp_path / "access.keep").write_text("x")
    for other_file in ("access.keep", "..", "missing_photo.jpg"):
        assert http.delete(f"/recordings/{other_file}").status_code == 404, other_file
    assert (tmp_path / "access.keep").exists()
    assert http.delete("/recordings/20261027-101011_camera.avi").status_code == 200
    assert [f["kind"] for f in http.get("/gallery").get_json()["files"]] == ["photo"]
    moods = http.get("/moods").get_json()["moods"]
    assert len(moods) == 20 and moods[-1] == {"name": "wiggle", "emoji": "✨"}
    report = http.get("/report")
    assert report.status_code == 200 and b"no bridge" in report.data and b'"percent": 42' in report.data
    assert not written[0].exists()  # no copy left on the robot

    client = socketio.test_client(app, flask_test_client=http)
    assert client.is_connected()
    client.emit("action", {"name": "greet"})
    client.emit("axes", {"ly": 0.5})
    client.emit("set_speed", {"value": 5.0})
    assert teleop.actions == ["greet"]
    assert web_input.read()["ly"] == 0.5
    assert teleop.speed_scale == web.MAX_SPEED_SCALE  # clamped


def _fake_report(path, use_bridge, extra, written):
    written.append(path)
    path.write_text(("bridge" if use_bridge else "no bridge") + "\n" + extra)
    return Path(path)


@pytest.mark.parametrize("name", ["greeting", "kick"])
def test_leg_moves_end_where_they_started(name):
    """A leg move lifts its leg, moves it, and puts it back exactly: zero offsets at both ends."""
    move = RecordedMoves().get(name, None)
    assert isinstance(move, RecordedLegMove) and move.description
    assert move.evaluate(0.0) == (0.0, 0.0, 0.0) and move.evaluate(move.duration) == (0.0, 0.0, 0.0)
    assert max(abs(v) for t in range(int(move.duration * 100)) for v in move.evaluate(t / 100.0)) > 0.1


def test_a_servo_is_missing_only_after_consecutive_misses():
    """One lost bus cycle is normal: a servo is reported missing only after SERVO_MISSING_CYCLES in a row."""
    from chromapi.chromapi import SERVO_MISSING_CYCLES, HardwareBackend, load_config

    backend = HardwareBackend(load_config(calibration_path="/nonexistent")["servos"])
    fresh = {"all": True}

    def snapshot():
        servos = [{"id": i, "pos": 2048, "speed": 0, "load": 0, "current_A": 0.0, "temp_C": 30,
                   "fresh": fresh["all"] or i != 1} for i in range(1, 13)]
        return {"t_us": 0, "servos": servos, "power": {"voltage_V": 7.8, "current_A": 1.0, "power_W": 7.8},
                "imu": {"quat": (1.0, 0.0, 0.0, 0.0), "gyro_rps": (0.0,) * 3, "gyro_corr_rps": (0.0,) * 3,
                        "acc_mps2": (0.0, 0.0, 9.8)},
                "switches": {"TL": True, "TR": True, "BL": True, "BR": True}}

    backend._bridge.get_state = snapshot  # type: ignore[method-assign]
    fresh["all"] = False  # servo 1 stops answering
    missing = [backend.read_state().missing_servos for _ in range(SERVO_MISSING_CYCLES)]
    assert missing[0] == [] and missing[-2] == [] and len(missing[-1]) == 1
    fresh["all"] = True
    assert backend.read_state().missing_servos == []


def _until(condition, timeout_s=1.0):
    """True as soon as ``condition()`` is, False after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_led_ring_flash_then_back_to_persistent_and_torch_over_all(monkeypatch):
    """A mood plays its sound and animation, then the ring goes back; the torch wins over everything."""
    from chromapi.media import led

    class Robot:
        def __init__(self):
            self.frames, self.sounds = [], []

        def set_led_all(self, r, g, b):
            self.frames.append([(r, g, b)] * 18)

        def set_led_bulk(self, colors):
            self.frames.append(list(colors))

        def play_sound(self, path, gain=1.0):
            self.sounds.append(path.name)

    monkeypatch.setattr(led, "MIN_FLASH_S", 0.1)
    monkeypatch.setattr(led, "sound_duration_s", lambda path: 0.0)
    robot = Robot()
    ring = led.LedRing(robot)
    ring.set_persistent(led.render_off)
    ring.start()
    ring.play_mood("alert")
    # The ring draws in its own thread: wait for each frame, not a fixed time (flaky on a loaded machine).
    assert _until(lambda: robot.sounds == ["alert.wav"] and any(f[0] != (0, 0, 0) for f in robot.frames))
    assert _until(lambda: robot.frames[-1][0] == (0, 0, 0))  # back to off
    ring.torch = True
    assert _until(lambda: robot.frames[-1][0] == (led.TORCH_LEVEL,) * 3)
    ring.stop()
    assert robot.frames[-1][0] == (0, 0, 0)



def _maintenance_teleop(monkeypatch):
    """A Teleop on a stub hardware robot, lying down; returns it and the robot calls it records."""

    class Backend(web.HardwareBackend):
        def __init__(self):
            self.applied = None

        def apply_calibration(self, config):
            self.applied = config

    class Robot:
        loop_hz, backend, calls = 100.0, Backend(), []

        def disconnect(self):
            self.calls.append("disconnect")

        def connect(self):
            self.calls.append("connect")
            return True

        def disable_motors(self):
            self.calls.append("limp")

    class Ring:
        def stop(self):
            Robot.calls.append("ring off")

        def start(self):
            Robot.calls.append("ring on")

    monkeypatch.setattr(web, "load_config", lambda: {"servos": {}})
    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    teleop.ring = Ring()
    return teleop, Robot


def _wait_maintenance(teleop):
    deadline = time.monotonic() + 5.0
    while teleop.maintenance.get("running") and time.monotonic() < deadline:
        time.sleep(0.01)


def test_maintenance_hands_the_port_over_then_applies_the_calibration(monkeypatch):
    """Refused standing up; lying down, the tool runs on a released port, then the new calibration is applied."""
    import subprocess

    ran = []

    class Tool:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            ran.append(cmd)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def communicate(self, timeout=None):
            return "[SUCCESS] Written\n", None

    monkeypatch.setattr(subprocess, "Popen", Tool)
    teleop, robot = _maintenance_teleop(monkeypatch)
    teleop.awake = True
    teleop._do("calibrate_imu")
    assert not teleop.maintenance and "lay Chromapi down" in teleop.status
    teleop.awake = False
    teleop._do("calibrate_imu")
    _wait_maintenance(teleop)
    assert ran[0][-3:] == ["chromapi", "calibrate", "imu"]
    assert robot.calls == ["ring off", "disconnect", "connect", "limp", "ring on"]
    assert robot.backend.applied == {"servos": {}} and teleop.maintenance["ok"]
    assert teleop.status == "Done: [SUCCESS] Written"


def test_maintenance_cancel_interrupts_the_tool_but_not_a_flash(monkeypatch, tmp_path):
    """Cancel is a Ctrl-C to the tool, acted on at once; the robot is taken back as after any job."""
    tool = tmp_path / "tool.sh"
    import sys

    tool.write_text(f"#!{sys.executable}\nimport sys, time\nprint('started', flush=True)\ntry:\n    time.sleep(10)\n"
                    "except KeyboardInterrupt:\n    print('stopped')\n    sys.exit(130)\n")
    tool.chmod(0o755)
    monkeypatch.setattr(web.sys, "executable", str(tool))  # runs "<tool> -m chromapi calibrate ..."
    teleop, robot = _maintenance_teleop(monkeypatch)
    teleop._do("calibrate_servos")
    deadline = time.monotonic() + 5.0
    while teleop._tool is None and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.5)  # the interpreter is up, inside its sleep
    teleop.request("maintenance_cancel")
    _wait_maintenance(teleop)
    assert teleop.maintenance["cancelled"] and not teleop.maintenance["ok"]
    assert teleop.status == "Cancelled: stopped" and robot.backend.applied is None
    assert robot.calls[-3:] == ["connect", "limp", "ring on"]

    teleop.maintenance, teleop._tool = {"job": "update_firmware", "running": True}, object()
    teleop.request("maintenance_cancel")  # never mid-flash
    assert "cancelled" not in teleop.maintenance


def test_speed_over_100_percent_only_while_the_battery_is_not_low():
    """Under Battery.low_percent the full-speed sag would starve the servos: the speed falls back to 100 %."""
    from chromapi.hardware.battery import LOW, OK

    class Robot:
        loop_hz = 100.0

        class battery:
            level = None  # not read yet: no cap

    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    teleop.speed_scale = web.MAX_SPEED_SCALE
    teleop._cap_speed()
    assert teleop.speed_scale == web.MAX_SPEED_SCALE
    Robot.battery.level = OK
    teleop._cap_speed()
    assert teleop.speed_scale == web.MAX_SPEED_SCALE
    Robot.battery.level = LOW
    teleop._cap_speed()
    assert teleop.speed_scale == teleop.max_speed_scale == 1.0 and "capped at 100%" in teleop.status


def test_battery_on_the_ring_matches_the_status_led():
    """As many LEDs as the charge, in the status LED's colour; under 10 % they blink."""
    from chromapi.media import led

    assert led.battery_color(80.0)[0] == 0 and led.battery_color(80.0)[1] > 0  # green
    r, g, _ = led.battery_color(30.0)
    assert r == g > 0  # yellow at the "low" threshold
    assert led.battery_color(15.0)[0] > led.battery_color(15.0)[1] > 0  # orange-red

    class Ring:
        def set_led_bulk(self, colors):
            self.lit = sum(c != (0, 0, 0) for c in colors)

    ring = Ring()
    led.render_battery(50.0)(ring, 0.0)
    assert ring.lit == 9  # half of 18
    led.render_battery(5.0)(ring, 0.0)
    on = ring.lit
    led.render_battery(5.0)(ring, 0.5)
    assert on == 1 and ring.lit == 0  # blinking, never fully dark at 0 % when on


def test_battery_moods_play_once_per_threshold_and_wait_for_the_emote():
    """tired entering "low", alert entering "critical", once each, never during an emote; show_battery flashes the ring."""
    from chromapi.hardware.battery import CRITICAL, LOW, OK

    class Robot:
        loop_hz = 100.0

        class battery:
            level, percent = OK, 80.0

    class Ring:
        def __init__(self):
            self.played, self.flashes = [], []

        def play_mood(self, name):
            self.played.append(name)

        def flash(self, render, duration_s):
            self.flashes.append(duration_s)

    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    teleop.ring = Ring()
    Robot.battery.level = LOW
    teleop._check_battery()
    assert teleop.ring.played == []  # waits for the startup sound
    assert teleop.startup_mood() == "tired"  # starting low: tired instead of the scream...
    teleop._check_battery()
    assert teleop.ring.played == []  # ...and the threshold does not play it again
    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    teleop.ring = Ring()
    Robot.battery.level = OK
    assert teleop.startup_mood() == "scream"
    for level in (OK, LOW, LOW, OK, LOW):  # back up and down again around the threshold: no replay
        Robot.battery.level = level
        teleop._check_battery()
    assert teleop.ring.played == ["tired"]
    teleop._emote = object()
    Robot.battery.level = CRITICAL
    teleop._check_battery()
    assert teleop.ring.played == ["tired"]  # waits for the emote
    teleop._emote = None
    teleop._check_battery()
    assert teleop.ring.played == ["tired", "alert"]

    teleop._do("show_battery")
    assert teleop.ring.flashes == [web.led.BATTERY_FLASH_S]


def test_access_code_is_hashed_and_wrong_codes_slow_down(tmp_path, monkeypatch):
    """The code never reaches the disk in clear, the file is owner-only, and guessing gets slower and slower."""
    from chromapi.web import access as A

    monkeypatch.setattr(A, "_SCRYPT", {"n": 2**8, "r": 8, "p": 1})  # fast hashing for the test
    path = tmp_path / "access.json"
    gate = A.Access(path)
    assert gate.choose("secret-code") and not gate.choose("another-one")
    assert "secret-code" not in path.read_text() and path.stat().st_mode & 0o077 == 0
    assert gate.check("secret-code") == (True, 0.0)
    waits = [gate.check("guess")[1] for _ in range(A.FREE_ATTEMPTS)]
    assert waits[:-1] == [0.0] * (A.FREE_ATTEMPTS - 1) and waits[-1] == A.FIRST_DELAY_S
    assert gate.check("secret-code")[0] is False  # even the right code waits out the delay
    generation = gate.generation
    gate.reset()
    assert not gate.configured and gate.generation != generation  # forgotten, every browser signed out


def test_camera_is_on_only_while_watched(monkeypatch):
    """Nothing is captured until someone watches; the camera closes once the last viewer leaves."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from chromapi.web import camera as cam

    events = []

    class FakeCapture:
        def __init__(self, device):
            events.append("open")

        def set(self, *args):
            return True

        def isOpened(self):
            return True

        def read(self):
            time.sleep(0.01)
            return True, np.zeros((48, 64, 3), np.uint8)

        def release(self):
            events.append("close")

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cam, "IDLE_CLOSE_S", 0.1)
    streamer = cam.CameraStreamer({})
    streamer.start()
    try:
        time.sleep(0.3)
        assert events == [] and streamer.viewers == 0
        stream = streamer.mjpeg()
        assert next(stream).startswith(b"--frame")  # opened for the viewer
        assert events == ["open"] and streamer.viewers == 1
        stream.close()  # the page went away
        time.sleep(0.5)
        assert events == ["open", "close"] and streamer.viewers == 0
    finally:
        streamer.stop()


def test_talkie_walkie_plays_the_chunks_then_closes(monkeypatch):
    """int16 chunks reach the speaker as float samples; odd or oversized chunks are dropped; silence closes it."""
    import numpy as np

    class Speaker:
        def __init__(self):
            self.chunks, self.stopped = [], False

        def write(self, samples, rate):
            self.chunks.append((samples, rate))

        def stop(self):
            self.stopped = True

    class Robot:
        loop_hz = 100.0
        sound_volume = 0.25

    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    speaker = Speaker()
    monkeypatch.setattr(teleop, "_open_speaker", lambda: speaker)
    teleop.talk(np.array([16384, -32768], dtype="<i2").tobytes())
    teleop.talk(b"\x00\x01\x02")  # odd length
    teleop.talk(bytes(web.TALK_MAX_CHUNK + 2))  # too long
    assert len(speaker.chunks) == 1
    samples, rate = speaker.chunks[0]
    assert rate == web.TALK_RATE and samples.tolist() == pytest.approx([0.125, -0.25])
    teleop._close_idle_talk()
    assert not speaker.stopped  # still talking
    monkeypatch.setattr(web, "TALK_IDLE_S", 0.0)
    teleop._close_idle_talk()
    assert speaker.stopped and teleop._talk_out is None


def test_the_first_page_drives_the_others_watch(tmp_path):
    """Two pages would fight over the robot: only the first one's commands count, until it leaves."""
    teleop, web_input = StubTeleop(), web.WebInput()
    access = web.Access(tmp_path / "access.json")
    app, socketio = web.build_server(teleop, web_input, None, tmp_path, lambda: {}, access)
    first, second = app.test_client(), app.test_client()
    assert first.post("/auth/setup", json={"code": "123456"}).status_code == 200
    assert second.post("/auth/login", json={"code": "123456"}).status_code == 200
    pilot = socketio.test_client(app, flask_test_client=first)
    watcher = socketio.test_client(app, flask_test_client=second)

    def role(client):
        return [m["args"][0] for m in client.get_received() if m["name"] == "role"][-1]

    assert role(pilot) == {"pilot": True, "watchers": 1} and role(watcher) == {"pilot": False, "watchers": 1}
    watcher.emit("action", {"name": "greet"})
    watcher.emit("axes", {"ly": -1.0})
    pilot.emit("axes", {"ly": 0.5})
    pilot.emit("action", {"name": "kick"})
    assert teleop.actions == ["kick"] and web_input.read()["ly"] == 0.5
    pilot.disconnect()  # the pilot leaves: its sticks are centred, the watcher takes over
    assert web_input.read()["ly"] == 0.0 and role(watcher) == {"pilot": True, "watchers": 0}
    watcher.emit("action", {"name": "greet"})
    assert teleop.actions == ["kick", "greet"]


class _Body:
    """Stands in for BalanceMove: keeps the last chassis pose."""

    def on_start(self):
        self.pose = None

    def set_body_pose(self, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0)):
        self.pose = (xyz, rpy)

    def step(self, state, command, dt):
        pass


@pytest.mark.parametrize("name", [n for n in RecordedMoves().list_moves()
                                  if isinstance(RecordedMoves().get(n, None), RecordedMove)])
def test_recorded_moves_start_and_end_at_rest(name):
    """Each move is a pose from rest back to rest, within what the legs reach (±3 cm, ±0.2 rad)."""
    import numpy as np

    move = RecordedMoves().get(name, _Body())
    pose = lambda s: move.evaluate(s * move.duration)  # noqa: E731
    assert move.description
    for s in (0.0, 1.0):
        xyz, rpy = pose(s)
        assert np.allclose(xyz, 0.0, atol=1e-9) and np.allclose(rpy, 0.0, atol=1e-9), (name, s)
    poses = [pose(s) for s in np.linspace(0.0, 1.0, 400)]
    assert max(abs(v) for xyz, _ in poses for v in xyz) <= 0.03
    assert max(abs(v) for _, rpy in poses for v in rpy) <= 0.2
    assert max(abs(v) for _, rpy in poses for v in rpy) + max(abs(v) for xyz, _ in poses for v in xyz) > 0.01  # it moves

    move.on_start()
    ticks = 0
    while not move.done and ticks < (MAX_DURATION_S + SETTLE_S) / 0.01 + 10:
        move.step(RobotState(), MotorCommand(target_angles={}), 0.01)
        ticks += 1
    assert ticks == pytest.approx((move.duration + SETTLE_S) / 0.01, abs=2)


def test_dizzy_comet_turns_with_the_chassis():
    """The ring's comet turns as fast as the chassis circles, for as long as the gesture lasts."""
    import math

    move = RecordedMoves().get("dizzy", _Body())
    duration = move.duration
    pose = lambda s: move.evaluate(s * duration)  # noqa: E731
    angles = [math.atan2(pose(s)[0][1], pose(s)[0][0]) for s in (0.25, 0.25 + 0.01)]  # CoM angle on its circle
    step = math.remainder(angles[1] - angles[0], 2.0 * math.pi)  # [rad], across the ±pi wrap
    chassis_turns_per_s = step / (0.01 * duration) / (2.0 * math.pi)
    assert chassis_turns_per_s == pytest.approx(-0.4, rel=0.01)  # clockwise, at the comet's rate in led.render_dizzy

    class Ring:
        def play_mood(self, name, duration_s=0.0):
            self.duration_s = duration_s

    class Robot:
        loop_hz = 100.0

        def play_move(self, move):
            pass

    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    teleop.ring = Ring()
    teleop._emote_refusal = lambda: None  # standing still, level
    teleop._play_mood("dizzy")
    assert teleop.ring.duration_s == duration


def test_listening_shows_on_the_ring_and_is_one_way_at_a_time(monkeypatch):
    """Chromapi's microphone goes to the page with the ring lit; talking closes it; asked meanwhile, it reopens once quiet."""
    import numpy as np

    from chromapi.media import audio

    stereo = np.array([[0.01, 0.0], [-0.02, 0.0], [0.5, 0.0]], dtype=np.float32)  # mono mic on the left channel
    pcm = np.frombuffer(audio.to_int16_mono(stereo, gain=30.0), dtype="<i2")
    assert pcm.tolist() == [9830, -19660, 32767]  # x30, the last one clipped

    opened = []

    class Mic:
        def __init__(self, send, samplerate, gain, denoise=True):
            self.send, self.gain, self.stopped = send, gain, False
            opened.append(self)

        def start(self):
            return True

        def stop(self):
            self.stopped = True

    class Speaker:
        def write(self, samples, rate):
            pass

        def stop(self):
            pass

    class Robot:
        loop_hz, config = 100.0, {"audio": {"mic_gain": 30.0}}
        sound_volume = 0.25

    monkeypatch.setattr(audio, "MicStream", Mic)
    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    monkeypatch.setattr(teleop, "_open_speaker", lambda: Speaker())
    assert teleop.listen_start(lambda chunk: None) and teleop.ring.listening and opened[0].gain == 30.0
    teleop.talk(bytes(64))  # the pilot talks: listening stops
    assert opened[0].stopped and not teleop.ring.listening
    assert teleop.listen_start(lambda chunk: None) and len(opened) == 1  # asked while talking: held back
    teleop._talk_last -= web.TALK_IDLE_S + 0.1
    teleop._close_idle_talk()  # the speaker went quiet: listening comes back
    assert len(opened) == 2 and teleop.ring.listening
    teleop.talk(bytes(64))
    teleop.listen_start(lambda chunk: None)
    teleop.listen_stop()  # released before the speaker went quiet: nothing reopens
    teleop._talk_last -= web.TALK_IDLE_S + 0.1
    teleop._close_idle_talk()
    assert len(opened) == 2 and not teleop.ring.listening


def test_denoiser_removes_the_fan_and_keeps_the_voice():
    """Steady noise (hum + hiss) drops by >10 dB once learnt; a voice-band tone on top comes through."""
    import numpy as np

    from chromapi.media import audio

    rng = np.random.default_rng(0)
    rate, chunk = 16000, 640
    t = np.arange(6 * rate) / rate
    noise = (0.05 * np.sin(2 * np.pi * 90 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    voice = (0.1 * np.sin(2 * np.pi * 1000 * t) * (t >= 5)).astype(np.float32)  # speaks in the last second
    x = noise + voice
    ns = audio.NoiseSuppressor(rate)
    y = np.concatenate([ns.apply(x[i:i + chunk]) for i in range(0, len(x), chunk)])
    assert len(y) == len(x)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
    quiet = slice(4 * rate, 5 * rate - 1000)
    assert rms(y[quiet]) < 0.3 * rms(x[quiet])  # noise only: >10 dB down
    talk = slice(5 * rate + 2000, 6 * rate)
    tone = lambda a: abs(np.fft.rfft(a)[1000 * len(a) // rate]) / len(a)  # noqa: E731
    assert tone(y[talk]) > 0.7 * tone(voice[talk])  # the voice kept

    loud = (0.3 * np.sin(2 * np.pi * 500 * t[:4 * rate]) * (t[:4 * rate] >= 2)).astype(np.float32)  # held 2 s
    x = noise[:4 * rate] + loud
    ns = audio.NoiseSuppressor(rate)
    y = np.concatenate([ns.apply(x[i:i + chunk]) for i in range(0, len(x), chunk)])
    late = slice(int(3.5 * rate), 4 * rate)
    assert rms(y[late]) > 0.8 * rms(loud[late])  # a loud, close voice is not learnt as noise


def test_auto_gain_lifts_a_quiet_voice_and_never_clips_a_loud_one():
    """Quiet speech rises towards -20 dBFS (capped), a sudden loud sound brings the gain down at once."""
    import numpy as np

    from chromapi.media import audio

    agc = audio.AutoGain()
    t = np.arange(640) / 16000.0
    quiet = (0.02 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # rms ~0.014
    for _ in range(100):  # 4 s of 40 ms chunks
        out = agc.apply(quiet)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(audio.AGC_TARGET_RMS, rel=0.02)  # x14, under the cap
    hiss = (0.001 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    for _ in range(100):
        agc.apply(hiss)
    assert agc.gain == pytest.approx(audio.AGC_MAX_GAIN, rel=0.01)  # silence's hiss: capped
    loud = (0.8 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = agc.apply(loud)
    assert agc.gain == 1.0 and np.abs(out).max() <= 0.8 + 1e-6  # down at once: no clipping


def test_gamepad_drives_the_teleop_and_hands_back_to_the_page(tmp_path):
    """Buttons become the page's actions, the pad's sticks win while it moves, and a dropped pad reads zero."""
    import math

    from chromapi.chromapi import IDLE_LED_RGB
    from chromapi.input.gamepad import UserInput

    device = tmp_path / "js0"

    class Pad:
        """Fake gamepad: the test sets what it reads; it records its light bar and its buzzes."""

        def __init__(self, path):
            self.connected, self.user = False, UserInput()
            self.gravity, self.colors, self.buzzes, self.charge = None, [], [], None

        def start(self):
            self.connected = True

        def stop(self):
            self.connected = False
            self.colors.append("default")

        def read(self):
            user, self.user = self.user, UserInput(forward=self.user.forward, held=self.user.held)
            return user

        def tilt(self):
            return self.gravity

        def set_color(self, rgb):
            self.colors.append(rgb)

        def rumble(self, pulses):
            self.buzzes.append(pulses)

        def battery(self):
            return self.charge

    class Robot:
        class battery:
            percent = 80.0

    class Ring:
        torch = False

    pads = []
    control = web.GamepadControl(str(device), lambda path: pads.append(Pad(path)) or pads[-1])
    teleop = StubTeleop()
    teleop.awake, teleop.gait, teleop.mode, teleop.height, teleop.status = False, "trot", "turn", 0.0, ""
    teleop.robot, teleop.ring = Robot(), Ring()
    control.poll(teleop, 0.0)
    assert not control.connected and not pads  # no device yet

    device.touch()
    control.poll(teleop, 10.0)
    pad = pads[-1]
    assert pad.colors == [IDLE_LED_RGB]  # lying down: the idle colour, not off
    pad.user = UserInput(pressed={web.PAD_CROSS, web.PAD_L3, web.PAD_R2, web.PAD_L1}, held={"dpad_up"})
    control.poll(teleop, 10.1)
    assert teleop.actions == ["wake", "crawl"] and teleop.ring.torch
    assert pad.colors[-1] == web.led.PAD_MOOD_COLORS[web.led.MOODS[1].name]  # the chosen mood on the light bar
    assert teleop.height == 2 * web.TICK_S * web.PAD_HEIGHT_RATE_MPS  # a long gap between polls: one capped step
    teleop.height = 0.0
    for t in (10.11, 10.12, 10.13):  # D-pad up held: the height rises steadily, up to its limit
        control.poll(teleop, t)
    assert teleop.height == pytest.approx(3 * 0.01 * web.PAD_HEIGHT_RATE_MPS, abs=1e-4)
    teleop.height = web.HEIGHT_RANGE_M[1]
    control.poll(teleop, 10.14)
    assert teleop.height == web.HEIGHT_RANGE_M[1]
    assert "Circle plays it" in teleop.status
    pad.user = UserInput(pressed={web.PAD_CIRCLE, web.PAD_SHARE}, held={"dpad_right"})
    control.poll(teleop, 10.2)
    assert teleop.actions[-2:] == [f"mood:{web.led.MOODS[1].name}", "show_battery"] and teleop.speed_scale == 1.25
    assert pad.colors[-1] == web.led.battery_color(80.0)
    control.poll(teleop, 10.2 + web.PAD_FLASH_S)
    assert pad.colors[-1] == IDLE_LED_RGB  # the flash is over

    pad.user = UserInput(forward=0.8)
    control.poll(teleop, 11.0)
    assert control.active(11.5) and control.sticks["ly"] == 0.8
    pad.user = UserInput()
    control.poll(teleop, 12.0)
    assert not control.active(13.1)  # at rest: the page's sticks again

    # Lean with the controller's tilt: zero where it was when tilting started, a full lean at PAD_TILT_FULL_RAD.
    control.tilt_lean, teleop.mode = True, "lean"
    pad.gravity = (0.0, 1.0, 0.0)
    control.poll(teleop, 14.0)
    assert control.sticks["rx"] == control.sticks["ry"] == 0.0
    a = web.PAD_TILT_FULL_RAD / 2
    pad.gravity = (0.0, math.cos(a), math.sin(a))  # tilted forward by half the full range
    control.poll(teleop, 14.1)
    assert control.sticks["ry"] == pytest.approx(0.5 * web.PAD_TILT_SIGNS[0]) and control.sticks["rx"] == 0.0
    pad.gravity = (math.sin(2 * a), math.cos(2 * a), 0.0)
    control.poll(teleop, 14.2)
    assert control.sticks["rx"] == pytest.approx(web.PAD_TILT_SIGNS[1]) and control.sticks["ry"] == 0.0
    pad.user = UserInput(turn=0.5)  # the right stick touched: it leans alone, the tilt waits
    control.poll(teleop, 14.25)
    assert control.sticks["rx"] == -0.5 and control.sticks["ry"] == 0.0
    pad.gravity = (math.sin(a), math.cos(a) ** 2, math.cos(a) * math.sin(a))  # tilted both ways
    pad.user = UserInput()
    control.poll(teleop, 14.27)
    assert control.sticks["rx"] != 0.0 and control.sticks["ry"] != 0.0  # stick released: the tilt, both axes
    pad.user = UserInput()
    control.tilt_lean = False
    control.poll(teleop, 14.3)
    assert control.sticks["rx"] == 0.0  # the right stick again

    pad.charge = (15, False)
    control.poll(teleop, 15.0)
    control.poll(teleop, 30.0)
    assert pad.buzzes == [2] and "Controller battery low" in teleop.status  # once, not every check

    pad.user = UserInput(held={web.PAD_START})
    for t in (40.0, 41.0, 42.0):
        control.poll(teleop, t)
    assert "power off" in teleop.status and "shutdown" not in teleop.actions
    control.poll(teleop, 43.0)
    control.poll(teleop, 44.0)
    assert teleop.actions.count("shutdown") == 1  # once per hold

    pad.user = UserInput(forward=1.0)
    control.poll(teleop, 50.0)
    pad.connected = False  # Bluetooth dropped: js0 goes away
    device.unlink()
    control.poll(teleop, 50.1)
    assert not control.connected and control.sticks["ly"] == 0.0
    control.close()
    assert pad.colors[-1] == "default"  # the driver's colour back


def test_every_mood_has_a_light_bar_colour():
    """L2/R2 show the chosen mood on the controller: each mood needs a colour."""
    assert set(web.led.PAD_MOOD_COLORS) == set(web.led.MOODS_BY_NAME)


def test_developer_mode_stops_the_app_without_powering_off(monkeypatch):
    """The page's Developer mode: Chromapi lying and limp, app ending cleanly (the service leaves it stopped), no systemctl."""
    calls = []

    class Robot:
        loop_hz = 100.0

        def disconnect(self, disable_torque=False):
            calls.append(("disconnect", disable_torque))

    class Ring:
        def stop(self):
            calls.append("ring off")

    monkeypatch.setattr(web.subprocess, "run", lambda *a, **k: calls.append("systemctl"))
    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    teleop.ring = Ring()
    teleop.on_shutdown = calls.append  # the pages are told first, before lying down
    teleop._do("quit")
    assert teleop.shutting_down and "Developer mode" in teleop.status
    assert calls == ["quit", "ring off", ("disconnect", True)]  # and no systemctl


def test_resume_if_held_stands_up_only_when_the_servos_hold_torque():
    """Restarted within the firmware's hold, Chromapi is still up: the app wakes it up from there."""
    from types import SimpleNamespace

    class Robot:
        loop_hz, backend, woken = 100.0, None, 0
        wake_up_duration_s = 0.0
        state = SimpleNamespace(joint_loads={"tl_2": 0.0, "tl_3": 2.0})

        def get_state(self):
            return self.state

        def wake_up(self, wait=False):
            Robot.woken += 1

    robot = Robot()
    teleop = web.Teleop(robot, web.WebInput(), hardware=True)
    teleop.resume_if_held()
    assert not teleop.awake and Robot.woken == 0  # limp: lying down, untouched
    robot.state = SimpleNamespace(joint_loads={"tl_2": -250.0, "tl_3": 120.0})
    teleop.resume_if_held()
    assert teleop.awake and Robot.woken == 1 and "resumed" in teleop.status


def test_ring_colour_follows_the_chosen_idle_colour_and_goes_back():
    """Idle and squeak use the chosen colour (squeak's accent: hot pink for the default, cyan for blue); None resets."""
    from chromapi.media import led

    class Ring:
        def set_led_all(self, *rgb):
            self.frame = rgb

    ring = Ring()
    try:
        led.set_idle_color((0, 0, 255))
        led.render_idle(ring, 0.5)  # top of the breath
        assert ring.frame == led.dimmed((0, 0, 255))
        accent = led.accent_color((0, 0, 255))
        assert accent[2] == 255 and accent[1] > 200 and accent[0] < 128  # cyan side, not violet
        r, g, b = led.accent_color(led.IDLE)
        assert r == 255 and g < b  # hot pink
    finally:
        led.set_idle_color(None)
    assert led.idle_color() == led.IDLE


def test_lean_sticks_stay_in_a_circle():
    """A square-gated stick's diagonal leans no further than straight ahead."""
    from chromapi.web.app import _in_circle

    assert _in_circle(1.0, 1.0) == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert _in_circle(0.3, -0.4) == (0.3, -0.4)


def test_dataset_moves_join_the_moods_menu(tmp_path):
    """Config moves.dataset (here a folder): loaded in the background, listed, played with the mood it names."""
    import json

    from chromapi.locomotion.recorded_move import MOVES_DIR

    yes = json.loads((MOVES_DIR / "yes.json").read_text())
    (tmp_path / "wiggle.json").write_text(json.dumps({**yes, "mood": "joy"}))
    (tmp_path / "shrug.json").write_text(json.dumps(yes))  # no mood: DATASET_MOOD

    class Robot:
        loop_hz, config = 100.0, {"moves": {"dataset": str(tmp_path)}}

    class Ring:
        played = []

        def play_mood(self, name, duration_s=0.0):
            self.played.append(name)

    teleop = web.Teleop(Robot(), web.WebInput(), hardware=True)
    teleop.ring = Ring()
    deadline = time.monotonic() + 5.0
    while "wiggle" not in teleop.extra_moves() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert teleop.extra_moves() == ["shrug", "wiggle"]  # not the moods, not greeting/kick
    for action in ("mood:wiggle", "mood:shrug", "mood:unknown"):
        teleop._do(action)
    assert Ring.played == ["joy", web.DATASET_MOOD]  # asleep: the ring and sound only, no move
