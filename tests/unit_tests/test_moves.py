# SPDX-License-Identifier: Apache-2.0
"""Recorded moves read from a folder or a Hugging Face dataset: what check_move refuses, the offline cache."""

import copy
import json
import sys
import types

import pytest

from chromapi.locomotion.recorded_move import MOVES_DIR, RecordedMoves, check_move

YES = json.loads((MOVES_DIR / "yes.json").read_text())
KICK = json.loads((MOVES_DIR / "kick.json").read_text())


def _with(move, **changes):
    move = copy.deepcopy(move)
    move.update(changes)
    return move


def _scaled(move, key, k):
    return _with(move, set_target_data=[{key: [v * k for v in s[key]]} for s in move["set_target_data"]])


def test_every_bundled_move_passes():
    """The bundled moves pass, and all of them load."""
    for path in sorted(MOVES_DIR.glob("*.json")):
        check_move(json.loads(path.read_text()))
    assert len(RecordedMoves().list_moves()) == len(list(MOVES_DIR.glob("*.json")))


@pytest.mark.parametrize("bad", [
    _with(YES, code="__import__('os')"),  # unknown key
    _with(YES, description=""),
    _with(YES, time=YES["time"][:-1]),  # one sample without its time
    _with(YES, time=[t * 50 for t in YES["time"]]),  # 70 s, beyond 60 s
    _with(YES, time=["0"] + YES["time"][1:]),  # not a number
    _scaled(YES, "body", 3.0),  # beyond 0.2 rad
    _with(_scaled(YES, "body", 1.5), time=[t / 4 for t in YES["time"]]),  # too fast
    _with(YES, set_target_data=YES["set_target_data"][:-1] + [{"body": [0, 0, 0, 0.1, 0, 0]}]),  # does not end at rest
    _with(YES, set_target_data=[{"leg": [0, 0, 0]}] + YES["set_target_data"][1:]),  # body and leg mixed
    _with(KICK, default_leg="br"),  # a back leg
    _scaled(KICK, "leg", 3.0),  # beyond 1.2 rad
])
def test_check_move_refuses(bad):
    """Unknown keys, bad times, too far, too fast, not back at rest, mixed or back-leg moves: refused."""
    with pytest.raises(ValueError):
        check_move(bad)


def test_a_bad_file_is_left_out_the_others_load(tmp_path):
    """A refused, broken or non-JSON file is not listed; the good one is."""
    (tmp_path / "yes.json").write_text(json.dumps(YES))
    (tmp_path / "huge.json").write_text(json.dumps(_scaled(YES, "body", 3.0)))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "payload.pkl").write_bytes(b"never read")
    assert RecordedMoves(tmp_path).list_moves() == ["yes"]


def test_dataset_from_the_cache_when_offline(tmp_path, monkeypatch):
    """No network: the dataset comes from the cache, JSON only, and joins the bundled moves."""
    (tmp_path / "wiggle.json").write_text(json.dumps(_with(YES, mood="joy")))
    calls = []

    def snapshot_download(repo_id, repo_type, allow_patterns, local_files_only=False):
        calls.append((repo_id, repo_type, allow_patterns, local_files_only))
        if not local_files_only:
            raise OSError("no network")
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    moves = RecordedMoves("someone/chromaposes")
    assert moves.list_moves() == ["wiggle"] and moves.mood("wiggle") == "joy"
    assert calls == [("someone/chromaposes", "dataset", ["*.json"], False),
                     ("someone/chromaposes", "dataset", ["*.json"], True)]
    bundled = RecordedMoves()
    bundled.update(moves)
    assert "wiggle" in bundled.list_moves() and "yes" in bundled.list_moves()


def test_lifted_leg_stays_within_the_joint_limits():
    """A lifted leg's joints are clamped to the model's limits."""
    from chromapi.locomotion.balance import BalanceMove

    body = BalanceMove(dt=0.01)
    body.lift("tl", {"tl_1": 5.0, "tl_2": -5.0, "tl_3": 0.1})
    lifted = body._lifted[1]
    assert lifted["tl_1"] == pytest.approx(body._kin.joint_limits("tl_1")[1])
    assert lifted["tl_2"] == pytest.approx(body._kin.joint_limits("tl_2")[0])
    assert lifted["tl_3"] == 0.1
