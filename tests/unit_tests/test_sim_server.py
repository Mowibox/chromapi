# SPDX-License-Identifier: Apache-2.0
"""The standalone simulation: a script drives it, a second one is refused, the next one takes over."""

import threading

import pytest

pytest.importorskip("mujoco")

from chromapi.chromapi import MuJoCoBackend, SimClientBackend  # noqa: E402
from chromapi.kinematics import kromatics as K  # noqa: E402
from chromapi.utils.sim_server import SimServer  # noqa: E402


def test_scripts_take_turns_on_the_simulation():
    """One script at a time: the second is refused while the first drives, the third takes over."""
    server = SimServer(MuJoCoBackend(realtime=False), address=("localhost", 0))  # any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()

    first = SimClientBackend(server.address)
    assert first.connect()
    state = first.read_state()
    assert set(state.joint_positions) == set(K.JOINT_NAMES)
    assert first.send_joint_targets({K.JOINT_NAMES[0]: 0.1})
    first.set_led_all(255, 0, 0)

    second = SimClientBackend(server.address)
    assert second.connect()
    with pytest.raises(RuntimeError, match="already driving"):
        second.read_state()
    second.close()

    first.close()
    third = SimClientBackend(server.address)
    assert third.connect()
    for _ in range(100):  # the first script just left: wait for the server to notice
        try:
            assert third.read_state() is not None
            break
        except RuntimeError:
            third.close()
            third.connect()
    else:
        pytest.fail("the next script never got the simulation")
    third.close()
