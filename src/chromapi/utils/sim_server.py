# SPDX-License-Identifier: Apache-2.0
"""Run the MuJoCo simulation on its own: scripts drive it with ``Chromapi(backend="sim-server")``.

The simulation stands in for the motherboard: a script keeps the whole SDK, and can stop and start again
while the simulation (and the robot where it left it) goes on. One script at a time, as on the robot.

Usage::

    chromapi sim                # viewer window: close it to stop
    chromapi sim --follow       # the camera follows the robot
    chromapi sim --headless     # execution without window (Ctrl+C to stop)

then, in another terminal::

    python my_script.py         # with Chromapi(backend="sim-server")
"""

import argparse
import logging
import threading
import time
from multiprocessing.connection import Connection, Listener
from typing import Any, Optional, Tuple

from chromapi.chromapi import SIM_ADDRESS, SIM_AUTHKEY, MuJoCoBackend

logger = logging.getLogger(__name__)

#: What a script may call: the RobotBackend methods, nothing else
CALLS = frozenset({"send_joint_targets", "read_state", "set_torque_enabled", "set_led_all", "set_led", "set_led_bulk"})
IDLE_STEP_S = 0.02  # [s] no script stepping it: the simulation steps itself at this period


class SimServer:
    """Serves one :class:`MuJoCoBackend` to the scripts, one at a time."""

    def __init__(self, backend: MuJoCoBackend, address: Tuple[str, int] = SIM_ADDRESS) -> None:
        """Listen on ``address`` (OSError if taken: another simulation runs)."""
        self.backend = backend
        self._listener = Listener(address, authkey=SIM_AUTHKEY)
        self.address: Tuple[str, int] = self._listener.address  # type: ignore[assignment]  # TCP: (host, port)
        self._lock = threading.Lock()  # the backend: the script's calls and the idle steps
        self._last_step = 0.0  # [s] monotonic
        self._driver: Optional[Connection] = None  # the script driving the simulation

    def call(self, name: str, args: Tuple[Any, ...]) -> Any:
        """The result of ``backend.name(*args)``, or the exception to raise in the script."""
        if name not in CALLS:
            return ValueError(f"not a simulation call: {name!r}")
        with self._lock:
            try:
                result = getattr(self.backend, name)(*args)
            except Exception as exc:  # raised in the script, the simulation goes on
                return exc
            if name == "send_joint_targets":
                self._last_step = time.monotonic()
        return result

    def step_when_idle(self) -> None:
        """Keep the physics going (targets held) while no script steps it."""
        with self._lock:
            if time.monotonic() - self._last_step > IDLE_STEP_S:
                self.backend.send_joint_targets({})
                self._last_step = time.monotonic()

    def serve_forever(self) -> None:
        """Accept the scripts, each in its own thread."""
        while True:
            try:
                conn = self._listener.accept()
            except Exception as exc:  # wrong key, or a connection dropped during the handshake
                logger.warning("Connection refused: %s", exc)
                continue
            threading.Thread(target=self._serve, args=(conn,), name="chromapi-sim-client", daemon=True).start()

    def _serve(self, conn: Connection) -> None:
        with conn:
            if self._driver is not None:
                conn.recv()
                conn.send(RuntimeError("A script is already driving the simulation: stop it first"))
                return
            self._driver = conn
            logger.info("Script connected")
            try:
                while True:
                    name, args = conn.recv()
                    conn.send(self.call(name, args))
            except (EOFError, OSError):
                pass
            except Exception:
                logger.exception("Script dropped: bad request")
            finally:
                self._driver = None
            logger.info("Script left: the simulation goes on")


def main() -> int:
    """Run the simulation until its window is closed (or Ctrl+C)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headless", action="store_true", help="no viewer window")
    parser.add_argument("--follow", action="store_true", help="the camera follows the robot")
    args = parser.parse_args()

    backend = MuJoCoBackend(launch_viewer=not args.headless, follow_robot=args.follow)
    try:
        server = SimServer(backend)
    except OSError as exc:
        print(f"[!] Cannot listen on {SIM_ADDRESS[0]}:{SIM_ADDRESS[1]} ({exc}): is a simulation already running?")
        return 1
    backend.connect()
    threading.Thread(target=server.serve_forever, name="chromapi-sim-server", daemon=True).start()
    print(f'Simulation running on {server.address[0]}:{server.address[1]}: drive it with Chromapi(backend="sim-server")')
    try:
        while backend.is_alive():
            server.step_when_idle()
            time.sleep(IDLE_STEP_S / 2)
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
