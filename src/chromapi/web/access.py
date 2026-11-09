# SPDX-License-Identifier: Apache-2.0
"""Who may use the web app: an access code chosen by the first visitor, and the HTTPS certificate.

Usage::

    chromapi access reset     # forget the code: the next visitor chooses a new one
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chromapi.chromapi import USER_CALIBRATION_PATH

logger = logging.getLogger(__name__)

CONFIG_DIR = USER_CALIBRATION_PATH.parent
ACCESS_PATH = CONFIG_DIR / "access.json"
TLS_DIR = CONFIG_DIR / "tls"

MIN_CODE_LENGTH = 6
MAX_CODE_LENGTH = 64
FREE_ATTEMPTS = 5  # wrong codes before the delays start
FIRST_DELAY_S = 2.0  # [s] then doubled at every further wrong code
MAX_DELAY_S = 300.0  # [s]
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def _hash(code: str, salt: bytes) -> str:
    return hashlib.scrypt(code.encode(), salt=salt, **_SCRYPT).hex()


class Access:
    """The access code and the session key, re-read on every call."""

    def __init__(self, path: Path = ACCESS_PATH) -> None:
        """Create the file with a fresh session key if it does not exist yet."""
        self.path = path
        self._lock = threading.Lock()
        self._failures = 0
        self._locked_until = 0.0  # [s] monotonic
        if not self._load():
            self._save({"secret_key": secrets.token_hex(32), "generation": secrets.token_hex(8)})

    def _load(self) -> Dict[str, Any]:
        try:
            data: Dict[str, Any] = json.loads(self.path.read_text())
            return data
        except (OSError, ValueError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # owner-only from the start
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    @property
    def secret_key(self) -> str:
        """Signs the session cookies; specific to this robot."""
        return str(self._load()["secret_key"])

    @property
    def generation(self) -> str:
        """Changes with the code, invalidating older sessions."""
        return str(self._load()["generation"])

    @property
    def configured(self) -> bool:
        """An access code has been chosen."""
        return "code_hash" in self._load()

    def set_code(self, code: str) -> None:
        """Store the hashed ``code`` and start a new generation: every browser is signed out."""
        with self._lock:
            self._store(code)

    def choose(self, code: str) -> bool:
        """The first visitor's code; False if one is already set."""
        with self._lock:
            if self.configured:
                return False
            self._store(code)
            return True

    def _store(self, code: str) -> None:
        if not MIN_CODE_LENGTH <= len(code) <= MAX_CODE_LENGTH:
            raise ValueError(f"the access code needs {MIN_CODE_LENGTH} to {MAX_CODE_LENGTH} characters")
        salt = secrets.token_bytes(16)
        data = self._load()
        data.update(salt=salt.hex(), code_hash=_hash(code, salt), generation=secrets.token_hex(8))
        self._save(data)
        self._failures, self._locked_until = 0, 0.0

    def reset(self) -> None:
        """Forget the code (the next visitor chooses one) and sign every browser out."""
        with self._lock:
            data = self._load()
            for key in ("salt", "code_hash"):
                data.pop(key, None)
            data["generation"] = secrets.token_hex(8)
            self._save(data)

    def check(self, code: str) -> Tuple[bool, float]:
        """(right code, seconds to wait before the next try)."""
        with self._lock:
            now = time.monotonic()
            if now < self._locked_until:
                return False, self._locked_until - now
            data = self._load()
            if "code_hash" not in data:
                return False, 0.0
            ok = hmac.compare_digest(_hash(code, bytes.fromhex(data["salt"])), data["code_hash"])
            if ok:
                self._failures = 0
                return True, 0.0
            self._failures += 1
            wait = 0.0
            if self._failures >= FREE_ATTEMPTS:
                wait = min(MAX_DELAY_S, FIRST_DELAY_S * 2 ** (self._failures - FREE_ATTEMPTS))
                self._locked_until = now + wait
            logger.warning("Wrong access code (%d in a row)", self._failures)
            return False, wait


def tls_certificate(directory: Path = TLS_DIR) -> Optional[Tuple[str, str]]:
    """(certificate, key) paths of this robot's self-signed certificate, made on first use; None without openssl."""
    host = socket.gethostname()
    cert, key = directory / f"{host}.cert.pem", directory / f"{host}.key.pem"
    if cert.exists() and key.exists():
        return str(cert), str(key)
    directory.mkdir(parents=True, exist_ok=True)
    names = f"DNS:{host}.local,DNS:{host},DNS:chromapi.local,DNS:localhost,IP:127.0.0.1"
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
             "-days", "3650", "-subj", f"/CN={host}.local", "-addext", f"subjectAltName={names}",
             "-keyout", str(key), "-out", str(cert)],
            check=True, capture_output=True, timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("Cannot make the HTTPS certificate (%s): serving plain HTTP", exc)
        return None
    key.chmod(0o600)
    logger.info("HTTPS certificate created for %s.local", host)
    return str(cert), str(key)


def reset_main() -> int:
    """``chromapi access reset``."""
    parser = argparse.ArgumentParser(description="Forget the web app's access code: the next visitor chooses one.")
    parser.parse_args()
    Access().reset()
    print("Access code forgotten: every browser is signed out, the next one to open the app chooses a new code.")
    return 0
