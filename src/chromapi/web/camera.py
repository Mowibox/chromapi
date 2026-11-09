# SPDX-License-Identifier: Apache-2.0
"""Camera for the web app: fisheye-corrected MJPEG stream, and recording to a file on the robot."""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STREAM_JPEG_QUALITY = 60
RECORD_FPS = 30.0  # [Hz]
RECORD_JPEG_QUALITY = 95
IDLE_CLOSE_S = 2.0  # [s]


class CameraStreamer:
    """Background capture thread; :meth:`mjpeg` feeds an HTTP multipart response."""

    def __init__(self, camera_config: Dict[str, Any], device: int = 0) -> None:
        """``camera_config``: the ``camera:`` section of the config (fisheye calibration), may be empty."""
        self.device = device
        self._maps: Optional[Tuple[Any, Any]] = None
        self._map_size: Optional[Tuple[int, int]] = None
        if camera_config.get("model") == "fisheye":
            K, D = np.array(camera_config["camera_matrix"]), np.array(camera_config["dist_coeffs"])
            w, h = camera_config["resolution"]
            self._maps = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
            self._map_size = (w, h)
        else:
            logger.info("No fisheye calibration in the config: raw camera stream")
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._image: Optional[np.ndarray] = None  # latest corrected frame, for photos
        self._viewers = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rec_request: Optional[Path] = None
        self._rec_stop = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._rec_path: Optional[Path] = None
        self._rec_t0 = 0.0  # [s]
        self._rec_frames = 0
        self.last_recording: Optional[Path] = None

    def start(self) -> None:
        """Start the capture thread."""
        self._thread = threading.Thread(target=self._loop, name="chromapi-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing, closing any recording."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def start_recording(self, path: Path) -> None:
        """Record the corrected stream to ``path`` (MJPG .avi), until :meth:`stop_recording`."""
        with self._lock:
            if self._writer is None:
                self._rec_request = path

    def stop_recording(self) -> None:
        """Close the current recording."""
        with self._lock:
            self._rec_stop = True

    def photo(self, path: Path) -> Optional[Path]:
        """Save the latest corrected frame to ``path`` (JPEG); None if no frame yet (nobody watching)."""
        with self._lock:
            image = self._image
        if image is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, RECORD_JPEG_QUALITY])
        return path

    @property
    def viewers(self) -> int:
        """Streams being watched right now (the camera is on while > 0, or while recording)."""
        with self._lock:
            return self._viewers

    def _wanted(self) -> bool:
        with self._lock:
            return self._viewers > 0 or self._writer is not None or self._rec_request is not None

    def recording_status(self) -> Dict[str, Any]:
        """``on``, ``seconds`` recorded, ``last`` file name."""
        with self._lock:
            return {
                "on": self._writer is not None or self._rec_request is not None,
                "seconds": self._rec_frames / RECORD_FPS,
                "last": self.last_recording.name if self.last_recording else None,
            }

    def _process(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, -1)  # the camera is mounted upside down
        if self._maps is not None and (frame.shape[1], frame.shape[0]) == self._map_size:
            map1, map2 = self._maps
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return frame

    def _record(self, frame: np.ndarray, now: float) -> None:
        """Write ``frame`` as often as the 30 fps clock asks, so the file plays at real speed."""
        assert self._writer is not None
        due = int((now - self._rec_t0) * RECORD_FPS) + 1
        while self._rec_frames < due:
            self._writer.write(frame)
            self._rec_frames += 1

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("Recording saved: %s (%.1f s)", self._rec_path, self._rec_frames / RECORD_FPS)
            with self._lock:
                self._writer = None
                self.last_recording = self._rec_path

    def _loop(self) -> None:
        """Open the camera when wanted, capture, close it once nobody watches nor records."""
        while not self._stop.is_set():
            if not self._wanted():
                self._stop.wait(0.1)
                continue
            cap = cv2.VideoCapture(self.device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, RECORD_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                logger.warning("Cannot open /dev/video%d: no camera stream", self.device)
                return
            logger.info("Camera on")
            try:
                self._capture(cap)
            finally:
                self._close_writer()
                cap.release()
                with self._lock:
                    self._jpeg, self._image = None, None
                logger.info("Camera off")

    def _capture(self, cap: Any) -> None:
        idle_since: Optional[float] = None  # [s]
        while not self._stop.is_set():
            now = time.monotonic()
            if self._wanted():
                idle_since = None
            elif idle_since is None:
                idle_since = now
            elif now - idle_since > IDLE_CLOSE_S:
                return
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            with self._lock:
                watching = self._viewers > 0
                request, self._rec_request = self._rec_request, None
                stop_rec, self._rec_stop = self._rec_stop, False
            if stop_rec:
                self._close_writer()
            if request is not None:
                request.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(request), cv2.VideoWriter.fourcc(*"MJPG"), RECORD_FPS,
                                         (frame.shape[1], frame.shape[0]))
                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, RECORD_JPEG_QUALITY)
                with self._lock:
                    self._writer, self._rec_path, self._rec_t0, self._rec_frames = writer, request, now, 0
            if not watching and self._writer is None:
                continue
            image = self._process(frame)
            if self._writer is not None:
                self._record(image, now)
            if watching:
                ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if ok:
                    with self._lock:
                        self._jpeg, self._image = buf.tobytes(), image

    def mjpeg(self) -> Iterator[bytes]:
        """Multipart JPEG frames, for as long as the client reads them."""
        with self._lock:
            self._viewers += 1
        try:
            last = None
            while not self._stop.is_set():
                with self._lock:
                    frame = self._jpeg
                if frame is not None and frame is not last:
                    last = frame
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.02)
        finally:
            with self._lock:
                self._viewers -= 1
