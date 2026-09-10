"""Audio playback and recording for Chromapi.

Optional dependency (``pip install chromapi[audio]``, i.e. ``sounddevice``) - every function
here is best-effort: a missing dependency, no audio device (headless CI, SSH without audio
forwarding, ...), or a bad file all log a warning and return ``False``/``None``, never raise. A
sound cue (or a failed recording) is not something that should ever take down a control loop.
"""

from __future__ import annotations

import logging
import threading
import wave
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

#: Maps WAV sample width (bytes) to the numpy dtype ``wave``'s raw frames decode to.
_DTYPE_BY_SAMPLE_WIDTH: Dict[int, "npt.DTypeLike"] = {1: np.uint8, 2: np.int16, 4: np.int32}


#: Full-scale divisor for each integer sample width ``wave`` can hand back, to convert to a
#: float32 [-1, 1] range before applying :func:`play_wav`'s ``volume`` gain. ``uint8`` PCM is
#: offset-encoded (silence = 128, not 0), unlike the signed 16/32-bit formats.
_FULL_SCALE_BY_SAMPLE_WIDTH: Dict[int, float] = {1: 128.0, 2: 32768.0, 4: 2147483648.0}


def _to_float32(samples: npt.NDArray[Any], sample_width: int) -> npt.NDArray[np.float32]:
    """Convert raw PCM samples (as decoded by ``wave``) to float32 in [-1, 1]."""
    full_scale = _FULL_SCALE_BY_SAMPLE_WIDTH[sample_width]
    offset = 128.0 if sample_width == 1 else 0.0
    return ((samples.astype(np.float32) - offset) / full_scale).astype(np.float32)


def _device_default_samplerate(sd: Any, device: Union[int, str, None]) -> Optional[int]:
    """Best-effort lookup of an output device's native sample rate, or ``None`` if unknown."""
    try:
        info = sd.query_devices(device, kind="output") if device is not None else sd.query_devices(kind="output")
        rate = info.get("default_samplerate")
        return int(round(rate)) if rate else None
    except Exception:
        return None


def _resample_to(samples: npt.NDArray[np.float32], orig_rate: int, target_rate: int) -> npt.NDArray[np.float32]:
    """Resample (mono or multi-channel, shape ``(n,)``/``(n, channels)``) float32 audio."""
    if orig_rate == target_rate:
        return samples
    divisor = gcd(orig_rate, target_rate)
    up, down = target_rate // divisor, orig_rate // divisor
    return resample_poly(samples, up, down, axis=0).astype(np.float32)


def _decode_wav(path: Union[str, Path], volume: float) -> Optional[Any]:
    """Decode a .wav file to float32 [-1, 1] samples, applying ``volume``."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
    except (OSError, wave.Error) as exc:
        logger.warning("_decode_wav(%s): could not read the file (%s)", path, exc)
        return None

    dtype = _DTYPE_BY_SAMPLE_WIDTH.get(sample_width)
    if dtype is None:
        logger.warning(
            "_decode_wav(%s): unsupported sample width (%d bytes)", path, sample_width
        )
        return None
    flat_samples = np.frombuffer(raw, dtype=dtype)
    samples: npt.NDArray[Any] = (
        flat_samples.reshape(-1, n_channels) if n_channels > 1 else flat_samples
    )
    scaled = np.clip(_to_float32(samples, sample_width) * volume, -1.0, 1.0)
    return scaled, sample_rate, n_channels


def play_wav(
    path: Union[str, Path],
    wait: bool = True,
    volume: float = 0.5,
    device: Union[int, str, None] = None,
) -> bool:
    """Play a .wav file through an output device by opening a stream, playing, and closing it.

    Args:
        path: Path to a .wav file.
        wait: If True (default), block until playback finishes. If False, start playback and
            return immediately - e.g. ``Chromapi.wake_up(play_sound=True)`` firing the cue
            without delaying the stand-up transition.
        volume: Linear gain applied before playback, defaults to 0.5.
        device: Passed straight to ``sounddevice`` (index, name substring, or ``None`` for the
            system default output).

    Returns:
        True if playback started, False if it was skipped.

    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.warning(
            "play_wav(%s): 'sounddevice' not installed (pip install chromapi[audio]) - "
            "skipping playback",
            path,
        )
        return False

    decoded = _decode_wav(path, volume)
    if decoded is None:
        return False
    scaled, sample_rate, _n_channels = decoded

    play_rate = sample_rate
    device_rate = _device_default_samplerate(sd, device)
    if device_rate and device_rate != sample_rate:
        try:
            scaled = _resample_to(scaled, sample_rate, device_rate)
            play_rate = device_rate
        except Exception:
            logger.exception(
                "play_wav(%s): resampling %d Hz -> %d Hz failed - playing at the file's own rate instead",
                path,
                sample_rate,
                device_rate,
            )

    try:
        sd.play(scaled, samplerate=play_rate, device=device)
        if wait:
            sd.wait()
    except Exception:
        logger.exception("play_wav(%s): playback failed - no audio device available?", path)
        return False
    return True


def find_input_device(name_hint: str) -> Optional[int]:
    """Return the index of the first input-capable device whose name contains ``name_hint``."""
    try:
        import sounddevice as sd
    except ImportError:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        logger.exception("find_input_device(%r): could not query audio devices", name_hint)
        return None

    hint = name_hint.lower()
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0 and hint in str(device.get("name", "")).lower():
            return index
    logger.warning(
        "find_input_device(%r): no input device matched - falling back to the system default",
        name_hint,
    )
    return None


def find_output_device(name_hint: str) -> Optional[int]:
    """Return the index of the first output-capable device whose name contains ``name_hint``."""
    try:
        import sounddevice as sd
    except ImportError:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        logger.exception("find_output_device(%r): could not query audio devices", name_hint)
        return None

    hint = name_hint.lower()
    for index, device in enumerate(devices):
        if device.get("max_output_channels", 0) > 0 and hint in str(device.get("name", "")).lower():
            return index
    logger.warning(
        "find_output_device(%r): no output device matched - falling back to the system default",
        name_hint,
    )
    return None


def _match_channels(samples: npt.NDArray[np.float32], n_channels: int, target_channels: int) -> npt.NDArray[np.float32]:
    """Adapt ``samples`` (``n_channels`` wide) to ``target_channels`` - mono <-> N only."""
    if n_channels == target_channels:
        return samples
    if n_channels == 1 and target_channels > 1:
        mono = samples.reshape(-1, 1) if samples.ndim == 1 else samples
        return np.tile(mono, (1, target_channels)).astype(np.float32)
    if target_channels == 1 and n_channels > 1:
        return samples.mean(axis=1).astype(np.float32)
    logger.warning(
        "_match_channels: cannot adapt %d-channel audio to %d channels - playing as-is",
        n_channels,
        target_channels,
    )
    return samples


class PersistentAudioOutput:
    """Plays .wav files through one long-lived, callback-driven ``sd.OutputStream``.

    This is more efficient than :func:`play_wav` if many short clips are played back-to-back.

    """

    def __init__(
        self, samplerate: Optional[int] = None, channels: int = 1, device: Union[int, str, None] = None
    ) -> None:
        """Configure the output - opens no stream until :meth:`start`."""
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self._stream: Optional[Any] = None
        self._lock = threading.Lock()
        self._pending: npt.NDArray[np.float32] = np.zeros((0, channels), dtype=np.float32)

    def start(self) -> bool:
        """Open the callback-driven output stream."""
        if self._stream is not None:
            return False
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("PersistentAudioOutput.start(): 'sounddevice' not installed - skipping")
            return False
        try:
            self._stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception:
            logger.exception("PersistentAudioOutput.start(): could not open the output device")
            self._stream = None
            return False
        self.samplerate = int(round(self._stream.samplerate))
        return True

    def _callback(self, outdata: npt.NDArray[np.float32], frames: int, time_info: object, status: object) -> None:
        """PortAudio calls this from its own real-time thread whenever it needs ``frames`` more."""
        if status:
            logger.warning("PersistentAudioOutput: %s", status)
        with self._lock:
            available = len(self._pending)
            n = min(frames, available)
            if n > 0:
                outdata[:n] = self._pending[:n]
                self._pending = self._pending[n:]
            if n < frames:
                outdata[n:] = 0.0

    def play(self, path: Union[str, Path], volume: float = 0.4) -> bool:
        """Decode ``path``, adapt it to this stream's rate/channels, and queue it for playback."""
        if self._stream is None:
            logger.warning("PersistentAudioOutput.play(%s): not started - skipping", path)
            return False
        decoded = _decode_wav(path, volume)
        if decoded is None:
            return False
        samples, sample_rate, n_channels = decoded
        if sample_rate != self.samplerate:
            try:
                samples = _resample_to(samples, sample_rate, self.samplerate)
            except Exception:
                logger.exception(
                    "PersistentAudioOutput.play(%s): resampling %d Hz -> %d Hz failed - skipping",
                    path,
                    sample_rate,
                    self.samplerate,
                )
                return False
        samples = _match_channels(samples, n_channels, self.channels)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        with self._lock:
            self._pending = np.concatenate([self._pending, samples], axis=0)
        return True

    def stop(self) -> None:
        """Close the stream and drop any pending audio."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("PersistentAudioOutput.stop(): error closing the output stream")
            self._stream = None
        with self._lock:
            self._pending = np.zeros((0, self.channels), dtype=np.float32)


def _write_pcm16_wav(path: Union[str, Path], samples: npt.NDArray[np.float32], samplerate: int, channels: int) -> None:
    """Write a float32 [-1, 1] array (shape ``(n,)`` or ``(n, channels)``) as 16-bit PCM."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(samplerate)
        wav_file.writeframes(pcm.tobytes())


def record_wav(
    path: Union[str, Path],
    duration_s: float,
    samplerate: int = 44100,
    channels: int = 1,
    device: Union[int, str, None] = None,
    gain: float = 1.0,
) -> bool:
    """Record a fixed ``duration_s`` from an input device straight to a .wav file.

    Args:
        path: Output .wav path (overwritten if it exists).
        duration_s: How long to record, in seconds.
        samplerate: Sample rate, in Hz.
        channels: Number of input channels to capture.
        device: Passed straight to ``sounddevice`` (index, name substring, or ``None`` for the system default input).
        gain: Linear gain applied to the recorded samples before writing to disk.

    Returns:
        True if a file was written, False if recording was skipped/failed.

    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.warning(
            "record_wav(%s): 'sounddevice' not installed (pip install chromapi[audio]) - "
            "skipping recording",
            path,
        )
        return False

    try:
        raw = sd.rec(
            int(duration_s * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception:
        logger.exception("record_wav(%s): recording failed - no input device available?", path)
        return False

    try:
        _write_pcm16_wav(path, raw * gain, samplerate, channels)
    except OSError:
        logger.exception("record_wav(%s): could not write the file", path)
        return False
    return True


class PushToTalkRecorder:
    """Variable-length recording: :meth:`start` on button-down, :meth:`stop` on button-up."""

    def __init__(
        self, samplerate: int = 44100, channels: int = 1, device: Union[int, str, None] = None, gain: float = 1.0
    ) -> None:
        """Configure the recorder - opens no device until :meth:`start`."""
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.gain = gain
        self._stream: Optional[Any] = None
        self._chunks: List[npt.NDArray[np.float32]] = []
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Open the input stream and begin buffering."""
        if self._stream is not None:
            return False
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("PushToTalkRecorder.start(): 'sounddevice' not installed - skipping")
            return False

        self._chunks = []

        def _callback(indata: npt.NDArray[np.float32], frames: int, time_info: object, status: object) -> None:
            """PortAudio calls this from its own real-time thread whenever it has ``frames`` more samples."""
            if status:
                logger.warning("PushToTalkRecorder: %s", status)
            with self._lock:
                self._chunks.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=_callback,
            )
            self._stream.start()
        except Exception:
            logger.exception("PushToTalkRecorder.start(): could not open the input device")
            self._stream = None
            return False
        return True

    def stop(self) -> Optional[npt.NDArray[np.float32]]:
        """Stop recording and return the captured samples."""
        if self._stream is None:
            return None
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            logger.exception("PushToTalkRecorder.stop(): error closing the input stream")
        self._stream = None

        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0) * self.gain

    def stop_to_wav(self, path: Union[str, Path]) -> bool:
        """Stop recording and write the result straight to ``path``."""
        samples = self.stop()
        if samples is None or len(samples) == 0:
            logger.warning("PushToTalkRecorder.stop_to_wav(%s): nothing was recorded", path)
            return False
        try:
            _write_pcm16_wav(path, samples, self.samplerate, self.channels)
        except OSError:
            logger.exception("PushToTalkRecorder.stop_to_wav(%s): could not write the file", path)
            return False
        return True

    @property
    def is_recording(self) -> bool:
        """Whether a recording is currently in progress."""
        return self._stream is not None
