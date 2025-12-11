"""Unified sensor interfaces: camera, microphone, and motion.

Provides camera access for ARM64 VMs, desktops, and Raspberry Pi with
automatic backend detection. Also includes microphone and PIR motion sensor interfaces.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Camera Configuration
# =============================================================================

@dataclass
class CameraConfig:
    """Camera configuration."""
    device_id: int = 0
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    flip_horizontal: bool = False
    flip_vertical: bool = False
    warmup_frames: int = 5
    auto_exposure: bool = True


# =============================================================================
# Camera Backends
# =============================================================================

class BaseCameraBackend(ABC):
    """Abstract base class for camera backends."""

    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        pass


class OpenCVCameraBackend(BaseCameraBackend):
    """OpenCV VideoCapture backend - works on most platforms."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        try:
            if hasattr(cv2, "CAP_GSTREAMER"):
                backends.insert(1, cv2.CAP_GSTREAMER)
        except Exception:
            pass

        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.device_id, backend)
                if self.cap.isOpened():
                    logger.info(f"Opened camera with backend {backend}")
                    return True
            except Exception as e:
                logger.debug(f"Backend {backend} failed: {e}")
                continue

        self.cap = cv2.VideoCapture(self.device_id)
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def set_resolution(self, width: int, height: int) -> bool:
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return True

    def set_fps(self, fps: int) -> bool:
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        return True


class PiCameraBackend(BaseCameraBackend):
    """Raspberry Pi Camera backend using picamera2."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.camera = None
        self.resolution = (640, 480)

    def open(self) -> bool:
        try:
            from picamera2 import Picamera2

            self.camera = Picamera2()
            config = self.camera.create_still_configuration(main={"size": self.resolution})
            self.camera.configure(config)
            self.camera.start()
            logger.info("Opened Raspberry Pi camera")
            return True
        except ImportError:
            logger.warning("picamera2 not available")
            return False
        except Exception as e:
            logger.error(f"Failed to open Pi camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.camera is None:
            return False, None
        try:
            frame = self.camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return False, None

    def close(self) -> None:
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception:
                pass
            self.camera = None

    def is_opened(self) -> bool:
        return self.camera is not None

    def set_resolution(self, width: int, height: int) -> bool:
        self.resolution = (width, height)
        return True


# =============================================================================
# Unified Camera Interface
# =============================================================================

class CameraInterface:
    """Unified camera interface with automatic backend selection."""

    def __init__(self, config: Optional[CameraConfig] = None, use_picamera: bool = False):
        self.config = config or CameraConfig()
        self.use_picamera = use_picamera
        self.backend: Optional[BaseCameraBackend] = None
        self._frame_count = 0
        self._start_time = 0.0
        self._current_fps = 0.0
        self._lock = threading.Lock()
        self.is_running = False

    def _detect_backend(self) -> BaseCameraBackend:
        is_raspberry_pi = False
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                is_raspberry_pi = "raspberry" in model
        except FileNotFoundError:
            pass

        if self.use_picamera or is_raspberry_pi:
            try:
                backend = PiCameraBackend(self.config.device_id)
                if backend.open():
                    backend.close()
                    logger.info("Using Raspberry Pi camera backend")
                    return PiCameraBackend(self.config.device_id)
            except Exception:
                pass

        logger.info("Using OpenCV camera backend")
        return OpenCVCameraBackend(self.config.device_id)

    def start(self) -> bool:
        if self.backend is not None and self.backend.is_opened():
            return True

        self.backend = self._detect_backend()

        if not self.backend.open():
            logger.error("Failed to open camera")
            return False

        self.backend.set_resolution(*self.config.resolution)
        if isinstance(self.backend, OpenCVCameraBackend):
            self.backend.set_fps(self.config.fps)

        for _ in range(self.config.warmup_frames):
            self.backend.read()

        self._start_time = time.time()
        self._frame_count = 0
        self.is_running = True
        logger.info(f"Camera started: {self.config.resolution[0]}x{self.config.resolution[1]} @ {self.config.fps}fps")
        return True

    def stop(self) -> None:
        if self.backend:
            self.backend.close()
            self.backend = None
        self.is_running = False
        logger.info("Camera stopped")

    def capture(self) -> np.ndarray:
        if self.backend is None or not self.backend.is_opened():
            raise RuntimeError("Camera not started")

        with self._lock:
            ret, frame = self.backend.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture frame")

            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)
            if self.config.flip_vertical:
                frame = cv2.flip(frame, 0)

            self._frame_count += 1
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._current_fps = self._frame_count / elapsed

            return frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        while self.backend and self.backend.is_opened():
            try:
                yield self.capture()
            except RuntimeError:
                time.sleep(0.01)

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.config.resolution

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# Alias for backward compatibility
UniversalCamera = CameraInterface


def list_cameras(max_cameras: int = 10) -> list:
    """List available cameras."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


# =============================================================================
# Microphone Interface
# =============================================================================

class MicrophoneInterface:
    """Microphone interface using PyAudio."""

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.audio = None
        self._stream = None
        self.is_running = False

    def start(self) -> None:
        try:
            import pyaudio

            self.audio = pyaudio.PyAudio()
            self._stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )
            self.is_running = True
        except ImportError:
            raise ImportError("PyAudio is required. Install with: pip install pyaudio")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None
        self.is_running = False

    def read(self, duration: float) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("Microphone not started")

        num_samples = int(self.sample_rate * duration)
        num_chunks = num_samples // self.chunk_size + 1
        frames = []
        for _ in range(num_chunks):
            data = self._stream.read(self.chunk_size)
            frames.append(np.frombuffer(data, dtype=np.float32))
        audio = np.concatenate(frames)[:num_samples]
        return audio

    def stream(self, chunk_duration: float = 0.1) -> Generator[np.ndarray, None, None]:
        if self._stream is None:
            self.start()
        chunk_samples = int(self.sample_rate * chunk_duration)
        try:
            while True:
                data = self._stream.read(chunk_samples)
                yield np.frombuffer(data, dtype=np.float32)
        finally:
            self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def list_devices() -> list:
        try:
            import pyaudio

            audio = pyaudio.PyAudio()
            devices = []
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append({
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                    })
            audio.terminate()
            return devices
        except ImportError:
            return []


# =============================================================================
# Motion Sensor Interface (PIR)
# =============================================================================

class BaseMotionSensor(ABC):
    """Abstract base class for motion sensors."""

    @abstractmethod
    def read(self) -> bool:
        """Read motion sensor state. Returns True if motion detected."""
        pass

    @abstractmethod
    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        """Wait for motion. Returns True if detected, False if timeout."""
        pass

    @abstractmethod
    def on_motion(self, callback: Callable[[], None]) -> None:
        """Register callback for motion detection."""
        pass


class GPIOMotionSensor(BaseMotionSensor):
    """PIR motion sensor using Raspberry Pi GPIO."""

    def __init__(self, pin: int = 17, pull_up: bool = False):
        self.pin = pin
        self.pull_up = pull_up
        self.sensor = None
        self._callback = None
        self._setup_gpio()

    def _setup_gpio(self) -> None:
        try:
            from gpiozero import MotionSensor as GPIOZeroMotionSensor
            self.sensor = GPIOZeroMotionSensor(self.pin, pull_up=self.pull_up)
        except ImportError:
            self.sensor = None
            logger.warning(f"gpiozero not available. Motion sensor on pin {self.pin} simulated.")

    def read(self) -> bool:
        return self.sensor.motion_detected if self.sensor else False

    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        return self.sensor.wait_for_motion(timeout=timeout) if self.sensor else False

    def on_motion(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        if self.sensor:
            self.sensor.when_motion = callback

    def on_no_motion(self, callback: Callable[[], None]) -> None:
        if self.sensor:
            self.sensor.when_no_motion = callback


class MockMotionSensor(BaseMotionSensor):
    """Mock motion sensor for testing without hardware."""

    def __init__(self, pin: int = 17):
        self.pin = pin
        self._motion_detected = False
        self._callback = None

    def read(self) -> bool:
        return self._motion_detected

    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        start = time.time()
        while True:
            if self._motion_detected:
                return True
            if timeout and (time.time() - start) >= timeout:
                return False
            time.sleep(0.1)

    def on_motion(self, callback: Callable[[], None]) -> None:
        self._callback = callback

    def simulate_motion(self) -> None:
        self._motion_detected = True
        if self._callback:
            self._callback()

    def simulate_no_motion(self) -> None:
        self._motion_detected = False


class MotionSensor:
    """Unified motion sensor interface with automatic backend selection."""

    def __init__(self, pin: int = 17, cooldown: float = 5.0, use_mock: bool = False):
        self.pin = pin
        self.cooldown = cooldown
        self.last_motion_time = 0.0

        if use_mock:
            self.sensor = MockMotionSensor(pin)
        else:
            try:
                self.sensor = GPIOMotionSensor(pin)
            except Exception:
                logger.info("Falling back to mock motion sensor")
                self.sensor = MockMotionSensor(pin)

    def read(self) -> bool:
        """Read motion sensor state with cooldown."""
        if time.time() - self.last_motion_time < self.cooldown:
            return False
        motion = self.sensor.read()
        if motion:
            self.last_motion_time = time.time()
        return motion

    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        return self.sensor.wait_for_motion(timeout)

    def on_motion(self, callback: Callable[[], None]) -> None:
        def cooldown_callback():
            if time.time() - self.last_motion_time >= self.cooldown:
                self.last_motion_time = time.time()
                callback()
        self.sensor.on_motion(cooldown_callback)