"""Camera interface module.

This module provides camera access for both development (OpenCV)
and Raspberry Pi (PiCamera2) environments.
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import numpy as np


class BaseCameraInterface(ABC):
    """Abstract base class for camera interfaces."""
    
    @abstractmethod
    def start(self) -> None:
        """Start the camera."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera."""
        pass
    
    @abstractmethod
    def capture(self) -> np.ndarray:
        """Capture a single frame.
        
        Returns:
            BGR image as numpy array
        """
        pass
    
    @abstractmethod
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Stream frames from camera.
        
        Yields:
            BGR frames as numpy arrays
        """
        pass


class OpenCVCamera(BaseCameraInterface):
    """Camera interface using OpenCV (for development/testing)."""
    
    def __init__(
        self,
        device_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        """Initialize OpenCV camera.
        
        Args:
            device_id: Camera device ID
            resolution: Frame resolution (width, height)
            fps: Target frames per second
        """
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
    
    def start(self) -> None:
        """Start the camera."""
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.device_id}")
        except ImportError:
            raise ImportError("OpenCV is required for camera access")
    
    def stop(self) -> None:
        """Stop the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def capture(self) -> np.ndarray:
        """Capture a single frame."""
        if self.cap is None:
            raise RuntimeError("Camera not started")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        return frame
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Stream frames from camera."""
        if self.cap is None:
            self.start()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame
        finally:
            self.stop()


class PiCamera(BaseCameraInterface):
    """Camera interface for Raspberry Pi Camera Module."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        """Initialize Pi Camera.
        
        Args:
            resolution: Frame resolution (width, height)
            fps: Target frames per second
        """
        self.resolution = resolution
        self.fps = fps
        self.camera = None
    
    def start(self) -> None:
        """Start the Pi camera."""
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
        except ImportError:
            raise ImportError(
                "picamera2 is required for Pi Camera. "
                "Install with: pip install picamera2"
            )
    
    def stop(self) -> None:
        """Stop the Pi camera."""
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None
    
    def capture(self) -> np.ndarray:
        """Capture a single frame."""
        if self.camera is None:
            raise RuntimeError("Camera not started")
        
        # Capture and convert RGB to BGR for OpenCV compatibility
        frame = self.camera.capture_array()
        frame = frame[:, :, ::-1]  # RGB to BGR
        return frame
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Stream frames from Pi camera."""
        if self.camera is None:
            self.start()
        
        try:
            while True:
                yield self.capture()
        finally:
            self.stop()


class CameraInterface:
    """Unified camera interface with automatic backend selection."""
    
    def __init__(
        self,
        use_picamera: bool = False,
        device_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        """Initialize camera interface.
        
        Args:
            use_picamera: Use Pi Camera instead of OpenCV
            device_id: Camera device ID (for OpenCV)
            resolution: Frame resolution
            fps: Target FPS
        """
        if use_picamera:
            self.camera = PiCamera(resolution=resolution, fps=fps)
        else:
            self.camera = OpenCVCamera(
                device_id=device_id,
                resolution=resolution,
                fps=fps,
            )
        
        self.is_running = False
    
    def start(self) -> None:
        """Start the camera."""
        self.camera.start()
        self.is_running = True
    
    def stop(self) -> None:
        """Stop the camera."""
        self.camera.stop()
        self.is_running = False
    
    def capture(self) -> np.ndarray:
        """Capture a single frame."""
        return self.camera.capture()
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Stream frames from camera."""
        yield from self.camera.stream()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
