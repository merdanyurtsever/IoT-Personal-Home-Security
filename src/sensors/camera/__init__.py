"""Camera module for video capture.

Provides a unified interface for different camera backends:
- OpenCV (USB webcams, RTSP streams)
- PiCamera (Raspberry Pi camera module)
"""

from .capture import Camera, CameraBackend, CameraConfig, Frame

__all__ = [
    "Camera",
    "CameraBackend",
    "CameraConfig",
    "Frame",
]
