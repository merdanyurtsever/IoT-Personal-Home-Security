"""Sensors module.

Contains:
- Camera: Video capture backends
"""

from .camera import Camera, CameraConfig, CameraBackend, Frame

__all__ = [
    "Camera",
    "CameraConfig",
    "CameraBackend",
    "Frame",
]
