"""Sensor interface modules for Raspberry Pi."""

from .camera import CameraInterface
from .microphone import MicrophoneInterface
from .motion import MotionSensor

__all__ = ["CameraInterface", "MicrophoneInterface", "MotionSensor"]
