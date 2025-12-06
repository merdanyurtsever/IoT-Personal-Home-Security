"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale test image."""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


@pytest.fixture
def sample_audio():
    """Create a sample audio signal (5 seconds at 22050 Hz)."""
    duration = 5.0
    sample_rate = 22050
    samples = int(duration * sample_rate)
    return np.random.randn(samples).astype(np.float32)


@pytest.fixture
def sample_audio_short():
    """Create a short audio signal (1 second at 22050 Hz)."""
    duration = 1.0
    sample_rate = 22050
    samples = int(duration * sample_rate)
    return np.random.randn(samples).astype(np.float32)


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        "face_detection": {
            "enabled": True,
            "model": "haar_cascade",
            "confidence_threshold": 0.8,
        },
        "face_recognition": {
            "enabled": True,
            "model": "facenet",
            "threshold": 0.6,
        },
        "sound_classification": {
            "enabled": True,
            "sample_rate": 22050,
            "confidence_threshold": 0.7,
            "target_classes": ["glass_breaking", "door_wood_knock", "dog"],
        },
        "camera": {
            "enabled": True,
            "resolution": [640, 480],
            "fps": 30,
        },
        "microphone": {
            "enabled": True,
            "sample_rate": 44100,
            "channels": 1,
        },
        "motion_sensor": {
            "enabled": True,
            "gpio_pin": 17,
            "cooldown_seconds": 5,
        },
        "alerts": {
            "local_alarm": {
                "enabled": True,
                "gpio_pin": 18,
                "duration_seconds": 10,
            },
        },
    }
