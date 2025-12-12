"""Centralized constants and configuration loader.

This module provides access to configuration values and sensible defaults
for all processing constants used throughout the application. Values are
loaded from config/config.yaml when available, otherwise defaults are used.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Path to the default configuration file
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Uses default if None.
        
    Returns:
        Configuration dictionary.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    
    try:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}")
    
    return {}


def _get_nested(config: Dict, *keys: str, default: Any = None) -> Any:
    """Get nested config value with default fallback."""
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value


# ============================================================
# Face Processing Constants
# ============================================================

@dataclass
class FaceProcessingConfig:
    """Face processing constants."""
    # Margin around detected face as fraction of size
    crop_margin_ratio: float = 0.2
    # Output size for face alignment
    alignment_output_size: Tuple[int, int] = (112, 112)
    # Target size for preprocessing before embedding
    preprocess_target_size: Tuple[int, int] = (160, 160)
    # Normalization values
    normalization_mean: float = 0.5
    normalization_std: float = 0.5
    # Pixel max value (standard, not configurable)
    pixel_max_value: float = 255.0
    
    # Quality score calculation
    sharpness_divisor: float = 500.0
    brightness_target: float = 0.5
    brightness_scale: float = 2.0
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FaceProcessingConfig":
        """Create from config dictionary."""
        fp = _get_nested(config, "face_processing") or {}
        quality = fp.get("quality", {})
        
        alignment_size = fp.get("alignment_output_size", [112, 112])
        preprocess_size = fp.get("preprocess_target_size", [160, 160])
        
        return cls(
            crop_margin_ratio=fp.get("crop_margin_ratio", 0.2),
            alignment_output_size=tuple(alignment_size),
            preprocess_target_size=tuple(preprocess_size),
            normalization_mean=fp.get("normalization_mean", 0.5),
            normalization_std=fp.get("normalization_std", 0.5),
            sharpness_divisor=quality.get("sharpness_divisor", 500.0),
            brightness_target=quality.get("brightness_target", 0.5),
            brightness_scale=quality.get("brightness_scale", 2.0),
        )


# ============================================================
# DNN Detection Constants
# ============================================================

@dataclass
class DNNDetectionConfig:
    """DNN face detection constants."""
    # Input size for SSD detector
    input_size: Tuple[int, int] = (300, 300)
    # Mean values for blob normalization (BGR)
    mean_values: Tuple[float, float, float] = (104.0, 177.0, 123.0)
    # Confidence threshold
    confidence_threshold: float = 0.7
    # NMS threshold
    nms_threshold: float = 0.15
    # Minimum face size
    min_face_size: Tuple[int, int] = (30, 30)
    # Aspect ratio constraints
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    # Maximum face size as fraction of image
    max_size_ratio: float = 0.8
    # Overlap threshold for removing duplicates
    overlap_threshold: float = 0.6
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DNNDetectionConfig":
        """Create from config dictionary."""
        dnn = _get_nested(config, "dnn_detection") or {}
        
        input_size = dnn.get("input_size", [300, 300])
        mean_values = dnn.get("mean_values", [104.0, 177.0, 123.0])
        min_face_size = dnn.get("min_face_size", [30, 30])
        
        return cls(
            input_size=tuple(input_size),
            mean_values=tuple(mean_values),
            confidence_threshold=dnn.get("confidence_threshold", 0.7),
            nms_threshold=dnn.get("nms_threshold", 0.15),
            min_face_size=tuple(min_face_size),
            min_aspect_ratio=dnn.get("min_aspect_ratio", 0.5),
            max_aspect_ratio=dnn.get("max_aspect_ratio", 2.0),
            max_size_ratio=dnn.get("max_size_ratio", 0.8),
            overlap_threshold=dnn.get("overlap_threshold", 0.6),
        )


# ============================================================
# OpenFace Embedding Constants
# ============================================================

@dataclass
class OpenFaceConfig:
    """OpenFace embedding constants."""
    # Input size for OpenFace network
    input_size: Tuple[int, int] = (96, 96)
    # Ratio for threat profile threshold
    threat_profile_threshold_ratio: float = 0.7
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OpenFaceConfig":
        """Create from config dictionary."""
        of = _get_nested(config, "openface") or {}
        input_size = of.get("input_size", [96, 96])
        
        return cls(
            input_size=tuple(input_size),
            threat_profile_threshold_ratio=of.get("threat_profile_threshold_ratio", 0.7),
        )


# ============================================================
# Audio Processing Constants
# ============================================================

@dataclass
class AudioProcessingConfig:
    """Audio processing constants."""
    # Default sample rate
    sample_rate: int = 22050
    # Default duration in seconds
    duration: float = 5.0
    # Silence detection threshold
    silence_threshold: float = 0.01
    # Top dB for trimming
    trim_top_db: int = 20
    
    # Feature extraction
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    power: float = 2.0
    log_epsilon: float = 1e-10
    
    # Dummy classifier thresholds
    dummy_loud_threshold: float = 0.5
    dummy_ambient_threshold: float = 0.1
    dummy_default_confidence: float = 0.5
    
    # Standard numeric constants (not configurable)
    int16_max: float = 32768.0
    int32_max: float = 2147483648.0
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AudioProcessingConfig":
        """Create from config dictionary."""
        ap = _get_nested(config, "audio_processing") or {}
        features = ap.get("features", {})
        dummy = ap.get("dummy_classifier", {})
        
        return cls(
            sample_rate=ap.get("sample_rate", 22050),
            duration=ap.get("duration", 5.0),
            silence_threshold=ap.get("silence_threshold", 0.01),
            trim_top_db=ap.get("trim_top_db", 20),
            n_mfcc=features.get("n_mfcc", 40),
            n_mels=features.get("n_mels", 128),
            n_fft=features.get("n_fft", 2048),
            hop_length=features.get("hop_length", 512),
            power=features.get("power", 2.0),
            log_epsilon=features.get("log_epsilon", 1e-10),
            dummy_loud_threshold=dummy.get("loud_threshold", 0.5),
            dummy_ambient_threshold=dummy.get("ambient_threshold", 0.1),
            dummy_default_confidence=dummy.get("default_confidence", 0.5),
        )


# ============================================================
# Global Config Instance (lazy loaded)
# ============================================================

class Config:
    """Global configuration singleton."""
    
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self) -> None:
        """Load configuration from file."""
        self._config = load_config()
        self._face_processing: Optional[FaceProcessingConfig] = None
        self._dnn_detection: Optional[DNNDetectionConfig] = None
        self._openface: Optional[OpenFaceConfig] = None
        self._audio_processing: Optional[AudioProcessingConfig] = None
    
    def reload(self, config_path: Optional[Path] = None) -> None:
        """Reload configuration from file."""
        self._config = load_config(config_path)
        # Reset cached configs
        self._face_processing = None
        self._dnn_detection = None
        self._openface = None
        self._audio_processing = None
    
    @property
    def face_processing(self) -> FaceProcessingConfig:
        """Get face processing config."""
        if self._face_processing is None:
            self._face_processing = FaceProcessingConfig.from_config(self._config)
        return self._face_processing
    
    @property
    def dnn_detection(self) -> DNNDetectionConfig:
        """Get DNN detection config."""
        if self._dnn_detection is None:
            self._dnn_detection = DNNDetectionConfig.from_config(self._config)
        return self._dnn_detection
    
    @property
    def openface(self) -> OpenFaceConfig:
        """Get OpenFace config."""
        if self._openface is None:
            self._openface = OpenFaceConfig.from_config(self._config)
        return self._openface
    
    @property
    def audio_processing(self) -> AudioProcessingConfig:
        """Get audio processing config."""
        if self._audio_processing is None:
            self._audio_processing = AudioProcessingConfig.from_config(self._config)
        return self._audio_processing
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a config value by key path."""
        return _get_nested(self._config, *keys, default=default)


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()


# Convenience accessors
def get_face_processing_config() -> FaceProcessingConfig:
    """Get face processing configuration."""
    return get_config().face_processing


def get_dnn_detection_config() -> DNNDetectionConfig:
    """Get DNN detection configuration."""
    return get_config().dnn_detection


def get_openface_config() -> OpenFaceConfig:
    """Get OpenFace configuration."""
    return get_config().openface


def get_audio_processing_config() -> AudioProcessingConfig:
    """Get audio processing configuration."""
    return get_config().audio_processing
