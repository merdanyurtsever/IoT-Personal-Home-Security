"""Audio processing module.

Contains:
- AudioPreprocessor: Audio preprocessing (resampling, filtering)
- FeatureExtractor: Audio feature extraction (MFCC, Mel spectrogram)
- SoundClassifier: Sound classification model
"""

from .preprocessing import AudioPreprocessor, AudioConfig
from .features import FeatureExtractor, FeatureConfig
from .classifier import SoundClassifier, ClassificationResult

__all__ = [
    "AudioPreprocessor",
    "AudioConfig",
    "FeatureExtractor",
    "FeatureConfig",
    "SoundClassifier",
    "ClassificationResult",
]
