"""Tests for sound classification module."""

import pytest
import numpy as np


class TestSoundClassifier:
    """Test cases for SoundClassifier class."""
    
    def test_classifier_initialization(self):
        """Test classifier initializes correctly."""
        from src.audio import SoundClassifier
        
        classifier = SoundClassifier(
            sample_rate=22050,
            confidence_threshold=0.7,
        )
        
        assert classifier.sample_rate == 22050
        assert classifier.confidence_threshold == 0.7
    
    def test_classify_returns_result(self):
        """Test classification returns a result."""
        from src.audio import SoundClassifier
        
        classifier = SoundClassifier()
        
        # Create dummy audio
        audio = np.random.randn(22050 * 5).astype(np.float32)
        
        result = classifier.classify(audio)
        
        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "all_scores")
    
    def test_classification_result_security_event(self):
        """Test ClassificationResult security event detection."""
        from src.audio import ClassificationResult
        
        # Security event
        result1 = ClassificationResult(
            label="glass_breaking",
            confidence=0.9,
            all_scores={"glass_breaking": 0.9},
        )
        assert result1.is_security_event is True
        
        # Non-security event
        result2 = ClassificationResult(
            label="music",
            confidence=0.9,
            all_scores={"music": 0.9},
        )
        assert result2.is_security_event is False


class TestAudioPreprocessor:
    """Test cases for AudioPreprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        from src.audio import AudioPreprocessor
        
        preprocessor = AudioPreprocessor(
            sample_rate=22050,
            duration=5.0,
        )
        
        assert preprocessor.sample_rate == 22050
        assert preprocessor.duration == 5.0
        assert preprocessor.target_length == 22050 * 5
    
    def test_pad_or_truncate_short_audio(self):
        """Test padding short audio."""
        from src.audio import AudioPreprocessor
        
        preprocessor = AudioPreprocessor(sample_rate=1000, duration=1.0)
        
        # Short audio
        audio = np.ones(500)
        result = preprocessor.pad_or_truncate(audio)
        
        assert len(result) == 1000
        assert result[500:].sum() == 0  # Padded with zeros
    
    def test_pad_or_truncate_long_audio(self):
        """Test truncating long audio."""
        from src.audio import AudioPreprocessor
        
        preprocessor = AudioPreprocessor(sample_rate=1000, duration=1.0)
        
        # Long audio
        audio = np.ones(1500)
        result = preprocessor.pad_or_truncate(audio)
        
        assert len(result) == 1000
    
    def test_normalize(self):
        """Test audio normalization."""
        from src.audio import AudioPreprocessor
        
        preprocessor = AudioPreprocessor()
        
        audio = np.array([0.5, -0.5, 0.25, -0.25])
        result = preprocessor.normalize(audio)
        
        assert np.abs(result).max() == 1.0


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    def test_extractor_initialization(self):
        """Test feature extractor initializes correctly."""
        from src.audio import FeatureExtractor
        
        extractor = FeatureExtractor(
            sample_rate=22050,
            n_mfcc=40,
            n_mels=128,
        )
        
        assert extractor.sample_rate == 22050
        assert extractor.n_mfcc == 40
        assert extractor.n_mels == 128
    
    def test_extract_mfcc_shape(self):
        """Test MFCC extraction returns correct shape."""
        from src.audio import FeatureExtractor
        
        extractor = FeatureExtractor(n_mfcc=40)
        
        audio = np.random.randn(22050)  # 1 second
        mfcc = extractor.extract_mfcc(audio)
        
        assert mfcc.shape[0] == 40  # n_mfcc features
    
    def test_extract_all_returns_audio_features(self):
        """Test extract_all returns AudioFeatures object."""
        from src.audio import FeatureExtractor
        from src.audio import AudioFeatures
        
        extractor = FeatureExtractor()
        
        audio = np.random.randn(22050)
        features = extractor.extract_all(audio)
        
        assert isinstance(features, AudioFeatures)
        assert features.mfcc is not None
        assert features.mel_spectrogram is not None
