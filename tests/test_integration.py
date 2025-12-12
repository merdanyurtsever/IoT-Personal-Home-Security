"""Integration tests for the security system."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestSecuritySystemIntegration:
    """Integration tests for the complete security system."""
    
    def test_audio_classification_pipeline(self):
        """Test complete audio classification pipeline."""
        from src.audio import (
            AudioPreprocessor,
            FeatureExtractor, 
            SoundClassifier,
        )
        
        # Initialize components
        preprocessor = AudioPreprocessor(sample_rate=22050, duration=5.0)
        extractor = FeatureExtractor(sample_rate=22050)
        classifier = SoundClassifier(sample_rate=22050)
        
        # Create test audio
        audio = np.random.randn(22050 * 5).astype(np.float32)
        
        # Preprocess
        processed = preprocessor.preprocess(audio)
        
        # Extract features
        features = extractor.extract_mel_spectrogram(processed)
        
        # Classify
        result = classifier.classify(processed)
        
        assert result.label in classifier.classes
        assert 0 <= result.confidence <= 1
    
    def test_alert_system_integration(self):
        """Test alert system components work together."""
        from src.alerts import NotificationManager, LocalAlarm, SecurityEvent
        from datetime import datetime
        
        # Initialize components
        notification_manager = NotificationManager()
        local_alarm = LocalAlarm(use_mock=True)
        
        # Create security event
        event = SecurityEvent(
            event_type="glass_breaking",
            timestamp=datetime.now(),
            confidence=0.95,
            details={"location": "living_room"},
        )
        
        # Trigger local alarm
        local_alarm.trigger(duration=1.0)
        assert local_alarm.is_active()
        
        # Stop alarm
        local_alarm.stop()
        assert not local_alarm.is_active()
        
        # Send notification (no notifiers configured, so results empty)
        results = notification_manager.notify(event)
        assert isinstance(results, dict)
    
    def test_sensor_interfaces(self):
        """Test sensor interface initialization."""
        pytest.skip("MotionSensor not yet implemented in new sensors module")


class TestConfigurationLoading:
    """Test configuration loading and validation."""
    
    def test_config_yaml_structure(self):
        """Test that config.yaml has expected structure."""
        import yaml
        from pathlib import Path
        
        config_path = Path("config/config.yaml")
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            assert "face_detection" in config
            assert "sound_classification" in config
            assert "camera" in config
            assert "alerts" in config


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_image_detection(self):
        """Test detection on empty/black image."""
        from src import FaceDetector
        
        detector = FaceDetector()
        
        # All black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect(image)
        
        assert isinstance(faces, list)
        assert len(faces) == 0  # No faces in black image
    
    def test_very_short_audio(self):
        """Test classification with very short audio."""
        from src.audio import AudioPreprocessor
        
        preprocessor = AudioPreprocessor(sample_rate=22050, duration=5.0)
        
        # Very short audio (0.1 seconds)
        audio = np.random.randn(2205)
        processed = preprocessor.preprocess(audio)
        
        # Should be padded to correct length
        assert len(processed) == 22050 * 5
    
    def test_silent_audio(self):
        """Test classification with silent audio."""
        from src.audio import SoundClassifier
        
        classifier = SoundClassifier()
        
        # Silent audio
        audio = np.zeros(22050 * 5)
        result = classifier.classify(audio)
        
        # Should still return a result
        assert result.label is not None
