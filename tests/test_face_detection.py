"""Tests for face detection module."""

import pytest
import numpy as np


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        from src.iot_home_security.face import FaceDetector
        
        detector = FaceDetector(backend="haar_cascade")
        assert detector.backend_name == "haar_cascade"
    
    def test_detector_invalid_backend(self):
        """Test detector raises error for invalid backend."""
        from src.iot_home_security.face import FaceDetector
        
        with pytest.raises(ValueError):
            FaceDetector(backend="invalid_backend")
    
    def test_detect_returns_list(self):
        """Test detection returns a list."""
        from src.iot_home_security.face import FaceDetector
        
        detector = FaceDetector()
        
        # Create dummy image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        faces = detector.detect(image)
        assert isinstance(faces, list)
    
    def test_detected_face_properties(self):
        """Test DetectedFace dataclass properties."""
        from src.iot_home_security.face.detector import DetectedFace
        
        face = DetectedFace(x=100, y=50, width=80, height=100, confidence=0.95)
        
        assert face.bbox == (100, 50, 80, 100)
        assert face.center == (140, 100)
        assert face.confidence == 0.95


class TestFaceRecognizer:
    """Test cases for FaceRecognizer class."""
    
    def test_recognizer_initialization(self):
        """Test recognizer initializes correctly."""
        from src.iot_home_security.face import FaceRecognizer
        
        recognizer = FaceRecognizer(model="facenet", threshold=0.6)
        assert recognizer.model_name == "facenet"
        assert recognizer.threshold == 0.6
    
    def test_recognize_returns_result(self):
        """Test recognition returns a result."""
        from src.iot_home_security.face import FaceRecognizer
        
        recognizer = FaceRecognizer()
        
        # Create dummy face image
        face_image = np.zeros((160, 160, 3), dtype=np.uint8)
        
        result = recognizer.recognize(face_image)
        assert hasattr(result, "identity")
        assert hasattr(result, "confidence")
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from src.iot_home_security.face import FaceRecognizer
        
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([0, 1, 0])
        
        # Same vectors should have similarity of 1
        assert abs(FaceRecognizer._cosine_similarity(a, b) - 1.0) < 1e-6
        
        # Orthogonal vectors should have similarity of 0
        assert abs(FaceRecognizer._cosine_similarity(a, c)) < 1e-6
