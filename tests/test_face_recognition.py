"""Tests for ArcFace face recognition module."""

import pytest
import numpy as np


class TestArcFaceRecognizer:
    """Test cases for ArcFaceRecognizer class."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.face import ArcFaceRecognizer, FaceDatabase
        assert ArcFaceRecognizer is not None
        assert FaceDatabase is not None
    
    def test_face_database(self):
        """Test FaceDatabase operations."""
        from src.face import FaceDatabase
        
        db = FaceDatabase()
        
        # Test empty database
        assert len(db) == 0
        
        # Add a face
        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)  # Normalize
        db.add("test_person", embedding)
        
        assert len(db) == 1
        
        # Find match
        name, score = db.find_match(embedding, threshold=0.5)
        assert name == "test_person"
        assert score > 0.9  # Should be very similar
        
        # Clear
        db.clear()
        assert len(db) == 0
    
    def test_cosine_similarity(self):
        """Test cosine similarity function."""
        from src.face import cosine_similarity
        
        # Identical vectors
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, b) - 1.0) < 0.001
        
        # Orthogonal vectors
        c = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, c)) < 0.001
        
        # Opposite vectors
        d = np.array([-1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, d) - (-1.0)) < 0.001
    
    def test_enhance_brightness(self):
        """Test brightness enhancement."""
        from src.face import enhance_brightness
        
        # Create dark image
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8)
        dark_image[:] = 30  # Very dark gray
        
        enhanced = enhance_brightness(dark_image)
        
        # Enhanced should be brighter (higher mean)
        assert enhanced.mean() >= dark_image.mean()
    
    def test_expand_bbox(self):
        """Test bounding box expansion."""
        from src.face import expand_bbox
        
        # Test expansion within bounds (frame_shape is a tuple)
        frame_shape = (200, 200, 3)
        x, y, w, h = expand_bbox(100, 100, 50, 50, frame_shape, margin=0.5)
        
        # Should be larger
        assert w > 50
        assert h > 50
        
        # Should still be within image bounds
        assert x >= 0
        assert y >= 0
        assert x + w <= 200
        assert y + h <= 200
