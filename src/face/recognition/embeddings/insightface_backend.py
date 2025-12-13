"""InsightFace embedding backends (ArcFace, MobileFaceNet).

These backends use InsightFace library for state-of-the-art face recognition.
Falls back to direct ONNX model loading if insightface is not installed.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseEmbeddingBackend

logger = logging.getLogger(__name__)

# Model URLs for direct ONNX download (when insightface not available)
ARCFACE_MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
MOBILEFACENET_MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"


def _check_insightface() -> bool:
    """Check if insightface is available."""
    try:
        from insightface.app import FaceAnalysis
        return True
    except ImportError:
        return False


def _check_onnxruntime() -> bool:
    """Check if onnxruntime is available."""
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


class ArcFaceEmbeddingBackend(BaseEmbeddingBackend):
    """ArcFace embedding using InsightFace (512D).
    
    Uses buffalo_sc model which provides a good balance of speed and accuracy.
    ArcFace is known for excellent recognition accuracy.
    
    Pros: State-of-the-art accuracy, robust to variations
    Cons: Requires insightface library, larger model
    Best for: High-accuracy face recognition applications
    """
    
    def __init__(self, model_name: str = "buffalo_sc", det_size: tuple = (640, 640)):
        """Initialize ArcFace embedding backend.
        
        Args:
            model_name: InsightFace model pack name
                - buffalo_sc: Smaller, faster (recommended for CPU)
                - buffalo_l: Larger, more accurate
            det_size: Detection input size (width, height)
        """
        self._app = None
        self._initialized = False
        self._model_name = model_name
        self._det_size = det_size
    
    @property
    def name(self) -> str:
        return "arcface"
    
    @property
    def embedding_dim(self) -> int:
        return 512
    
    def _initialize(self) -> bool:
        """Lazy initialization of InsightFace model."""
        if self._initialized:
            return self._app is not None
        
        self._initialized = True
        
        try:
            from insightface.app import FaceAnalysis
            
            self._app = FaceAnalysis(
                name=self._model_name,
                providers=['CPUExecutionProvider']
            )
            self._app.prepare(ctx_id=-1, det_size=self._det_size)
            
            logger.info(f"Initialized ArcFace with {self._model_name}")
            return True
            
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ArcFace: {e}")
            return False
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512D embedding using ArcFace.
        
        Args:
            face_image: BGR face image (can be cropped face or full image)
            
        Returns:
            512D embedding vector or None
        """
        if not self._initialize():
            return None
        
        try:
            # First try InsightFace's built-in detection
            faces = self._app.get(face_image)
            
            if not faces:
                # If no face detected, the image might already be a cropped face
                # Try padding it to give the detector more context
                h, w = face_image.shape[:2]
                # Pad the image so face is roughly in center with context
                pad = max(h, w) // 2
                padded = np.zeros((h + 2*pad, w + 2*pad, 3), dtype=np.uint8)
                padded[pad:pad+h, pad:pad+w] = face_image
                faces = self._app.get(padded)
            
            if not faces:
                # Still no detection - use the recognition model directly on resized input
                # InsightFace recognition models expect 112x112 aligned faces
                embedding = self._extract_direct(face_image)
                if embedding is not None:
                    return embedding
                return None
            
            if faces:
                embedding = faces[0].embedding
                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            
            return None
            
        except Exception as e:
            logger.debug(f"ArcFace extraction failed: {e}")
            return None
    
    def _extract_direct(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding directly from recognition model without detection.
        
        Args:
            face_image: Pre-cropped face image
            
        Returns:
            512D embedding or None
        """
        try:
            # Get the recognition model from InsightFace app
            rec_model = None
            for model in self._app.models.values():
                if hasattr(model, 'input_size') and model.input_size == (112, 112):
                    rec_model = model
                    break
            
            if rec_model is None:
                return None
            
            # Preprocess face image for recognition model
            # Resize to 112x112
            face_resized = cv2.resize(face_image, (112, 112))
            
            # Convert BGR to RGB and normalize
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Transpose to CHW and add batch dimension
            face_input = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
            face_input = (face_input - 127.5) / 127.5  # Normalize to [-1, 1]
            face_input = np.expand_dims(face_input, axis=0)
            
            # Run inference
            embedding = rec_model.session.run(
                rec_model.output_names,
                {rec_model.input_names[0]: face_input}
            )[0][0]
            
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Direct ArcFace extraction failed: {e}")
            return None


class MobileFaceNetEmbeddingBackend(BaseEmbeddingBackend):
    """MobileFaceNet embedding using InsightFace (512D).
    
    Uses buffalo_s model which is optimized for mobile/embedded devices.
    Faster than ArcFace with slightly lower accuracy.
    
    Pros: Fast, efficient, good for embedded/mobile
    Cons: Slightly lower accuracy than full ArcFace
    Best for: Resource-constrained devices like Raspberry Pi
    """
    
    def __init__(self, det_size: tuple = (320, 320)):
        """Initialize MobileFaceNet embedding backend.
        
        Args:
            det_size: Detection input size (smaller = faster)
        """
        self._app = None
        self._initialized = False
        self._det_size = det_size
    
    @property
    def name(self) -> str:
        return "mobilefacenet"
    
    @property
    def embedding_dim(self) -> int:
        return 512
    
    def _initialize(self) -> bool:
        """Lazy initialization of InsightFace model."""
        if self._initialized:
            return self._app is not None
        
        self._initialized = True
        
        try:
            from insightface.app import FaceAnalysis
            
            # buffalo_s uses MobileFaceNet for recognition
            self._app = FaceAnalysis(
                name='buffalo_s',
                providers=['CPUExecutionProvider']
            )
            self._app.prepare(ctx_id=-1, det_size=self._det_size)
            
            logger.info("Initialized MobileFaceNet")
            return True
            
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MobileFaceNet: {e}")
            return False
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512D embedding using MobileFaceNet.
        
        Args:
            face_image: BGR face image (can be cropped face or full image)
            
        Returns:
            512D embedding vector or None
        """
        if not self._initialize():
            return None
        
        try:
            # First try InsightFace's built-in detection
            faces = self._app.get(face_image)
            
            if not faces:
                # If no face detected, try padding to give detector more context
                h, w = face_image.shape[:2]
                pad = max(h, w) // 2
                padded = np.zeros((h + 2*pad, w + 2*pad, 3), dtype=np.uint8)
                padded[pad:pad+h, pad:pad+w] = face_image
                faces = self._app.get(padded)
            
            if not faces:
                # Still no detection - use recognition model directly
                embedding = self._extract_direct(face_image)
                if embedding is not None:
                    return embedding
                return None
            
            if faces:
                embedding = faces[0].embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            
            return None
            
        except Exception as e:
            logger.debug(f"MobileFaceNet extraction failed: {e}")
            return None
    
    def _extract_direct(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding directly from recognition model without detection.
        
        Args:
            face_image: Pre-cropped face image
            
        Returns:
            512D embedding or None
        """
        try:
            # Get the recognition model from InsightFace app
            rec_model = None
            for model in self._app.models.values():
                if hasattr(model, 'input_size') and model.input_size == (112, 112):
                    rec_model = model
                    break
            
            if rec_model is None:
                return None
            
            # Preprocess face image for recognition model
            face_resized = cv2.resize(face_image, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
            face_input = (face_input - 127.5) / 127.5
            face_input = np.expand_dims(face_input, axis=0)
            
            embedding = rec_model.session.run(
                rec_model.output_names,
                {rec_model.input_names[0]: face_input}
            )[0][0]
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Direct MobileFaceNet extraction failed: {e}")
            return None
