"""Main face recognizer combining detection, embeddings, and recognition."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..detection import FaceDetector, DetectedFace
from ..utils import crop_face, preprocess_face
from .types import FaceCategory, RecognitionResult
from .attributes import AttributeProfile, AttributeFilterChain, HaarAttributeDetector
from .embeddings import (
    BaseEmbeddingBackend,
    DlibEmbeddingBackend,
    TFLiteEmbeddingBackend,
    MobileNetV2EmbeddingBackend,
    OpenCVDNNEmbeddingBackend,
    EMBEDDING_BACKENDS,
)

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Main face recognition class combining detection, embedding, and matching."""
    
    def __init__(
        self,
        embedding_backend: str = "opencv_dnn",
        detection_backend: str = "opencv_dnn",
        similarity_threshold: float = 0.6,
        attribute_filters: Optional[AttributeFilterChain] = None,
    ):
        """Initialize face recognizer.
        
        Args:
            embedding_backend: Which embedding backend to use
                              ("opencv_dnn", "dlib", "tflite", "mobilenetv2")
            detection_backend: Which detection backend to use
            similarity_threshold: Minimum similarity for a match (0.0 to 1.0)
            attribute_filters: Optional chain of attribute filters
        """
        self.similarity_threshold = similarity_threshold
        self.attribute_filters = attribute_filters
        
        self._detector = FaceDetector(backend=detection_backend)
        self._embedder = self._create_embedding_backend(embedding_backend)
        self._attribute_detector = HaarAttributeDetector()
        
        self._face_database: Dict[str, List[np.ndarray]] = {}
        self._face_id_counter = 0
    
    def _create_embedding_backend(self, backend: str) -> BaseEmbeddingBackend:
        """Create embedding backend by name."""
        backend = backend.lower()
        
        if backend == "opencv_dnn":
            return OpenCVDNNEmbeddingBackend()
        elif backend == "dlib":
            return DlibEmbeddingBackend()
        elif backend == "tflite":
            return TFLiteEmbeddingBackend()
        elif backend == "mobilenetv2":
            return MobileNetV2EmbeddingBackend()
        else:
            logger.warning(f"Unknown backend '{backend}', using opencv_dnn")
            return OpenCVDNNEmbeddingBackend()
    
    def register_face(
        self,
        name: str,
        face_image: np.ndarray,
        detect: bool = True,
    ) -> bool:
        """Register a face in the database.
        
        Args:
            name: Identity name for this face
            face_image: Image containing the face (full image or cropped)
            detect: If True, detect face first; if False, assume already cropped
            
        Returns:
            True if registration succeeded
        """
        if detect:
            faces = self._detector.detect(face_image)
            if not faces:
                logger.warning(f"No face detected for {name}")
                return False
            
            face = faces[0]
            x, y, w, h = face.bbox
            face_crop = face_image[y:y+h, x:x+w]
        else:
            face_crop = face_image
        
        face_crop = preprocess_face(face_crop)
        
        embedding = self._embedder.extract(face_crop)
        if embedding is None:
            logger.warning(f"Could not extract embedding for {name}")
            return False
        
        if name not in self._face_database:
            self._face_database[name] = []
        
        self._face_database[name].append(embedding)
        logger.info(f"Registered face for {name} ({len(self._face_database[name])} samples)")
        return True
    
    def register_from_directory(self, directory: str) -> Dict[str, int]:
        """Register faces from a directory structure.
        
        Expected structure:
            directory/
                person1/
                    image1.jpg
                    image2.jpg
                person2/
                    image1.jpg
                    
        Args:
            directory: Path to face database directory
            
        Returns:
            Dict mapping name to number of registered faces
        """
        import cv2
        
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return {}
        
        results = {}
        
        for person_dir in dir_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            name = person_dir.name
            count = 0
            
            for image_file in person_dir.iterdir():
                if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                if self.register_face(name, image, detect=True):
                    count += 1
            
            if count > 0:
                results[name] = count
                logger.info(f"Registered {count} faces for {name}")
        
        return results
    
    def recognize(
        self,
        image: np.ndarray,
        detect_attributes: bool = False,
    ) -> List[RecognitionResult]:
        """Recognize faces in an image.
        
        Args:
            image: Input image (BGR)
            detect_attributes: Whether to detect face attributes
            
        Returns:
            List of recognition results for each detected face
        """
        faces = self._detector.detect(image)
        results = []
        
        for face in faces:
            result = self._recognize_face(image, face, detect_attributes)
            results.append(result)
        
        return results
    
    def recognize_face(
        self,
        face_image: np.ndarray,
        detect_attributes: bool = False,
    ) -> RecognitionResult:
        """Recognize a single cropped face image.
        
        Args:
            face_image: Cropped face image
            detect_attributes: Whether to detect face attributes
            
        Returns:
            Recognition result
        """
        self._face_id_counter += 1
        
        face_image = preprocess_face(face_image)
        
        embedding = self._embedder.extract(face_image)
        if embedding is None:
            return RecognitionResult(
                identity="unknown",
                confidence=0.0,
                category=FaceCategory.NO_MATCH,
                face_id=self._face_id_counter,
            )
        
        identity, similarity = self._find_best_match(embedding)
        
        attribute_profile = None
        if detect_attributes:
            attribute_profile = self._attribute_detector.detect(face_image)
        
        category = FaceCategory.NO_MATCH
        matched_filter = None
        
        if identity and similarity >= self.similarity_threshold:
            category = FaceCategory.WATCH_LIST
        elif attribute_profile and self.attribute_filters:
            match = self.attribute_filters.best_match(attribute_profile)
            if match:
                category = FaceCategory.THREAT_PROFILE
                matched_filter = match[0]
        
        return RecognitionResult(
            identity=identity or "unknown",
            confidence=similarity,
            category=category,
            embedding=embedding,
            face_id=self._face_id_counter,
            attribute_profile=attribute_profile,
            matched_filter=matched_filter,
        )
    
    def _recognize_face(
        self,
        image: np.ndarray,
        face: DetectedFace,
        detect_attributes: bool,
    ) -> RecognitionResult:
        """Internal method to recognize a detected face."""
        self._face_id_counter += 1
        
        x, y, w, h = face.bbox
        face_crop = image[y:y+h, x:x+w]
        face_crop = preprocess_face(face_crop)
        
        embedding = self._embedder.extract(face_crop)
        if embedding is None:
            return RecognitionResult(
                identity="unknown",
                confidence=0.0,
                category=FaceCategory.NO_MATCH,
                face_id=self._face_id_counter,
            )
        
        identity, similarity = self._find_best_match(embedding)
        
        attribute_profile = None
        if detect_attributes:
            attribute_profile = self._attribute_detector.detect(face_crop)
        
        category = FaceCategory.NO_MATCH
        matched_filter = None
        
        if identity and similarity >= self.similarity_threshold:
            category = FaceCategory.WATCH_LIST
        elif attribute_profile and self.attribute_filters:
            match = self.attribute_filters.best_match(attribute_profile)
            if match:
                category = FaceCategory.THREAT_PROFILE
                matched_filter = match[0]
        
        return RecognitionResult(
            identity=identity or "unknown",
            confidence=similarity,
            category=category,
            embedding=embedding,
            face_id=self._face_id_counter,
            attribute_profile=attribute_profile,
            matched_filter=matched_filter,
        )
    
    def _find_best_match(
        self,
        embedding: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """Find the best matching identity for an embedding.
        
        Args:
            embedding: Face embedding to match
            
        Returns:
            Tuple of (identity_name, similarity_score) or (None, 0.0)
        """
        best_identity = None
        best_similarity = 0.0
        
        for name, embeddings in self._face_database.items():
            for stored_embedding in embeddings:
                similarity = self._embedder.compare(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_identity = name
        
        return best_identity, best_similarity
    
    def get_registered_identities(self) -> List[str]:
        """Get list of registered identity names."""
        return list(self._face_database.keys())
    
    def get_sample_count(self, name: str) -> int:
        """Get number of registered samples for an identity."""
        return len(self._face_database.get(name, []))
    
    def remove_identity(self, name: str) -> bool:
        """Remove an identity from the database."""
        if name in self._face_database:
            del self._face_database[name]
            return True
        return False
    
    def clear_database(self):
        """Clear all registered faces."""
        self._face_database.clear()
        self._face_id_counter = 0
    
    @property
    def embedding_backend(self) -> str:
        """Get current embedding backend name."""
        return self._embedder.name
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self._embedder.embedding_dim
