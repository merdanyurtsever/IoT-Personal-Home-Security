"""Face recognition using embeddings."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class FaceCategory(Enum):
    """Category of recognized face."""
    WATCH_LIST = "watch_list"        # Known threat - watch list match
    THREAT_PROFILE = "threat_profile"  # Matches threat attributes
    UNKNOWN = "unknown"               # No match - safe


@dataclass
class RecognitionResult:
    """Result of face recognition."""
    identity: Optional[str] = None
    confidence: float = 0.0
    category: FaceCategory = FaceCategory.UNKNOWN
    attributes: Dict = field(default_factory=dict)
    
    @property
    def should_alert(self) -> bool:
        """Whether this result should trigger an alert."""
        return self.category in (FaceCategory.WATCH_LIST, FaceCategory.THREAT_PROFILE)


class FaceRecognizer:
    """Face recognizer using OpenCV DNN embeddings."""
    
    # OpenFace model for embeddings
    OPENFACE_URL = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"
    
    def __init__(
        self,
        embedding_backend: str = "opencv_dnn",
        detection_backend: str = "opencv_dnn",
        similarity_threshold: float = 0.6
    ):
        """Initialize the face recognizer.
        
        Args:
            embedding_backend: Backend for face embeddings
            detection_backend: Backend for face detection
            similarity_threshold: Threshold for face matching (lower = stricter)
        """
        self.embedding_backend = embedding_backend
        self.detection_backend = detection_backend
        self.similarity_threshold = similarity_threshold
        
        # Registered identities: name -> list of embeddings
        self._identities: Dict[str, List[np.ndarray]] = {}
        
        # Initialize embedding network
        self._net = None
        self._init_embedding_network()
    
    def _get_model_dir(self) -> Path:
        """Get or create model directory."""
        model_dir = Path(__file__).parent.parent.parent / "models" / "face_recognition"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file from URL."""
        import urllib.request
        print(f"Downloading {dest.name}...")
        urllib.request.urlretrieve(url, str(dest))
    
    def _init_embedding_network(self) -> None:
        """Initialize the face embedding network."""
        model_dir = self._get_model_dir()
        model_path = model_dir / "nn4.small2.v1.t7"
        
        if not model_path.exists():
            self._download_file(self.OPENFACE_URL, model_path)
        
        self._net = cv2.dnn.readNetFromTorch(str(model_path))
    
    def _get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image.
        
        Args:
            face_img: BGR face image
            
        Returns:
            128-dimensional embedding vector or None
        """
        if face_img is None or face_img.size == 0:
            return None
        
        try:
            # Preprocess: resize to 96x96 and normalize
            face_blob = cv2.dnn.blobFromImage(
                face_img,
                1.0 / 255,
                (96, 96),
                (0, 0, 0),
                swapRB=True,
                crop=False
            )
            
            self._net.setInput(face_blob)
            embedding = self._net.forward()
            
            # Normalize embedding
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2))
    
    def register_face(self, name: str, face_img: np.ndarray) -> bool:
        """Register a face for an identity.
        
        Args:
            name: Identity name
            face_img: BGR face image
            
        Returns:
            True if registration successful
        """
        embedding = self._get_embedding(face_img)
        
        if embedding is None:
            return False
        
        if name not in self._identities:
            self._identities[name] = []
        
        self._identities[name].append(embedding)
        return True
    
    def register_from_directory(self, directory: str) -> Dict[str, int]:
        """Register faces from a directory structure.
        
        Expected structure:
            directory/
                person1/
                    img1.jpg
                    img2.jpg
                person2/
                    img1.jpg
        
        Args:
            directory: Path to directory
            
        Returns:
            Dict mapping name to number of enrolled images
        """
        results = {}
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return results
        
        for person_dir in dir_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            name = person_dir.name
            count = 0
            
            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    img = cv2.imread(str(img_path))
                    if img is not None and self.register_face(name, img):
                        count += 1
            
            if count > 0:
                results[name] = count
        
        return results
    
    def recognize_face(
        self,
        face_img: np.ndarray,
        detect_attributes: bool = False
    ) -> RecognitionResult:
        """Recognize a face.
        
        Args:
            face_img: BGR face image
            detect_attributes: Whether to detect face attributes
            
        Returns:
            RecognitionResult with identity and category
        """
        embedding = self._get_embedding(face_img)
        
        if embedding is None:
            return RecognitionResult(
                identity=None,
                confidence=0.0,
                category=FaceCategory.UNKNOWN
            )
        
        best_match = None
        best_similarity = -1.0
        
        # Compare against all registered identities
        for name, embeddings in self._identities.items():
            for ref_embedding in embeddings:
                similarity = self._compute_similarity(embedding, ref_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        # Determine category based on similarity
        if best_similarity >= self.similarity_threshold:
            return RecognitionResult(
                identity=best_match,
                confidence=best_similarity,
                category=FaceCategory.WATCH_LIST
            )
        elif best_similarity >= self.similarity_threshold * 0.7:
            # Partial match - could be threat profile
            return RecognitionResult(
                identity=best_match,
                confidence=best_similarity,
                category=FaceCategory.THREAT_PROFILE
            )
        else:
            return RecognitionResult(
                identity="unknown",
                confidence=1.0 - best_similarity if best_similarity > 0 else 1.0,
                category=FaceCategory.UNKNOWN
            )
    
    def get_registered_identities(self) -> List[str]:
        """Get list of registered identity names."""
        return list(self._identities.keys())
    
    def clear_identities(self) -> None:
        """Clear all registered identities."""
        self._identities.clear()
