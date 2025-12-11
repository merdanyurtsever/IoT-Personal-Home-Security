"""Face detection module with utility functions.

This module provides face detection functionality using various backends
including Haar Cascades, MediaPipe, and OpenCV DNN.
Optimized for ARM64/Linux compatibility.
Also includes utility functions for face image processing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2,
) -> np.ndarray:
    """Crop face from image with margin.
    
    Args:
        image: Full image
        bbox: Bounding box (x, y, w, h)
        margin: Margin around face as fraction of size
        
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image.shape[1], x + w + margin_w)
    y2 = min(image.shape[0], y + h + margin_h)
    return image[y1:y2, x1:x2]


def align_face(
    face_image: np.ndarray,
    landmarks: dict[str, tuple[int, int]],
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Align face using eye landmarks.
    
    Args:
        face_image: Cropped face image
        landmarks: Dictionary with 'left_eye' and 'right_eye' coordinates
        output_size: Output image size
        
    Returns:
        Aligned face image
    """
    # TODO: Implement face alignment
    return cv2.resize(face_image, output_size)


def preprocess_face(
    face_image: np.ndarray,
    target_size: Tuple[int, int] = (160, 160),
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess face image for model input.
    
    Args:
        face_image: Face image
        target_size: Target size for resizing
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed face image
    """
    face = cv2.resize(face_image, target_size)
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    if normalize:
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
    return face


def compute_face_quality(face_image: np.ndarray) -> float:
    """Compute face image quality score.
    
    Args:
        face_image: Face image
        
    Returns:
        Quality score between 0 and 1
    """
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 500.0, 1.0)
    brightness = gray.mean() / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2
    return float((sharpness + brightness_score) / 2)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box and confidence."""
    
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    landmarks: Optional[dict[str, tuple[int, int]]] = None
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Return area of bounding box."""
        return self.width * self.height


class BaseFaceDetector(ABC):
    """Abstract base class for face detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        pass


class HaarCascadeDetector(BaseFaceDetector):
    """Face detector using OpenCV Haar Cascades."""
    
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        """Initialize Haar Cascade detector.
        
        Args:
            scale_factor: Scale factor for multi-scale detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size to detect
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load pre-trained cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using Haar Cascade.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        
        # Convert to DetectedFace objects
        detected = []
        for (x, y, w, h) in faces:
            detected.append(DetectedFace(x=x, y=y, width=w, height=h))
        
        return detected


class MediaPipeDetector(BaseFaceDetector):
    """Face detector using Google MediaPipe - fast and accurate for ARM64.
    
    Pros: Fast, provides landmarks, good ARM64 support
    Cons: Requires mediapipe package
    Best for: Real-time detection on edge devices
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,  # 0=short-range (2m), 1=full-range (5m)
    ):
        """Initialize MediaPipe face detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold
            model_selection: 0 for short-range, 1 for full-range detection
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence,
                model_selection=model_selection,
            )
            logger.info("Initialized MediaPipe face detector")
        except ImportError:
            raise ImportError(
                "MediaPipe is required. Install with: pip install mediapipe"
            )
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using MediaPipe."""
        # MediaPipe expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.detector.process(rgb_image)
        
        detected = []
        if results.detections:
            h, w = image.shape[:2]
            
            for detection in results.detections:
                # Get bounding box (relative coordinates)
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to absolute coordinates
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)
                
                # Get confidence
                confidence = detection.score[0] if detection.score else 1.0
                
                # Extract landmarks if available
                landmarks = None
                if detection.location_data.relative_keypoints:
                    landmarks = {}
                    keypoint_names = [
                        'right_eye', 'left_eye', 'nose_tip',
                        'mouth_center', 'right_ear_tragion', 'left_ear_tragion'
                    ]
                    for i, kp in enumerate(detection.location_data.relative_keypoints):
                        if i < len(keypoint_names):
                            landmarks[keypoint_names[i]] = (int(kp.x * w), int(kp.y * h))
                
                detected.append(DetectedFace(
                    x=x, y=y, width=width, height=height,
                    confidence=confidence, landmarks=landmarks
                ))
        
        return detected
    
    def __del__(self):
        """Clean up detector resources."""
        if hasattr(self, 'detector'):
            self.detector.close()


class OpenCVDNNDetector(BaseFaceDetector):
    """Face detector using OpenCV DNN module with pre-trained SSD model.
    
    Pros: Good accuracy, comes with OpenCV, no extra dependencies
    Cons: Needs model files (auto-downloads)
    Best for: Balance of speed and accuracy without extra deps
    """
    
    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        input_size: Tuple[int, int] = (300, 300),
        nms_threshold: float = 0.15,
    ):
        """Initialize OpenCV DNN face detector.
        
        Args:
            confidence_threshold: Minimum confidence for detection (default: 0.7)
            model_path: Path to caffemodel file (optional, will download)
            config_path: Path to prototxt file (optional, will download)
            input_size: Input size for the network
            nms_threshold: Non-maximum suppression threshold to remove overlaps (default: 0.15)
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.nms_threshold = nms_threshold
        
        # Load model
        self.net = self._load_model(model_path, config_path)
        logger.info("Initialized OpenCV DNN face detector")
    
    def _load_model(
        self, 
        model_path: Optional[Union[str, Path]], 
        config_path: Optional[Union[str, Path]]
    ) -> cv2.dnn.Net:
        """Load the DNN model."""
        # Default paths - src/face.py -> parent = src -> parent = project root
        default_model_dir = Path(__file__).parent.parent / "data" / "models" / "face_detection"
        default_model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_path is None:
            model_path = default_model_dir / "opencv_face_detector.caffemodel"
        if config_path is None:
            config_path = default_model_dir / "opencv_face_detector.prototxt"
        
        model_path = Path(model_path)
        config_path = Path(config_path)
        
        # Download if not exists
        if not model_path.exists():
            logger.info(f"Downloading model to {model_path}...")
            self._download_file(self.MODEL_URL, model_path)
        
        if not config_path.exists():
            logger.info(f"Downloading config to {config_path}...")
            self._download_file(self.CONFIG_URL, config_path)
        
        # Load network
        net = cv2.dnn.readNetFromCaffe(str(config_path), str(model_path))
        
        # Use CPU backend (for ARM64 compatibility)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net
    
    def _download_file(self, url: str, path: Path) -> None:
        """Download a file from URL."""
        import urllib.request
        
        try:
            urllib.request.urlretrieve(url, str(path))
            logger.info(f"Downloaded to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {url}: {e}")
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using OpenCV DNN with NMS to remove overlapping detections."""
        h, w = image.shape[:2]
        
        # Create blob from image - use native aspect ratio
        # The model expects 300x300 but we scale coordinates back properly
        blob = cv2.dnn.blobFromImage(
            image, 1.0, self.input_size,
            (104.0, 177.0, 123.0),  # Mean subtraction values
            swapRB=False, crop=False
        )
        
        # Forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Collect valid detections
        boxes = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            
            if confidence > self.confidence_threshold:
                # Get bounding box (model outputs normalized 0-1 coordinates)
                x1_norm = detections[0, 0, i, 3]
                y1_norm = detections[0, 0, i, 4]
                x2_norm = detections[0, 0, i, 5]
                y2_norm = detections[0, 0, i, 6]
                
                # Validate normalized coordinates are in valid range
                if not (0 <= x1_norm <= 1 and 0 <= y1_norm <= 1 and 
                        0 <= x2_norm <= 1 and 0 <= y2_norm <= 1):
                    continue
                
                # Scale to image dimensions
                x1 = int(x1_norm * w)
                y1 = int(y1_norm * h)
                x2 = int(x2_norm * w)
                y2 = int(y2_norm * h)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Sanity checks for valid face dimensions
                # Face should be roughly square-ish (aspect ratio 0.5-2.0)
                if box_w < 30 or box_h < 30:
                    continue
                    
                aspect_ratio = box_w / box_h if box_h > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Face shouldn't be larger than 80% of the image
                if box_w > w * 0.8 or box_h > h * 0.8:
                    continue
                
                boxes.append([x1, y1, box_w, box_h])
                confidences.append(confidence)
        
        # Apply Non-Maximum Suppression
        detected = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            for i in indices:
                idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                x, y, width, height = boxes[idx]
                detected.append(DetectedFace(
                    x=x, y=y, width=width, height=height,
                    confidence=confidences[idx]
                ))
        
        # Final deduplication: if faces overlap significantly, keep only highest confidence
        if len(detected) > 1:
            detected = self._remove_overlapping_faces(detected)
        
        return detected
    
    def _remove_overlapping_faces(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        """Remove overlapping face detections, keeping highest confidence."""
        if not faces:
            return faces
        
        # Sort by confidence (highest first)
        sorted_faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        
        kept = []
        for face in sorted_faces:
            x1, y1, w1, h1 = face.bbox
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            
            # Check overlap with already kept faces
            is_duplicate = False
            for kept_face in kept:
                x2, y2, w2, h2 = kept_face.bbox
                cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
                
                # Calculate center distance
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                avg_size = (w1 + h1 + w2 + h2) / 4
                
                # If centers are within 60% of average face size, consider duplicate
                if dist < avg_size * 0.6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(face)
        
        return kept


class FaceDetector:
    """Main face detector class with configurable backend.
    
    Default backend is opencv_dnn which works on all platforms (x86, ARM64, Pi).
    """
    
    BACKENDS = {
        "haar_cascade": HaarCascadeDetector,
        "mediapipe": MediaPipeDetector,
        "opencv_dnn": OpenCVDNNDetector,
    }
    
    def __init__(self, backend: str = "opencv_dnn", **kwargs):
        """Initialize face detector with specified backend.
        
        Args:
            backend: Detection backend to use (default: opencv_dnn)
            **kwargs: Additional arguments for the detector
        """
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Available: {list(self.BACKENDS.keys())}"
            )
        
        self.backend_name = backend
        self.detector = self.BACKENDS[backend](**kwargs)
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        return self.detector.detect(image)
    
    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
        show_landmarks: bool = True,
    ) -> np.ndarray:
        """Draw detection boxes on image.
        
        Args:
            image: BGR image as numpy array
            faces: List of detected faces
            color: Box color (BGR)
            thickness: Line thickness
            show_confidence: Whether to display confidence score
            show_landmarks: Whether to draw facial landmarks
            
        Returns:
            Image with drawn boxes
        """
        output = image.copy()
        
        for face in faces:
            # Draw bounding box
            cv2.rectangle(
                output,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                color,
                thickness,
            )
            
            # Draw confidence score
            if show_confidence:
                label = f"{face.confidence:.0%}"
                cv2.putText(
                    output, label,
                    (face.x, face.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
            
            # Draw landmarks
            if show_landmarks and face.landmarks:
                for name, (lx, ly) in face.landmarks.items():
                    cv2.circle(output, (lx, ly), 2, (0, 0, 255), -1)
        
        return output
    
    @classmethod
    def available_backends(cls) -> List[str]:
        """Return list of available detection backends."""
        return list(cls.BACKENDS.keys())


# Module-level export for convenience
BACKENDS = FaceDetector.BACKENDS


# =============================================================================
# RECOGNIZER (from recognizer.py)
# =============================================================================

"""Face recognition module.

This module provides face recognition functionality for identifying
watch list individuals from face images using multiple embedding backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceCategory(Enum):
    """Category of face for security classification.
    
    This system is designed to detect people you want to be alerted about:
    - WATCH_LIST: People identified by photo (you registered their face)
    - THREAT_PROFILE: People matching attribute filters (glasses, beard, etc.)
    - NO_MATCH: Face detected but doesn't match any threat criteria
    """
    WATCH_LIST = "watch_list"       # Person from registered watch list (photo-based)
    THREAT_PROFILE = "threat_profile"  # Matched via attribute filter (feature-based)
    NO_MATCH = "no_match"           # Detected but no threat criteria matched


@dataclass
class RecognitionResult:
    """Result of face recognition.
    
    This represents whether a detected face matches threat criteria:
    - On watch list (photo-based identification)
    - Matches a threat profile (attribute-based detection)
    """
    
    identity: str
    confidence: float
    category: FaceCategory = FaceCategory.NO_MATCH
    embedding: Optional[np.ndarray] = None
    face_id: Optional[int] = None  # Database ID if matched
    attribute_profile: Optional["AttributeProfile"] = None  # Detected attributes
    matched_filter: Optional[str] = None  # Name of matched attribute filter
    
    @property
    def is_threat(self) -> bool:
        """Check if face matches any threat criteria (watch list or profile)."""
        return self.category in (FaceCategory.WATCH_LIST, FaceCategory.THREAT_PROFILE)
    
    @property
    def is_watch_list(self) -> bool:
        """Check if face is on watch list (photo-based match)."""
        return self.category == FaceCategory.WATCH_LIST
    
    @property
    def is_threat_profile(self) -> bool:
        """Check if face matches a threat profile (attribute-based)."""
        return self.category == FaceCategory.THREAT_PROFILE
    
    @property
    def should_alert(self) -> bool:
        """Check if this result should trigger a security alert.
        
        Alerts when face matches any threat criteria:
        - WATCH_LIST: Person you registered to be alerted about
        - THREAT_PROFILE: Person matching attribute filters you defined
        """
        return self.category in (FaceCategory.WATCH_LIST, FaceCategory.THREAT_PROFILE)
    
    # Backwards compatibility aliases
    @property
    def is_known(self) -> bool:
        """Deprecated: Use is_threat instead."""
        return self.is_threat
    
    @property
    def is_attribute_match(self) -> bool:
        """Deprecated: Use is_threat_profile instead."""
        return self.is_threat_profile


# =============================================================================
# Face Attribute Detection (Decorator Pattern)
# =============================================================================

class FaceAttribute(Enum):
    """Detectable face attributes for filtering."""
    GLASSES = "glasses"
    SUNGLASSES = "sunglasses"
    BEARD = "beard"
    MUSTACHE = "mustache"
    BALD = "bald"
    BLONDE_HAIR = "blonde_hair"
    BROWN_HAIR = "brown_hair"
    BLACK_HAIR = "black_hair"
    RED_HAIR = "red_hair"
    GRAY_HAIR = "gray_hair"
    TATTOO = "tattoo"
    HAT = "hat"
    MASK = "mask"
    YOUNG = "young"
    MIDDLE_AGED = "middle_aged"
    SENIOR = "senior"
    MALE = "male"
    FEMALE = "female"


@dataclass
class AttributeResult:
    """Result of a single attribute detection."""
    attribute: FaceAttribute
    detected: bool
    confidence: float
    
    def __bool__(self) -> bool:
        return self.detected


@dataclass
class AttributeProfile:
    """Complete attribute profile for a face."""
    attributes: Dict[FaceAttribute, AttributeResult]
    
    def has(self, attr: FaceAttribute) -> bool:
        """Check if face has the given attribute."""
        if attr in self.attributes:
            return self.attributes[attr].detected
        return False
    
    def get_confidence(self, attr: FaceAttribute) -> float:
        """Get confidence for an attribute."""
        if attr in self.attributes:
            return self.attributes[attr].confidence
        return 0.0
    
    def matches(self, required: List[FaceAttribute], min_confidence: float = 0.5) -> bool:
        """Check if face matches ALL required attributes."""
        for attr in required:
            if attr not in self.attributes:
                return False
            result = self.attributes[attr]
            if not result.detected or result.confidence < min_confidence:
                return False
        return True
    
    def match_score(self, required: List[FaceAttribute]) -> float:
        """Calculate match score for required attributes (0.0 to 1.0)."""
        if not required:
            return 0.0
        total = 0.0
        for attr in required:
            if attr in self.attributes and self.attributes[attr].detected:
                total += self.attributes[attr].confidence
        return total / len(required)
    
    @property
    def detected_attributes(self) -> List[FaceAttribute]:
        """List of all detected attributes."""
        return [attr for attr, result in self.attributes.items() if result.detected]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            attr.value: {"detected": r.detected, "confidence": r.confidence}
            for attr, r in self.attributes.items()
        }


class BaseAttributeDetector(ABC):
    """Abstract base class for face attribute detection."""
    
    @property
    @abstractmethod
    def supported_attributes(self) -> List[FaceAttribute]:
        """Return list of attributes this detector can detect."""
        pass
    
    @abstractmethod
    def detect(self, face_image: np.ndarray) -> AttributeProfile:
        """Detect attributes in a face image.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            AttributeProfile with detection results
        """
        pass


class HaarAttributeDetector(BaseAttributeDetector):
    """Simple attribute detector using Haar cascades and color analysis.
    
    This is a lightweight detector suitable for Raspberry Pi.
    For better accuracy, use a deep learning-based detector.
    """
    
    def __init__(self):
        self._cascades = {}
        self._load_cascades()
    
    def _load_cascades(self):
        """Load available Haar cascades for attribute detection."""
        cascade_dir: str = cv2.data.haarcascades  # type: ignore
        
        # Eye glasses detection uses eye cascade
        eye_cascade = f"{cascade_dir}haarcascade_eye.xml"
        glasses_cascade = f"{cascade_dir}haarcascade_eye_tree_eyeglasses.xml"
        
        if Path(eye_cascade).exists():
            self._cascades["eyes"] = cv2.CascadeClassifier(eye_cascade)
        if Path(glasses_cascade).exists():
            self._cascades["glasses"] = cv2.CascadeClassifier(glasses_cascade)
        
        logger.info(f"Loaded {len(self._cascades)} attribute cascades")
    
    @property
    def supported_attributes(self) -> List[FaceAttribute]:
        return [
            FaceAttribute.GLASSES,
            FaceAttribute.BEARD,
            FaceAttribute.BLONDE_HAIR,
            FaceAttribute.BROWN_HAIR,
            FaceAttribute.BLACK_HAIR,
            FaceAttribute.GRAY_HAIR,
        ]
    
    def detect(self, face_image: np.ndarray) -> AttributeProfile:
        """Detect attributes using Haar cascades and color analysis."""
        results = {}
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = face_image.shape[:2]
        
        # Glasses detection
        glasses_detected, glasses_conf = self._detect_glasses(gray, h, w)
        results[FaceAttribute.GLASSES] = AttributeResult(
            FaceAttribute.GLASSES, glasses_detected, glasses_conf
        )
        
        # Beard detection (analyze lower face region)
        beard_detected, beard_conf = self._detect_beard(face_image, h, w)
        results[FaceAttribute.BEARD] = AttributeResult(
            FaceAttribute.BEARD, beard_detected, beard_conf
        )
        
        # Hair color detection
        hair_color, hair_conf = self._detect_hair_color(face_image, h, w)
        for color in [FaceAttribute.BLONDE_HAIR, FaceAttribute.BROWN_HAIR, 
                      FaceAttribute.BLACK_HAIR, FaceAttribute.GRAY_HAIR]:
            is_this_color = (color == hair_color)
            results[color] = AttributeResult(
                color, is_this_color, hair_conf if is_this_color else 0.0
            )
        
        return AttributeProfile(results)
    
    def _detect_glasses(self, gray: np.ndarray, h: int, w: int) -> Tuple[bool, float]:
        """Detect glasses using eye region analysis."""
        if "glasses" not in self._cascades:
            return False, 0.0
        
        # Focus on upper face (eye region)
        eye_region = gray[int(h*0.2):int(h*0.5), :]
        
        glasses = self._cascades["glasses"].detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=3
        )
        
        if len(glasses) >= 1:
            return True, min(0.5 + len(glasses) * 0.15, 0.95)
        
        # Fallback: check for reflection patterns typical of glasses
        edges = cv2.Canny(eye_region, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        if edge_ratio > 0.15:  # High edge density in eye region
            return True, 0.6
        
        return False, 0.3
    
    def _detect_beard(self, face_image: np.ndarray, h: int, w: int) -> Tuple[bool, float]:
        """Detect beard by analyzing lower face texture."""
        # Lower third of face
        lower_face = face_image[int(h*0.65):, int(w*0.2):int(w*0.8)]
        
        if lower_face.size == 0:
            return False, 0.0
        
        gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture using standard deviation
        texture = np.std(gray)
        
        # Bearded faces typically have higher texture variance
        if texture > 40:
            return True, min(0.5 + (texture - 40) / 60, 0.9)
        
        return False, max(0.1, 0.5 - texture / 80)
    
    def _detect_hair_color(self, face_image: np.ndarray, h: int, w: int) -> Tuple[FaceAttribute, float]:
        """Detect hair color from upper face region."""
        # Upper portion (hair region)
        hair_region = face_image[:int(h*0.25), int(w*0.2):int(w*0.8)]
        
        if hair_region.size == 0:
            return FaceAttribute.BROWN_HAIR, 0.3
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        
        # Calculate average values
        avg_h = np.mean(hsv[:, :, 0])
        avg_s = np.mean(hsv[:, :, 1])
        avg_v = np.mean(hsv[:, :, 2])
        
        # Classify hair color based on HSV values
        if avg_v < 50 and avg_s < 50:
            return FaceAttribute.BLACK_HAIR, 0.7
        elif avg_v > 180 and avg_s < 30:
            return FaceAttribute.GRAY_HAIR, 0.65
        elif avg_h < 25 and avg_s > 100 and avg_v > 150:
            return FaceAttribute.BLONDE_HAIR, 0.6
        else:
            return FaceAttribute.BROWN_HAIR, 0.55


class AttributeFilter:
    """Composable filter for face attributes (decorator pattern).
    
    Allows combining multiple attribute requirements that must all match.
    Can be used as fallback when face recognition doesn't find a match.
    
    Example:
        # Create a filter for "person with glasses and beard"
        filter = AttributeFilter()
        filter.require(FaceAttribute.GLASSES)
        filter.require(FaceAttribute.BEARD)
        
        # Check if face matches
        if filter.matches(attribute_profile):
            print("Match found!")
        
        # Chain filters (decorator pattern)
        filter1 = AttributeFilter().require(FaceAttribute.GLASSES)
        filter2 = filter1.and_require(FaceAttribute.BEARD)
        filter3 = filter2.and_require(FaceAttribute.BLONDE_HAIR)
    """
    
    def __init__(
        self,
        required_attributes: Optional[List[FaceAttribute]] = None,
        min_confidence: float = 0.5,
        match_all: bool = True,
    ):
        """Initialize attribute filter.
        
        Args:
            required_attributes: List of attributes to require
            min_confidence: Minimum confidence threshold for each attribute
            match_all: If True, ALL attributes must match. If False, ANY can match.
        """
        self.required: List[FaceAttribute] = required_attributes or []
        self.min_confidence = min_confidence
        self.match_all = match_all
    
    def require(self, attribute: FaceAttribute) -> "AttributeFilter":
        """Add a required attribute. Returns self for chaining."""
        if attribute not in self.required:
            self.required.append(attribute)
        return self
    
    def and_require(self, attribute: FaceAttribute) -> "AttributeFilter":
        """Create new filter with additional requirement (decorator pattern)."""
        new_filter = AttributeFilter(
            required_attributes=self.required.copy(),
            min_confidence=self.min_confidence,
            match_all=self.match_all,
        )
        new_filter.require(attribute)
        return new_filter
    
    def or_require(self, attribute: FaceAttribute) -> "AttributeFilter":
        """Create new filter that matches ANY of the attributes."""
        new_filter = AttributeFilter(
            required_attributes=self.required.copy(),
            min_confidence=self.min_confidence,
            match_all=False,
        )
        new_filter.require(attribute)
        return new_filter
    
    def matches(self, profile: AttributeProfile) -> bool:
        """Check if the profile matches this filter's requirements."""
        if not self.required:
            return False
        
        if self.match_all:
            return profile.matches(self.required, self.min_confidence)
        else:
            # Match ANY
            for attr in self.required:
                if attr in profile.attributes:
                    result = profile.attributes[attr]
                    if result.detected and result.confidence >= self.min_confidence:
                        return True
            return False
    
    def score(self, profile: AttributeProfile) -> float:
        """Calculate match score (0.0 to 1.0)."""
        return profile.match_score(self.required)
    
    def __repr__(self) -> str:
        attrs = [a.value for a in self.required]
        mode = "ALL" if self.match_all else "ANY"
        return f"AttributeFilter({mode}: {attrs}, min_conf={self.min_confidence})"


class AttributeFilterChain:
    """Chain of attribute filters for complex matching scenarios.
    
    Supports multiple named filter profiles that can be checked against.
    
    Example:
        chain = AttributeFilterChain()
        chain.add_profile("suspicious_type_1", 
            AttributeFilter().require(FaceAttribute.GLASSES).require(FaceAttribute.BEARD))
        chain.add_profile("suspicious_type_2",
            AttributeFilter().require(FaceAttribute.TATTOO).require(FaceAttribute.BALD))
        
        matches = chain.find_matches(profile)
        # Returns: ["suspicious_type_1"] if glasses+beard detected
    """
    
    def __init__(self):
        self.profiles: Dict[str, AttributeFilter] = {}
    
    def add_profile(self, name: str, filter: AttributeFilter) -> "AttributeFilterChain":
        """Add a named filter profile."""
        self.profiles[name] = filter
        return self
    
    def remove_profile(self, name: str) -> "AttributeFilterChain":
        """Remove a filter profile."""
        self.profiles.pop(name, None)
        return self
    
    def find_matches(self, profile: AttributeProfile) -> List[Tuple[str, float]]:
        """Find all matching profiles with scores.
        
        Returns:
            List of (profile_name, score) tuples, sorted by score descending
        """
        matches = []
        for name, filter in self.profiles.items():
            if filter.matches(profile):
                score = filter.score(profile)
                matches.append((name, score))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def best_match(self, profile: AttributeProfile) -> Optional[Tuple[str, float]]:
        """Get the best matching profile."""
        matches = self.find_matches(profile)
        return matches[0] if matches else None


class BaseEmbeddingBackend(ABC):
    """Abstract base class for face embedding extraction backends."""
    
    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Return the dimensionality of embeddings."""
        pass
    
    @abstractmethod
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding from an image.
        
        Args:
            face_image: Cropped face image as numpy array (BGR format)
            
        Returns:
            Face embedding vector
        """
        pass
    
    @abstractmethod
    def extract_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings from multiple face images.
        
        Args:
            face_images: List of cropped face images
            
        Returns:
            List of embedding vectors
        """
        pass


class DlibEmbeddingBackend(BaseEmbeddingBackend):
    """Face embedding using dlib/face_recognition library (128D).
    
    Uses the pre-trained dlib model which is robust and well-tested.
    Slower than TFLite models but more accurate for small datasets.
    """
    
    def __init__(self):
        try:
            import face_recognition
            self._face_recognition = face_recognition
            logger.info("Initialized dlib embedding backend (128D)")
        except ImportError:
            raise ImportError(
                "face_recognition library not installed. "
                "Install with: pip install face-recognition"
            )
    
    @property
    def embedding_size(self) -> int:
        return 128
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """Extract 128D face embedding using dlib."""
        if face_image is None or face_image.size == 0:
            raise ValueError("Empty or invalid face image")
        
        # face_recognition expects RGB format
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = face_image
        
        # Get face encodings - assume the whole image is a face
        # Use the whole image as the face location
        h, w = rgb_image.shape[:2]
        face_locations = [(0, w, h, 0)]  # top, right, bottom, left
        
        encodings = self._face_recognition.face_encodings(
            rgb_image, 
            known_face_locations=face_locations,
            num_jitters=1
        )
        
        if not encodings:
            # Fallback: try without specifying location
            encodings = self._face_recognition.face_encodings(rgb_image, num_jitters=1)
        
        if not encodings:
            raise ValueError("Could not extract embedding from face image")
        
        return np.array(encodings[0], dtype=np.float32)
    
    def extract_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings from multiple faces."""
        embeddings = []
        for face_image in face_images:
            try:
                embedding = self.extract(face_image)
                embeddings.append(embedding)
            except ValueError as e:
                logger.warning(f"Failed to extract embedding: {e}")
                # Return zero embedding for failed extractions
                embeddings.append(np.zeros(self.embedding_size, dtype=np.float32))
        return embeddings


class TFLiteEmbeddingBackend(BaseEmbeddingBackend):
    """Face embedding using TFLite MobileFaceNet model (512D).
    
    Optimized for edge deployment on Raspberry Pi.
    Fast inference (~50ms per face on RPi4B).
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._input_shape = (112, 112)  # MobileFaceNet input size
        
        if self.model_path and self.model_path.exists():
            self._load_model()
        else:
            logger.warning(
                f"TFLite model not found at {model_path}. "
                "Embedding extraction will fail until model is loaded."
            )
    
    def _load_model(self):
        """Load TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                raise ImportError(
                    "TFLite runtime not installed. "
                    "Install with: pip install tflite-runtime"
                )
        
        self._interpreter = tflite.Interpreter(model_path=str(self.model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        
        # Get input shape from model
        input_shape = self._input_details[0]['shape']
        self._input_shape = (input_shape[1], input_shape[2])
        
        logger.info(f"Loaded TFLite model from {self.model_path}")
    
    @property
    def embedding_size(self) -> int:
        if self._output_details:
            return self._output_details[0]['shape'][-1]
        return 512  # Default MobileFaceNet embedding size
    
    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input."""
        # Resize to model input size
        face = cv2.resize(face_image, self._input_shape)
        
        # Convert BGR to RGB
        if len(face.shape) == 3 and face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using TFLite model."""
        if self._interpreter is None:
            raise RuntimeError("TFLite model not loaded")
        
        if face_image is None or face_image.size == 0:
            raise ValueError("Empty or invalid face image")
        
        # Preprocess
        input_data = self._preprocess(face_image)
        
        # Run inference
        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        self._interpreter.invoke()
        
        # Get output
        embedding = self._interpreter.get_tensor(self._output_details[0]['index'])
        embedding = embedding.flatten().astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def extract_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings from multiple faces."""
        embeddings = []
        for face_image in face_images:
            try:
                embedding = self.extract(face_image)
                embeddings.append(embedding)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to extract embedding: {e}")
                embeddings.append(np.zeros(self.embedding_size, dtype=np.float32))
        return embeddings


class MobileNetV2EmbeddingBackend(BaseEmbeddingBackend):
    """Face embedding using MobileNetV2 from Keras (1280D or custom).
    
    Works on x86, ARM64, and Raspberry Pi.
    Uses Keras with TensorFlow backend - can export to TFLite for edge deployment.
    
    The model is built on MobileNetV2 pretrained on ImageNet, with custom
    head for face embedding extraction. Training is done with triplet loss.
    """
    
    # Pre-trained weights URL (you can host your own or use a public one)
    WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
    
    def __init__(
        self, 
        model_path: Optional[Union[str, Path]] = None,
        embedding_dim: int = 512,
        use_pretrained: bool = True,
    ):
        """Initialize MobileNetV2 embedding backend.
        
        Args:
            model_path: Path to saved model weights (optional)
            embedding_dim: Output embedding dimensionality (default 512)
            use_pretrained: Whether to use pretrained ImageNet weights for base
        """
        self.model_path = Path(model_path) if model_path else None
        self._embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained
        self._model = None
        self._input_shape = (160, 160, 3)  # Standard face input size
        
        self._build_model()
    
    def _build_model(self):
        """Build the MobileNetV2-based embedding model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
            from tensorflow.keras.models import Model
        except ImportError:
            try:
                # Try standalone keras
                from keras.applications import MobileNetV2
                from keras.layers import Dense, GlobalAveragePooling2D, Input
                from keras.models import Model
            except ImportError:
                raise ImportError(
                    "TensorFlow/Keras not installed. "
                    "Install with: pip install tensorflow"
                )
        
        # Check if we have a saved model
        if self.model_path and self.model_path.exists():
            try:
                self._model = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"Loaded MobileNetV2 model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Building new model.")
        
        # Build MobileNetV2 base
        weights = 'imagenet' if self.use_pretrained else None
        base_model = MobileNetV2(
            input_shape=self._input_shape,
            include_top=False,
            weights=weights,
        )
        
        # Freeze base layers for transfer learning
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add embedding head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self._embedding_dim, name='embedding')(x)
        
        self._model = Model(inputs=base_model.input, outputs=x)
        
        logger.info(
            f"Built MobileNetV2 embedding model "
            f"(input: {self._input_shape}, output: {self._embedding_dim}D)"
        )
    
    @property
    def embedding_size(self) -> int:
        return self._embedding_dim
    
    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for MobileNetV2."""
        # Resize to input shape
        face = cv2.resize(face_image, (self._input_shape[0], self._input_shape[1]))
        
        # Convert BGR to RGB
        if len(face.shape) == 3 and face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # MobileNetV2 preprocessing: scale to [-1, 1]
        face = face.astype(np.float32) / 127.5 - 1.0
        
        return face
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using MobileNetV2."""
        if self._model is None:
            raise RuntimeError("MobileNetV2 model not loaded")
        
        if face_image is None or face_image.size == 0:
            raise ValueError("Empty or invalid face image")
        
        # Preprocess
        face = self._preprocess(face_image)
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        
        # Run inference
        embedding = self._model.predict(face, verbose=0)
        embedding = embedding.flatten().astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def extract_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings from multiple faces (batch processing)."""
        if self._model is None:
            raise RuntimeError("MobileNetV2 model not loaded")
        
        if not face_images:
            return []
        
        # Preprocess all faces
        batch = np.array([self._preprocess(f) for f in face_images])
        
        # Batch inference
        embeddings_batch = self._model.predict(batch, verbose=0)
        
        # Normalize each embedding
        embeddings = []
        for emb in embeddings_batch:
            emb = emb.flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
        
        return embeddings
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save the model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        logger.info(f"Saved MobileNetV2 model to {path}")
    
    def export_tflite(self, output_path: Union[str, Path]) -> None:
        """Export model to TFLite format for edge deployment."""
        if self._model is None:
            raise RuntimeError("No model to export")
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for TFLite export")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"Exported TFLite model to {output_path}")


class OpenCVDNNEmbeddingBackend(BaseEmbeddingBackend):
    """Face embedding using OpenCV DNN with a pre-trained model.
    
    This backend uses OpenCV's DNN module to run face embedding models,
    works on all platforms without extra dependencies beyond OpenCV.
    """
    
    # OpenFace model (small, fast, 128D embeddings)
    MODEL_URL = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        input_size: Tuple[int, int] = (96, 96),
    ):
        """Initialize OpenCV DNN embedding backend.
        
        Args:
            model_path: Path to model file (optional, will download)
            input_size: Input image size for the network
        """
        self.input_size = input_size
        self._embedding_dim = 128  # OpenFace output size
        
        self.net = self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[Union[str, Path]]) -> cv2.dnn.Net:
        """Load the embedding model."""
        default_model_dir = Path(__file__).parent.parent / "data" / "models" / "face_recognition"
        default_model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_path is None:
            model_path = default_model_dir / "openface_nn4.small2.v1.t7"
        
        model_path = Path(model_path)
        
        # Download if not exists
        if not model_path.exists():
            logger.info(f"Downloading OpenFace model to {model_path}...")
            self._download_file(self.MODEL_URL, model_path)
        
        # Load network
        net = cv2.dnn.readNetFromTorch(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        logger.info(f"Loaded OpenFace model from {model_path}")
        return net
    
    def _download_file(self, url: str, path: Path) -> None:
        """Download a file from URL."""
        import urllib.request
        try:
            urllib.request.urlretrieve(url, str(path))
            logger.info(f"Downloaded to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {url}: {e}")
    
    @property
    def embedding_size(self) -> int:
        return self._embedding_dim
    
    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face for OpenFace model with expression-invariant normalization."""
        # Resize to model input size
        face = cv2.resize(face_image, self.input_size)
        
        # Convert to LAB color space for better lighting normalization
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab = cv2.merge([l, a, b])
        face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Create blob (OpenFace uses specific preprocessing)
        blob = cv2.dnn.blobFromImage(
            face, 1.0 / 255, self.input_size,
            (0, 0, 0), swapRB=True, crop=False
        )
        return blob
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using OpenCV DNN."""
        if face_image is None or face_image.size == 0:
            raise ValueError("Empty or invalid face image")
        
        # Preprocess
        blob = self._preprocess(face_image)
        
        # Run inference
        self.net.setInput(blob)
        embedding = self.net.forward()
        embedding = embedding.flatten().astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def extract_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings from multiple faces."""
        embeddings = []
        for face_image in face_images:
            try:
                embedding = self.extract(face_image)
                embeddings.append(embedding)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to extract embedding: {e}")
                embeddings.append(np.zeros(self.embedding_size, dtype=np.float32))
        return embeddings


# Registry of available embedding backends
EMBEDDING_BACKENDS = {
    "dlib": DlibEmbeddingBackend,
    "face_recognition": DlibEmbeddingBackend,  # Alias
    "tflite": TFLiteEmbeddingBackend,
    "mobilefacenet": TFLiteEmbeddingBackend,  # Alias
    "mobilenetv2": MobileNetV2EmbeddingBackend,  # Keras MobileNetV2
    "keras": MobileNetV2EmbeddingBackend,  # Alias
    "opencv_dnn": OpenCVDNNEmbeddingBackend,  # OpenCV DNN (OpenFace)
    "openface": OpenCVDNNEmbeddingBackend,  # Alias
}


class FaceRecognizer:
    """Face recognition using face embeddings and similarity matching.
    
    Supports multiple embedding backends and maintains a registry of
    watch list faces for matching.
    
    Available backends:
    - opencv_dnn: OpenFace model via OpenCV DNN (default, works everywhere)
    - mobilenetv2: Keras MobileNetV2 model (requires tensorflow)
    - tflite: TFLite model for edge deployment
    - dlib: dlib/face_recognition library (requires compilation)
    """
    
    def __init__(
        self,
        model: str = "opencv_dnn",
        threshold: float = 0.6,
        watch_list_dir: Optional[Path] = None,
        tflite_model_path: Optional[Path] = None,
    ):
        """Initialize face recognizer.
        
        Args:
            model: Embedding backend to use (opencv_dnn, mobilenetv2, tflite, dlib)
            threshold: Similarity threshold for recognition
            watch_list_dir: Directory containing watch list face images
            tflite_model_path: Path to TFLite model (required for tflite backend)
        """
        self.model_name = model.lower()
        self.threshold = threshold
        self.watch_list_dir = Path(watch_list_dir) if watch_list_dir else None
        self.tflite_model_path = Path(tflite_model_path) if tflite_model_path else None
        
        # Storage for face embeddings: name -> list of (embedding, category, face_id)
        self.watch_list: Dict[str, List[Tuple[np.ndarray, FaceCategory, Optional[int]]]] = {}
        
        # Initialize embedding backend
        self._init_backend()
        
        # Load watch list faces if directory provided
        if self.watch_list_dir and self.watch_list_dir.exists():
            self.load_watch_list(self.watch_list_dir)
    
    def _init_backend(self):
        """Initialize the embedding backend."""
        if self.model_name not in EMBEDDING_BACKENDS:
            raise ValueError(
                f"Unknown backend: {self.model_name}. "
                f"Available: {list(EMBEDDING_BACKENDS.keys())}"
            )
        
        backend_class = EMBEDDING_BACKENDS[self.model_name]
        
        if self.model_name in ("tflite", "mobilefacenet"):
            self.backend = backend_class(model_path=self.tflite_model_path)
        else:
            self.backend = backend_class()
        
        logger.info(f"Initialized {self.model_name} backend (embedding size: {self.backend.embedding_size})")
    
    @property
    def embedding_size(self) -> int:
        """Get the embedding dimensionality."""
        return self.backend.embedding_size
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding from an image.
        
        Args:
            face_image: Cropped face image as numpy array
            
        Returns:
            Face embedding vector
        """
        return self.backend.extract(face_image)
    
    def load_watch_list(self, faces_dir: Path) -> None:
        """Load watch list faces from a directory.
        
        All faces in this directory are people you want to be alerted about.
        
        Args:
            faces_dir: Directory with subdirectories for each person to watch.
        """
        faces_dir = Path(faces_dir)
        
        for person_dir in faces_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                
                # All faces are on watch list (people we want alerts for)
                category = FaceCategory.WATCH_LIST
                display_name = person_name
                
                self.watch_list[display_name] = []
                
                # Load all images for this person
                image_extensions = ["*.jpg", "*.jpeg", "*.png"]
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(person_dir.glob(ext))
                
                for image_path in image_paths:
                    try:
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            embedding = self.get_embedding(image)
                            self.watch_list[display_name].append(
                                (embedding, category, None)
                            )
                    except Exception as e:
                        logger.warning(f"Failed to process {image_path}: {e}")
                
                count = len(self.watch_list[display_name])
                logger.info(f"Loaded {count} faces for {display_name} (category: {category.value})")
    
    def enroll(
        self, 
        name: str, 
        face_image: np.ndarray,
        category: FaceCategory = FaceCategory.WATCH_LIST,
        face_id: Optional[int] = None,
    ) -> np.ndarray:
        """Enroll a face to watch list.
        
        Args:
            name: Person's name/identifier (who you want to be alerted about)
            face_image: Face image to enroll
            category: Face category (default: WATCH_LIST)
            face_id: Database ID for this face
            
        Returns:
            The extracted embedding
        """
        embedding = self.get_embedding(face_image)
        
        if name not in self.watch_list:
            self.watch_list[name] = []
        
        self.watch_list[name].append((embedding, category, face_id))
        logger.info(f"Enrolled face for {name} (category: {category.value})")
        
        return embedding
    
    def enroll_embedding(
        self,
        name: str,
        embedding: np.ndarray,
        category: FaceCategory = FaceCategory.WATCH_LIST,
        face_id: Optional[int] = None,
    ) -> None:
        """Enroll a pre-computed embedding to watch list.
        
        Args:
            name: Person's name/identifier
            embedding: Pre-computed face embedding
            category: Face category (default: WATCH_LIST)
            face_id: Database ID for this face
        """
        if name not in self.watch_list:
            self.watch_list[name] = []
        
        self.watch_list[name].append((embedding, category, face_id))
        logger.debug(f"Enrolled embedding for {name} (category: {category.value})")
    
    def remove_face(self, name: str, face_id: Optional[int] = None) -> bool:
        """Remove a face from the recognizer.
        
        Args:
            name: Person's name/identifier
            face_id: Specific face ID to remove, or None to remove all faces for this person
            
        Returns:
            True if face was removed, False if not found
        """
        if name not in self.watch_list:
            return False
        
        if face_id is None:
            # Remove all faces for this person
            del self.watch_list[name]
            logger.info(f"Removed all faces for {name}")
            return True
        
        # Remove specific face by ID
        original_len = len(self.watch_list[name])
        self.watch_list[name] = [
            (emb, cat, fid) for emb, cat, fid in self.watch_list[name]
            if fid != face_id
        ]
        
        if len(self.watch_list[name]) < original_len:
            if not self.watch_list[name]:
                del self.watch_list[name]
            logger.info(f"Removed face {face_id} for {name}")
            return True
        
        return False
    
    def clear_embeddings(self) -> None:
        """Clear all enrolled embeddings."""
        self.watch_list.clear()
        logger.info("Cleared all face embeddings")
    
    def get_enrolled_names(self) -> List[str]:
        """Get list of enrolled person names."""
        return list(self.watch_list.keys())
    
    def get_enrolled_count(self, name: Optional[str] = None) -> int:
        """Get count of enrolled faces.
        
        Args:
            name: Person name, or None for total count
            
        Returns:
            Number of enrolled faces
        """
        if name:
            return len(self.watch_list.get(name, []))
        return sum(len(faces) for faces in self.watch_list.values())
    
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize a face.
        
        Args:
            face_image: Face image to recognize
            
        Returns:
            RecognitionResult with identity, confidence, and category
        """
        if not self.watch_list:
            return RecognitionResult(
                identity="unknown", 
                confidence=0.0,
                category=FaceCategory.NO_MATCH
            )
        
        # Get embedding for query face
        try:
            query_embedding = self.get_embedding(face_image)
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return RecognitionResult(
                identity="unknown",
                confidence=0.0,
                category=FaceCategory.NO_MATCH
            )
        
        # Find best match
        best_match = "unknown"
        best_similarity = 0.0
        best_category = FaceCategory.NO_MATCH
        best_face_id = None
        
        for name, face_entries in self.watch_list.items():
            for enrolled_embedding, category, face_id in face_entries:
                similarity = self._cosine_similarity(query_embedding, enrolled_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
                    best_category = category
                    best_face_id = face_id
        
        # Check threshold
        if best_similarity < self.threshold:
            return RecognitionResult(
                identity="unknown",
                confidence=best_similarity,
                category=FaceCategory.NO_MATCH,
                embedding=query_embedding,
            )
        
        return RecognitionResult(
            identity=best_match,
            confidence=best_similarity,
            category=best_category,
            embedding=query_embedding,
            face_id=best_face_id,
        )
    
    def recognize_embedding(self, query_embedding: np.ndarray) -> RecognitionResult:
        """Recognize a face from a pre-computed embedding.
        
        Args:
            query_embedding: Face embedding to match
            
        Returns:
            RecognitionResult with identity, confidence, and category
        """
        if not self.watch_list:
            return RecognitionResult(
                identity="unknown",
                confidence=0.0,
                category=FaceCategory.NO_MATCH,
                embedding=query_embedding
            )
        
        best_match = "unknown"
        best_similarity = 0.0
        best_category = FaceCategory.NO_MATCH
        best_face_id = None
        
        for name, face_entries in self.watch_list.items():
            for enrolled_embedding, category, face_id in face_entries:
                similarity = self._cosine_similarity(query_embedding, enrolled_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
                    best_category = category
                    best_face_id = face_id
        
        if best_similarity < self.threshold:
            return RecognitionResult(
                identity="unknown",
                confidence=best_similarity,
                category=FaceCategory.NO_MATCH,
                embedding=query_embedding,
            )
        
        return RecognitionResult(
            identity=best_match,
            confidence=best_similarity,
            category=best_category,
            embedding=query_embedding,
            face_id=best_face_id,
        )
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# PIPELINE (from pipeline.py)
# =============================================================================

"""End-to-end face security pipeline.

Motion-gated pipeline that detects faces then runs recognition.
Designed to work with MCU-triggered cameras (ESP32/Arduino) or
laptop webcams via OpenCV.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional
import logging

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class FaceEvent:
    """Combined detection and recognition result for a single face."""

    detection: DetectedFace
    recognition: RecognitionResult
    is_threat: bool
    attribute_profile: Optional[AttributeProfile] = None


class DetectionMode(Enum):
    """Detection mode for the security pipeline."""
    EMBEDDING_ONLY = "embedding_only"      # Only use face embedding matching
    ATTRIBUTE_ONLY = "attribute_only"      # Only use attribute-based matching
    EMBEDDING_FIRST = "embedding_first"    # Try embedding first, fallback to attributes
    ATTRIBUTE_FIRST = "attribute_first"    # Try attributes first, fallback to embedding
    BOTH = "both"                          # Run both, combine results


class FaceSecurityPipeline:
    """Runs motion-gated face detection followed by recognition.
    
    Supports multiple detection modes:
    - EMBEDDING_ONLY: Only match against watch list database (traditional)
    - ATTRIBUTE_ONLY: Only use attribute filters (no face DB needed)
    - EMBEDDING_FIRST: Try embedding match, fallback to attributes if no match
    - ATTRIBUTE_FIRST: Try attributes first, fallback to embedding if no match
    - BOTH: Run both methods, combine results
    
    Example:
        # Mode 1: Traditional face recognition only
        pipeline = FaceSecurityPipeline(mode=DetectionMode.EMBEDDING_ONLY)
        
        # Mode 2: Attribute detection only (no face database needed)
        pipeline = FaceSecurityPipeline(mode=DetectionMode.ATTRIBUTE_ONLY)
        pipeline.add_attribute_filter(
            "suspicious",
            AttributeFilter()
                .require(FaceAttribute.GLASSES)
                .require(FaceAttribute.BEARD)
        )
        
        # Mode 3: Embedding with attribute fallback
        pipeline = FaceSecurityPipeline(mode=DetectionMode.EMBEDDING_FIRST)
        pipeline.add_attribute_filter("suspicious", ...)
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        recognizer_service=None,
        motion_sensor=None,
        camera=None,
        motion_check: Optional[Callable[[], bool]] = None,
        recognition_margin: float = 0.2,
        attribute_detector: Optional[BaseAttributeDetector] = None,
        mode: DetectionMode = DetectionMode.EMBEDDING_FIRST,
    ):
        """Initialize the security pipeline.
        
        Args:
            detector: Face detector instance
            recognizer: Face recognizer instance
            recognizer_service: Face recognizer service (alternative to recognizer)
            motion_sensor: Motion sensor for motion-gated detection
            camera: Camera interface
            motion_check: Callable to check motion status
            recognition_margin: Margin for cropping faces
            attribute_detector: Attribute detector instance
            mode: Detection mode (see DetectionMode enum)
        """
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer
        self.recognizer_service = recognizer_service
        self.motion_sensor = motion_sensor
        self.camera = camera
        self.motion_check = motion_check
        self.recognition_margin = recognition_margin
        self.mode = mode
        
        # Attribute detection
        self.attribute_detector = attribute_detector
        self.attribute_filters = AttributeFilterChain()

        # Only initialize recognizer if mode requires it
        if self.mode in (DetectionMode.EMBEDDING_ONLY, DetectionMode.EMBEDDING_FIRST, 
                         DetectionMode.ATTRIBUTE_FIRST, DetectionMode.BOTH):
            if self.recognizer is None and self.recognizer_service is None:
                self.recognizer = FaceRecognizer()
    
    def set_mode(self, mode: DetectionMode) -> "FaceSecurityPipeline":
        """Change detection mode at runtime."""
        self.mode = mode
        return self
    
    def add_attribute_filter(
        self, 
        name: str, 
        filter: AttributeFilter
    ) -> "FaceSecurityPipeline":
        """Add an attribute filter.
        
        Args:
            name: Unique name for this filter profile
            filter: AttributeFilter with required attributes
            
        Returns:
            Self for chaining
        """
        self.attribute_filters.add_profile(name, filter)
        
        # Auto-create attribute detector if needed
        if self.attribute_detector is None:
            self.attribute_detector = HaarAttributeDetector()
        
        return self
    
    def remove_attribute_filter(self, name: str) -> "FaceSecurityPipeline":
        """Remove an attribute filter."""
        self.attribute_filters.remove_profile(name)
        return self
    
    def clear_attribute_filters(self) -> "FaceSecurityPipeline":
        """Remove all attribute filters."""
        self.attribute_filters = AttributeFilterChain()
        return self

    def _motion_triggered(self) -> bool:
        if self.motion_check:
            return bool(self.motion_check())
        if self.motion_sensor and hasattr(self.motion_sensor, "read"):
            return bool(self.motion_sensor.read())
        return True

    def _recognize(self, face_image: np.ndarray) -> RecognitionResult:
        if self.recognizer_service is not None:
            return self.recognizer_service.recognize(face_image)
        return self.recognizer.recognize(face_image)
    
    def _detect_attributes(self, face_image: np.ndarray) -> Optional[AttributeProfile]:
        """Detect attributes in a face image."""
        if self.attribute_detector is None:
            return None
        try:
            return self.attribute_detector.detect(face_image)
        except Exception as e:
            logger.warning(f"Attribute detection failed: {e}")
            return None
    
    def _recognize_by_embedding(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize face using embedding matching."""
        if self.recognizer_service is not None:
            return self.recognizer_service.recognize(face_image)
        if self.recognizer is not None:
            return self.recognizer.recognize(face_image)
        return RecognitionResult(
            identity="unknown",
            confidence=0.0,
            category=FaceCategory.NO_MATCH
        )
    
    def _recognize_by_attributes(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize face using attribute matching."""
        if not self.attribute_filters.profiles:
            return RecognitionResult(
                identity="unknown",
                confidence=0.0,
                category=FaceCategory.NO_MATCH
            )
        
        profile = self._detect_attributes(face_image)
        if profile is None:
            return RecognitionResult(
                identity="unknown",
                confidence=0.0,
                category=FaceCategory.NO_MATCH
            )
        
        match = self.attribute_filters.best_match(profile)
        if match:
            filter_name, score = match
            logger.info(f"Attribute match: {filter_name} (score={score:.2f})")
            return RecognitionResult(
                identity=f"attribute:{filter_name}",
                confidence=score,
                category=FaceCategory.THREAT_PROFILE,
                attribute_profile=profile,
                matched_filter=filter_name,
            )
        
        return RecognitionResult(
            identity="unknown",
            confidence=0.0,
            category=FaceCategory.NO_MATCH,
            attribute_profile=profile,
        )
    
    def _process_face(self, face_image: np.ndarray) -> RecognitionResult:
        """Process a face image according to the current detection mode."""
        
        if self.mode == DetectionMode.EMBEDDING_ONLY:
            # Only use face embedding matching
            return self._recognize_by_embedding(face_image)
        
        elif self.mode == DetectionMode.ATTRIBUTE_ONLY:
            # Only use attribute matching (skip embedding entirely)
            return self._recognize_by_attributes(face_image)
        
        elif self.mode == DetectionMode.EMBEDDING_FIRST:
            # Try embedding first, fallback to attributes if unknown
            result = self._recognize_by_embedding(face_image)
            if result.category == FaceCategory.NO_MATCH:
                attr_result = self._recognize_by_attributes(face_image)
                if attr_result.category == FaceCategory.THREAT_PROFILE:
                    return attr_result
                # Attach attribute profile even if no match
                result.attribute_profile = attr_result.attribute_profile
            return result
        
        elif self.mode == DetectionMode.ATTRIBUTE_FIRST:
            # Try attributes first, fallback to embedding if no match
            result = self._recognize_by_attributes(face_image)
            if result.category != FaceCategory.THREAT_PROFILE:
                embed_result = self._recognize_by_embedding(face_image)
                if embed_result.category != FaceCategory.NO_MATCH:
                    embed_result.attribute_profile = result.attribute_profile
                    return embed_result
            return result
        
        elif self.mode == DetectionMode.BOTH:
            # Run both, prefer embedding match but include attribute info
            embed_result = self._recognize_by_embedding(face_image)
            attr_result = self._recognize_by_attributes(face_image)
            
            # Attach attribute profile to embedding result
            embed_result.attribute_profile = attr_result.attribute_profile
            
            # If embedding found a match, use it
            if embed_result.category != FaceCategory.NO_MATCH:
                return embed_result
            
            # Otherwise use attribute result if it matched
            if attr_result.category == FaceCategory.THREAT_PROFILE:
                return attr_result
            
            return embed_result
        
        # Default fallback
        return self._recognize_by_embedding(face_image)

    def process_frame(self, frame: np.ndarray) -> List[FaceEvent]:
        """Detect faces in a frame and run recognition on each."""
        faces = self.detector.detect(frame)
        events: List[FaceEvent] = []

        for face in faces:
            face_img = crop_face(frame, face.bbox, margin=self.recognition_margin)
            
            # Process according to current detection mode
            rec = self._process_face(face_img)
            
            events.append(
                FaceEvent(
                    detection=face,
                    recognition=rec,
                    is_threat=rec.should_alert,
                    attribute_profile=rec.attribute_profile,
                )
            )

        return events

    def run(
        self,
        frames: Optional[Iterable[np.ndarray]] = None,
        on_event: Optional[Callable[[FaceEvent], None]] = None,
        stop_after: Optional[int] = None,
    ) -> None:
        """Iterate over frames (webcam or provided iterable) and emit events."""
        processed = 0

        stream = frames
        camera_started = False
        if stream is None and self.camera is not None:
            if hasattr(self.camera, "start"):
                self.camera.start()
                camera_started = True
            if hasattr(self.camera, "stream"):
                stream = self.camera.stream()

        if stream is None:
            raise RuntimeError("No frame source provided (camera or iterable).")

        try:
            for frame in stream:
                if stop_after is not None and processed >= stop_after:
                    break

                if not self._motion_triggered():
                    continue

                for event in self.process_frame(frame):
                    if on_event:
                        on_event(event)
                    else:
                        logger.debug(
                            "Face detected: bbox=%s id=%s cat=%s conf=%.2f",
                            event.detection.bbox,
                            event.recognition.identity,
                            event.recognition.category.value,
                            event.recognition.confidence,
                        )
                processed += 1
        finally:
            if camera_started and hasattr(self.camera, "stop"):
                try:
                    self.camera.stop()
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    logger.debug("Failed to stop camera: %s", exc)
