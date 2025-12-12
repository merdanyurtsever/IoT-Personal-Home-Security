"""MediaPipe face detector."""

import logging
from typing import List

import cv2
import numpy as np

from .base import BaseFaceDetector
from .types import DetectedFace

logger = logging.getLogger(__name__)


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
        """Initialize MediaPipe face detector."""
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
