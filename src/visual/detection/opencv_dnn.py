"""OpenCV DNN face detector."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .base import BaseFaceDetector
from .types import DetectedFace

logger = logging.getLogger(__name__)


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
        """Initialize OpenCV DNN face detector."""
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.nms_threshold = nms_threshold
        
        self.net = self._load_model(model_path, config_path)
        logger.info("Initialized OpenCV DNN face detector")
    
    def _load_model(
        self, 
        model_path: Optional[Union[str, Path]], 
        config_path: Optional[Union[str, Path]]
    ) -> cv2.dnn.Net:
        """Load the DNN model."""
        default_model_dir = Path(__file__).parent.parent.parent.parent / "data" / "models" / "face_detection"
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
        """Detect faces using OpenCV DNN with NMS."""
        h, w = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            image, 1.0, self.input_size,
            (104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            
            if confidence > self.confidence_threshold:
                x1_norm = detections[0, 0, i, 3]
                y1_norm = detections[0, 0, i, 4]
                x2_norm = detections[0, 0, i, 5]
                y2_norm = detections[0, 0, i, 6]
                
                if not (0 <= x1_norm <= 1 and 0 <= y1_norm <= 1 and 
                        0 <= x2_norm <= 1 and 0 <= y2_norm <= 1):
                    continue
                
                x1 = int(x1_norm * w)
                y1 = int(y1_norm * h)
                x2 = int(x2_norm * w)
                y2 = int(y2_norm * h)
                
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                if box_w < 30 or box_h < 30:
                    continue
                    
                aspect_ratio = box_w / box_h if box_h > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                if box_w > w * 0.8 or box_h > h * 0.8:
                    continue
                
                boxes.append([x1, y1, box_w, box_h])
                confidences.append(confidence)
        
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
        
        if len(detected) > 1:
            detected = self._remove_overlapping_faces(detected)
        
        return detected
    
    def _remove_overlapping_faces(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        """Remove overlapping face detections, keeping highest confidence."""
        if not faces:
            return faces
        
        sorted_faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        
        kept = []
        for face in sorted_faces:
            x1, y1, w1, h1 = face.bbox
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            
            is_duplicate = False
            for kept_face in kept:
                x2, y2, w2, h2 = kept_face.bbox
                cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
                
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                avg_size = (w1 + h1 + w2 + h2) / 4
                
                if dist < avg_size * 0.6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(face)
        
        return kept
