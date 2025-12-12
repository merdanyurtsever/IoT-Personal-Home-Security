"""Face processing pipeline for end-to-end detection and recognition."""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .detection import FaceDetector, DetectedFace
from .recognition import (
    FaceRecognizer,
    FaceCategory,
    RecognitionResult,
    AttributeFilterChain,
)
from .utils import crop_face

logger = logging.getLogger(__name__)


class DetectionMode(Enum):
    """Pipeline detection modes."""
    CONTINUOUS = "continuous"
    MOTION_TRIGGERED = "motion_triggered"
    SCHEDULED = "scheduled"


@dataclass
class FaceEvent:
    """Event generated when a face is detected/recognized."""
    
    timestamp: float
    face_id: int
    result: RecognitionResult
    bbox: Tuple[int, int, int, int]
    face_image: Optional[np.ndarray] = None
    frame_image: Optional[np.ndarray] = None
    
    @property
    def is_alert(self) -> bool:
        """Check if this event should trigger an alert."""
        return self.result.should_alert
    
    @property
    def identity(self) -> str:
        """Get the identity name."""
        return self.result.identity
    
    @property
    def category(self) -> FaceCategory:
        """Get the face category."""
        return self.result.category
    
    @property
    def confidence(self) -> float:
        """Get the confidence score."""
        return self.result.confidence
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "face_id": self.face_id,
            "identity": self.identity,
            "category": self.category.value,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "is_alert": self.is_alert,
        }


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    detection_backend: str = "opencv_dnn"
    embedding_backend: str = "opencv_dnn"
    similarity_threshold: float = 0.6
    detection_interval: float = 0.1
    min_face_size: int = 30
    detect_attributes: bool = False
    save_face_crops: bool = False
    mode: DetectionMode = DetectionMode.CONTINUOUS


class FaceSecurityPipeline:
    """End-to-end face detection and recognition pipeline."""
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        attribute_filters: Optional[AttributeFilterChain] = None,
        on_face_detected: Optional[Callable[[FaceEvent], None]] = None,
        on_alert: Optional[Callable[[FaceEvent], None]] = None,
    ):
        """Initialize the face security pipeline.
        
        Args:
            config: Pipeline configuration
            attribute_filters: Optional attribute filters for threat profiles
            on_face_detected: Callback for any face detection
            on_alert: Callback specifically for alert-worthy detections
        """
        self.config = config or PipelineConfig()
        
        self._detector = FaceDetector(backend=self.config.detection_backend)
        self._recognizer = FaceRecognizer(
            embedding_backend=self.config.embedding_backend,
            detection_backend=self.config.detection_backend,
            similarity_threshold=self.config.similarity_threshold,
            attribute_filters=attribute_filters,
        )
        
        self._on_face_detected = on_face_detected
        self._on_alert = on_alert
        
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._frame_queue: List[np.ndarray] = []
        self._queue_lock = threading.Lock()
        
        self._stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "alerts_triggered": 0,
            "start_time": None,
        }
    
    def register_face(
        self,
        name: str,
        image: np.ndarray,
        detect: bool = True,
    ) -> bool:
        """Register a face in the recognition database.
        
        Args:
            name: Identity name
            image: Image containing the face
            detect: If True, detect face first
            
        Returns:
            True if registration succeeded
        """
        return self._recognizer.register_face(name, image, detect=detect)
    
    def register_from_directory(self, directory: str) -> Dict[str, int]:
        """Register faces from a directory structure.
        
        Args:
            directory: Path to face database directory
            
        Returns:
            Dict mapping name to number of registered faces
        """
        return self._recognizer.register_from_directory(directory)
    
    def process_frame(
        self,
        frame: np.ndarray,
        save_crops: Optional[bool] = None,
    ) -> List[FaceEvent]:
        """Process a single frame for faces.
        
        Args:
            frame: Input frame (BGR)
            save_crops: Whether to save face crops in events
            
        Returns:
            List of face events
        """
        save_crops = save_crops if save_crops is not None else self.config.save_face_crops
        
        detected_faces = self._detector.detect(frame)
        
        detected_faces = [
            f for f in detected_faces
            if f.bbox[2] >= self.config.min_face_size and f.bbox[3] >= self.config.min_face_size
        ]
        
        events = []
        
        for face in detected_faces:
            x, y, w, h = face.bbox
            face_crop = crop_face(frame, face.bbox)
            
            result = self._recognizer.recognize_face(
                face_crop,
                detect_attributes=self.config.detect_attributes,
            )
            
            event = FaceEvent(
                timestamp=time.time(),
                face_id=result.face_id or 0,
                result=result,
                bbox=face.bbox,
                face_image=face_crop if save_crops else None,
                frame_image=frame if save_crops else None,
            )
            
            events.append(event)
            
            if self._on_face_detected:
                try:
                    self._on_face_detected(event)
                except Exception as e:
                    logger.error(f"Face detected callback error: {e}")
            
            if event.is_alert and self._on_alert:
                try:
                    self._on_alert(event)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
                self._stats["alerts_triggered"] += 1
        
        self._stats["frames_processed"] += 1
        self._stats["faces_detected"] += len(events)
        
        return events
    
    def start(self):
        """Start background processing."""
        if self._running:
            return
        
        self._running = True
        self._stats["start_time"] = time.time()
        
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
        )
        self._processing_thread.start()
        logger.info("Face security pipeline started")
    
    def stop(self):
        """Stop background processing."""
        self._running = False
        
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None
        
        logger.info("Face security pipeline stopped")
    
    def submit_frame(self, frame: np.ndarray):
        """Submit a frame for background processing.
        
        Args:
            frame: Input frame to process
        """
        with self._queue_lock:
            if len(self._frame_queue) < 10:
                self._frame_queue.append(frame.copy())
    
    def _processing_loop(self):
        """Background processing loop."""
        last_process_time = 0.0
        
        while self._running:
            with self._queue_lock:
                if self._frame_queue:
                    frame = self._frame_queue.pop(0)
                else:
                    frame = None
            
            if frame is not None:
                current_time = time.time()
                if current_time - last_process_time >= self.config.detection_interval:
                    self.process_frame(frame)
                    last_process_time = current_time
            else:
                time.sleep(0.01)
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        stats = self._stats.copy()
        
        if stats["start_time"]:
            runtime = time.time() - stats["start_time"]
            stats["runtime_seconds"] = runtime
            stats["fps"] = stats["frames_processed"] / runtime if runtime > 0 else 0
        
        return stats
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self._stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "alerts_triggered": 0,
            "start_time": time.time() if self._running else None,
        }
    
    def get_registered_identities(self) -> List[str]:
        """Get list of registered identity names."""
        return self._recognizer.get_registered_identities()
    
    def remove_identity(self, name: str) -> bool:
        """Remove an identity from the database."""
        return self._recognizer.remove_identity(name)
    
    def clear_database(self):
        """Clear all registered faces."""
        self._recognizer.clear_database()
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
