"""Face attribute detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


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
        """Detect attributes in a face image."""
        pass


class HaarAttributeDetector(BaseAttributeDetector):
    """Simple attribute detector using Haar cascades and color analysis."""
    
    def __init__(self):
        self._cascades = {}
        self._load_cascades()
    
    def _load_cascades(self):
        """Load available Haar cascades for attribute detection."""
        cascade_dir: str = cv2.data.haarcascades  # type: ignore
        
        eye_cascade = f"{cascade_dir}haarcascade_eye.xml"
        glasses_cascade = f"{cascade_dir}haarcascade_eye_tree_eyeglasses.xml"
        
        if Path(eye_cascade).exists():
            self._cascades["eyes"] = cv2.CascadeClassifier(eye_cascade)
        if Path(glasses_cascade).exists():
            self._cascades["glasses"] = cv2.CascadeClassifier(glasses_cascade)
    
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
        
        # Beard detection
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
        
        eye_region = gray[int(h*0.2):int(h*0.5), :]
        
        glasses = self._cascades["glasses"].detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=3
        )
        
        if len(glasses) >= 1:
            return True, min(0.5 + len(glasses) * 0.15, 0.95)
        
        edges = cv2.Canny(eye_region, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        if edge_ratio > 0.15:
            return True, 0.6
        
        return False, 0.3
    
    def _detect_beard(self, face_image: np.ndarray, h: int, w: int) -> Tuple[bool, float]:
        """Detect beard by analyzing lower face texture."""
        lower_face = face_image[int(h*0.65):, int(w*0.2):int(w*0.8)]
        
        if lower_face.size == 0:
            return False, 0.0
        
        gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        texture = np.std(gray)
        
        if texture > 40:
            return True, min(0.5 + (texture - 40) / 60, 0.9)
        
        return False, max(0.1, 0.5 - texture / 80)
    
    def _detect_hair_color(self, face_image: np.ndarray, h: int, w: int) -> Tuple[FaceAttribute, float]:
        """Detect hair color from upper face region."""
        hair_region = face_image[:int(h*0.25), int(w*0.2):int(w*0.8)]
        
        if hair_region.size == 0:
            return FaceAttribute.BROWN_HAIR, 0.3
        
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        
        avg_h = np.mean(hsv[:, :, 0])
        avg_s = np.mean(hsv[:, :, 1])
        avg_v = np.mean(hsv[:, :, 2])
        
        if avg_v < 50 and avg_s < 50:
            return FaceAttribute.BLACK_HAIR, 0.7
        elif avg_v > 180 and avg_s < 30:
            return FaceAttribute.GRAY_HAIR, 0.65
        elif avg_h < 25 and avg_s > 100 and avg_v > 150:
            return FaceAttribute.BLONDE_HAIR, 0.6
        else:
            return FaceAttribute.BROWN_HAIR, 0.55


class AttributeFilter:
    """Composable filter for face attributes."""
    
    def __init__(
        self,
        required_attributes: Optional[List[FaceAttribute]] = None,
        min_confidence: float = 0.5,
        match_all: bool = True,
    ):
        self.required: List[FaceAttribute] = required_attributes or []
        self.min_confidence = min_confidence
        self.match_all = match_all
    
    def require(self, attribute: FaceAttribute) -> "AttributeFilter":
        """Add a required attribute. Returns self for chaining."""
        if attribute not in self.required:
            self.required.append(attribute)
        return self
    
    def and_require(self, attribute: FaceAttribute) -> "AttributeFilter":
        """Create new filter with additional requirement."""
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
    """Chain of attribute filters for complex matching scenarios."""
    
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
        """Find all matching profiles with scores."""
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
