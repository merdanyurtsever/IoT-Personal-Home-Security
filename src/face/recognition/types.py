"""Face recognition types."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .attributes import AttributeProfile


class FaceCategory(Enum):
    """Category of face for classification.
    
    - WATCH_LIST: People identified by photo (registered face)
    - THREAT_PROFILE: People matching attribute filters (glasses, beard, etc.)
    - NO_MATCH: Face detected but doesn't match any criteria
    """
    WATCH_LIST = "watch_list"
    THREAT_PROFILE = "threat_profile"
    NO_MATCH = "no_match"


@dataclass
class RecognitionResult:
    """Result of face recognition."""
    
    identity: str
    confidence: float
    category: FaceCategory = FaceCategory.NO_MATCH
    embedding: Optional[np.ndarray] = None
    face_id: Optional[int] = None
    attribute_profile: Optional["AttributeProfile"] = None
    matched_filter: Optional[str] = None
    
    @property
    def is_threat(self) -> bool:
        """Check if face matches any threat criteria."""
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
        """Check if this result should trigger an alert."""
        return self.category in (FaceCategory.WATCH_LIST, FaceCategory.THREAT_PROFILE)
    
    # Backwards compatibility
    @property
    def is_known(self) -> bool:
        return self.is_threat
    
    @property
    def is_attribute_match(self) -> bool:
        return self.is_threat_profile
