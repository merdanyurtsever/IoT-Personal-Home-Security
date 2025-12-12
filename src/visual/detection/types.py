"""Detected face data type."""

from dataclasses import dataclass
from typing import Optional, Tuple


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
