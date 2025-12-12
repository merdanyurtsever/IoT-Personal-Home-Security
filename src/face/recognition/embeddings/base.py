"""Base class for face embedding backends."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseEmbeddingBackend(ABC):
    """Abstract base class for face embedding extraction."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the embedding backend."""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vector."""
        pass
    
    @property
    def embedding_size(self) -> int:
        """Alias for embedding_dim for backward compatibility."""
        return self.embedding_dim
    
    @abstractmethod
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from a face image.
        
        Args:
            face_image: BGR face image (cropped and aligned)
            
        Returns:
            Embedding vector as numpy array, or None if extraction failed
        """
        pass
    
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two embeddings and return similarity score.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (higher is more similar)
        """
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(cosine_sim)
    
    def distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Euclidean distance (lower means more similar)
        """
        return float(np.linalg.norm(embedding1 - embedding2))
