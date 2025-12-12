"""Face database for persistent storage of face embeddings."""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Persistent storage for face embeddings and metadata."""
    
    def __init__(self, database_path: Optional[str] = None):
        """Initialize face database.
        
        Args:
            database_path: Path to store database files
                          If None, uses in-memory storage only
        """
        self._database_path = Path(database_path) if database_path else None
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._metadata: Dict[str, dict] = {}
        
        if self._database_path:
            self._database_path.mkdir(parents=True, exist_ok=True)
            self._load()
    
    def add_face(
        self,
        name: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add a face embedding to the database.
        
        Args:
            name: Identity name
            embedding: Face embedding vector
            metadata: Optional metadata for this face
            
        Returns:
            True if added successfully
        """
        if name not in self._embeddings:
            self._embeddings[name] = []
            self._metadata[name] = metadata or {}
        
        self._embeddings[name].append(embedding.copy())
        
        if metadata:
            self._metadata[name].update(metadata)
        
        logger.debug(f"Added embedding for {name} ({len(self._embeddings[name])} total)")
        return True
    
    def get_embeddings(self, name: str) -> List[np.ndarray]:
        """Get all embeddings for an identity."""
        return self._embeddings.get(name, [])
    
    def get_metadata(self, name: str) -> dict:
        """Get metadata for an identity."""
        return self._metadata.get(name, {})
    
    def get_all_identities(self) -> List[str]:
        """Get list of all registered identity names."""
        return list(self._embeddings.keys())
    
    def get_sample_count(self, name: str) -> int:
        """Get number of samples for an identity."""
        return len(self._embeddings.get(name, []))
    
    def has_identity(self, name: str) -> bool:
        """Check if an identity exists in the database."""
        return name in self._embeddings
    
    def remove_identity(self, name: str) -> bool:
        """Remove an identity from the database.
        
        Args:
            name: Identity name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self._embeddings:
            del self._embeddings[name]
            self._metadata.pop(name, None)
            logger.info(f"Removed identity: {name}")
            return True
        return False
    
    def clear(self):
        """Clear all data from the database."""
        self._embeddings.clear()
        self._metadata.clear()
        logger.info("Database cleared")
    
    def find_best_match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.0,
    ) -> Tuple[Optional[str], float]:
        """Find the best matching identity for an embedding.
        
        Args:
            embedding: Embedding to match
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (identity_name, similarity) or (None, 0.0)
        """
        best_identity = None
        best_similarity = 0.0
        
        for name, embeddings in self._embeddings.items():
            for stored in embeddings:
                similarity = self._cosine_similarity(embedding, stored)
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_identity = name
        
        return best_identity, best_similarity
    
    def find_all_matches(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Find all matching identities above threshold.
        
        Args:
            embedding: Embedding to match
            threshold: Minimum similarity threshold
            
        Returns:
            List of (identity_name, similarity) tuples, sorted by similarity
        """
        matches = []
        
        for name, embeddings in self._embeddings.items():
            max_similarity = 0.0
            for stored in embeddings:
                similarity = self._cosine_similarity(embedding, stored)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                matches.append((name, max_similarity))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        a = a.flatten()
        b = b.flatten()
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def save(self):
        """Save database to disk."""
        if not self._database_path:
            logger.warning("No database path set, cannot save")
            return
        
        embeddings_file = self._database_path / "embeddings.pkl"
        with open(embeddings_file, "wb") as f:
            pickle.dump(self._embeddings, f)
        
        metadata_file = self._database_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2)
        
        logger.info(f"Database saved to {self._database_path}")
    
    def _load(self):
        """Load database from disk."""
        if not self._database_path:
            return
        
        embeddings_file = self._database_path / "embeddings.pkl"
        if embeddings_file.exists():
            try:
                with open(embeddings_file, "rb") as f:
                    self._embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self._embeddings)} identities from database")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
        
        metadata_file = self._database_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
    
    def export_to_npz(self, output_path: str):
        """Export embeddings to NPZ format.
        
        Args:
            output_path: Path for output file
        """
        data = {}
        for name, embeddings in self._embeddings.items():
            data[name] = np.array(embeddings)
        
        np.savez_compressed(output_path, **data)
        logger.info(f"Exported database to {output_path}")
    
    def import_from_npz(self, input_path: str, merge: bool = False):
        """Import embeddings from NPZ format.
        
        Args:
            input_path: Path to NPZ file
            merge: If True, merge with existing; if False, replace
        """
        if not merge:
            self._embeddings.clear()
            self._metadata.clear()
        
        data = np.load(input_path)
        for name in data.files:
            embeddings = data[name]
            if name not in self._embeddings:
                self._embeddings[name] = []
            self._embeddings[name].extend([e for e in embeddings])
        
        logger.info(f"Imported {len(data.files)} identities from {input_path}")
    
    def __len__(self) -> int:
        return len(self._embeddings)
    
    def __contains__(self, name: str) -> bool:
        return name in self._embeddings
