"""Face database module.

This module provides SQLite-based persistence for face metadata,
embeddings, and processing status. Designed for thread-safe access
and efficient retrieval of face data for recognition.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .face import FaceRecognizer

logger = logging.getLogger(__name__)


class FaceStatus(Enum):
    """Processing status for a face entry."""
    PENDING = "pending"       # Uploaded, waiting for processing
    PROCESSING = "processing" # Currently extracting embedding
    READY = "ready"          # Embedding extracted, ready for matching
    FAILED = "failed"        # Processing failed
    

class FaceType(Enum):
    """Type of face for security classification.
    
    All registered faces are people you want to be ALERTED about.
    This is a threat detection system, not a known/unknown system.
    """
    WATCH_LIST = "watch_list"  # Person on watch list (trigger alert when detected)


@dataclass
class FaceRecord:
    """Represents a face record in the database."""
    
    id: Optional[int]
    person_name: str
    face_type: FaceType
    status: FaceStatus
    image_path: str
    embedding: Optional[np.ndarray] = None
    embedding_size: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "person_name": self.person_name,
            "face_type": self.face_type.value,
            "status": self.status.value,
            "image_path": self.image_path,
            "embedding_size": self.embedding_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
        }


@dataclass
class ProcessingJob:
    """Represents a face processing job."""
    
    id: str
    face_ids: List[int]
    status: str  # "pending", "processing", "completed", "failed"
    total_faces: int
    processed_faces: int
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "face_ids": self.face_ids,
            "status": self.status,
            "total_faces": self.total_faces,
            "processed_faces": self.processed_faces,
            "progress": self.processed_faces / self.total_faces if self.total_faces > 0 else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


class FaceDatabase:
    """SQLite-based face database with thread-safe access.
    
    Stores face metadata, embeddings, and processing status.
    Designed for integration with FaceRecognizer and API endpoints.
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize face database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.RLock()
        
        # Initialize database schema
        self._init_schema()
        
        logger.info(f"Initialized face database at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection
    
    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _init_schema(self):
        """Initialize database schema."""
        with self._transaction() as cursor:
            # Faces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    face_type TEXT NOT NULL DEFAULT 'watch_list',
                    status TEXT NOT NULL DEFAULT 'pending',
                    image_path TEXT NOT NULL,
                    embedding BLOB,
                    embedding_size INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT
                )
            """)
            
            # Processing jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id TEXT PRIMARY KEY,
                    face_ids TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total_faces INTEGER NOT NULL,
                    processed_faces INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT
                )
            """)
            
            # Indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_faces_person_name 
                ON faces(person_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_faces_status 
                ON faces(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_faces_face_type 
                ON faces(face_type)
            """)
            
            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO schema_version (version) VALUES (?)
            """, (self.SCHEMA_VERSION,))
    
    # -------------------------------------------------------------------------
    # Face CRUD Operations
    # -------------------------------------------------------------------------
    
    def add_face(
        self,
        person_name: str,
        image_path: str,
        face_type: FaceType = FaceType.WATCH_LIST,
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        """Add a new face to the database.
        
        Args:
            person_name: Name/identifier for the person
            image_path: Path to the face image
            face_type: Type of face (watch_list)
            embedding: Pre-computed embedding (optional)
            
        Returns:
            ID of the inserted face record
        """
        status = FaceStatus.READY if embedding is not None else FaceStatus.PENDING
        embedding_blob = self._serialize_embedding(embedding) if embedding is not None else None
        embedding_size = len(embedding) if embedding is not None else 0
        
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("""
                    INSERT INTO faces (
                        person_name, face_type, status, image_path, 
                        embedding, embedding_size
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    person_name,
                    face_type.value,
                    status.value,
                    image_path,
                    embedding_blob,
                    embedding_size,
                ))
                face_id = cursor.lastrowid
        
        logger.info(f"Added face {face_id} for {person_name} (type: {face_type.value})")
        return face_id
    
    def get_face(self, face_id: int) -> Optional[FaceRecord]:
        """Get a face record by ID.
        
        Args:
            face_id: Face record ID
            
        Returns:
            FaceRecord or None if not found
        """
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM faces WHERE id = ?", (face_id,))
            row = cursor.fetchone()
            
        if row is None:
            return None
        
        return self._row_to_face_record(row)
    
    def get_faces_by_person(self, person_name: str) -> List[FaceRecord]:
        """Get all faces for a person.
        
        Args:
            person_name: Person name/identifier
            
        Returns:
            List of FaceRecords
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM faces WHERE person_name = ? ORDER BY created_at",
                (person_name,)
            )
            rows = cursor.fetchall()
        
        return [self._row_to_face_record(row) for row in rows]
    
    def get_faces_by_status(self, status: FaceStatus) -> List[FaceRecord]:
        """Get all faces with a specific status.
        
        Args:
            status: Face status to filter by
            
        Returns:
            List of FaceRecords
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM faces WHERE status = ? ORDER BY created_at",
                (status.value,)
            )
            rows = cursor.fetchall()
        
        return [self._row_to_face_record(row) for row in rows]
    
    def get_faces_by_type(self, face_type: FaceType) -> List[FaceRecord]:
        """Get all faces of a specific type.
        
        Args:
            face_type: Face type to filter by
            
        Returns:
            List of FaceRecords
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM faces WHERE face_type = ? ORDER BY person_name, created_at",
                (face_type.value,)
            )
            rows = cursor.fetchall()
        
        return [self._row_to_face_record(row) for row in rows]
    
    def get_all_faces(
        self, 
        include_embeddings: bool = False,
        ready_only: bool = False,
    ) -> List[FaceRecord]:
        """Get all faces from the database.
        
        Args:
            include_embeddings: Whether to include embedding data
            ready_only: Only return faces with READY status
            
        Returns:
            List of FaceRecords
        """
        query = "SELECT * FROM faces"
        if ready_only:
            query += " WHERE status = 'ready'"
        query += " ORDER BY person_name, created_at"
        
        with self._transaction() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        
        records = []
        for row in rows:
            record = self._row_to_face_record(row, include_embedding=include_embeddings)
            records.append(record)
        
        return records
    
    def get_ready_embeddings(self) -> Dict[str, List[Tuple[np.ndarray, str, int]]]:
        """Get all ready embeddings grouped by person name.
        
        Returns:
            Dict mapping person_name to list of (embedding, face_type, face_id)
        """
        with self._transaction() as cursor:
            cursor.execute("""
                SELECT id, person_name, face_type, embedding 
                FROM faces 
                WHERE status = 'ready' AND embedding IS NOT NULL
                ORDER BY person_name
            """)
            rows = cursor.fetchall()
        
        embeddings: Dict[str, List[Tuple[np.ndarray, str, int]]] = {}
        
        for row in rows:
            person_name = row["person_name"]
            embedding = self._deserialize_embedding(row["embedding"])
            face_type = row["face_type"]
            face_id = row["id"]
            
            if embedding is not None:
                if person_name not in embeddings:
                    embeddings[person_name] = []
                embeddings[person_name].append((embedding, face_type, face_id))
        
        return embeddings
    
    def update_face_embedding(
        self,
        face_id: int,
        embedding: np.ndarray,
        status: FaceStatus = FaceStatus.READY,
    ) -> bool:
        """Update the embedding for a face.
        
        Args:
            face_id: Face record ID
            embedding: Computed face embedding
            status: New status (default: READY)
            
        Returns:
            True if updated, False if not found
        """
        embedding_blob = self._serialize_embedding(embedding)
        
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("""
                    UPDATE faces 
                    SET embedding = ?, embedding_size = ?, status = ?, 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (embedding_blob, len(embedding), status.value, face_id))
                updated = cursor.rowcount > 0
        
        if updated:
            logger.debug(f"Updated embedding for face {face_id}")
        
        return updated
    
    def update_face_status(
        self,
        face_id: int,
        status: FaceStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update the status of a face.
        
        Args:
            face_id: Face record ID
            status: New status
            error_message: Error message (for FAILED status)
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("""
                    UPDATE faces 
                    SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status.value, error_message, face_id))
                updated = cursor.rowcount > 0
        
        return updated
    
    def delete_face(self, face_id: int) -> bool:
        """Delete a face from the database.
        
        Args:
            face_id: Face record ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
                deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"Deleted face {face_id}")
        
        return deleted
    
    def delete_faces_by_person(self, person_name: str) -> int:
        """Delete all faces for a person.
        
        Args:
            person_name: Person name/identifier
            
        Returns:
            Number of faces deleted
        """
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute(
                    "DELETE FROM faces WHERE person_name = ?",
                    (person_name,)
                )
                deleted = cursor.rowcount
        
        if deleted > 0:
            logger.info(f"Deleted {deleted} faces for {person_name}")
        
        return deleted
    
    # -------------------------------------------------------------------------
    # Processing Job Operations
    # -------------------------------------------------------------------------
    
    def create_job(self, job_id: str, face_ids: List[int]) -> ProcessingJob:
        """Create a new processing job.
        
        Args:
            job_id: Unique job identifier
            face_ids: List of face IDs to process
            
        Returns:
            Created ProcessingJob
        """
        face_ids_json = json.dumps(face_ids)
        
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("""
                    INSERT INTO processing_jobs (
                        id, face_ids, status, total_faces, processed_faces
                    )
                    VALUES (?, ?, 'pending', ?, 0)
                """, (job_id, face_ids_json, len(face_ids)))
        
        logger.info(f"Created processing job {job_id} with {len(face_ids)} faces")
        
        return ProcessingJob(
            id=job_id,
            face_ids=face_ids,
            status="pending",
            total_faces=len(face_ids),
            processed_faces=0,
            created_at=datetime.now(),
        )
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a processing job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ProcessingJob or None if not found
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM processing_jobs WHERE id = ?",
                (job_id,)
            )
            row = cursor.fetchone()
        
        if row is None:
            return None
        
        return ProcessingJob(
            id=row["id"],
            face_ids=json.loads(row["face_ids"]),
            status=row["status"],
            total_faces=row["total_faces"],
            processed_faces=row["processed_faces"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
        )
    
    def update_job_progress(
        self,
        job_id: str,
        processed_faces: int,
        status: Optional[str] = None,
    ) -> bool:
        """Update job progress.
        
        Args:
            job_id: Job identifier
            processed_faces: Number of faces processed
            status: New status (optional)
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            with self._transaction() as cursor:
                if status:
                    cursor.execute("""
                        UPDATE processing_jobs 
                        SET processed_faces = ?, status = ?
                        WHERE id = ?
                    """, (processed_faces, status, job_id))
                else:
                    cursor.execute("""
                        UPDATE processing_jobs 
                        SET processed_faces = ?
                        WHERE id = ?
                    """, (processed_faces, job_id))
                
                return cursor.rowcount > 0
    
    def complete_job(
        self,
        job_id: str,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> bool:
        """Mark a job as completed.
        
        Args:
            job_id: Job identifier
            status: Final status (completed/failed)
            error_message: Error message for failed jobs
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("""
                    UPDATE processing_jobs 
                    SET status = ?, completed_at = CURRENT_TIMESTAMP, 
                        error_message = ?
                    WHERE id = ?
                """, (status, error_message, job_id))
                return cursor.rowcount > 0
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> dict:
        """Get database statistics.
        
        Returns:
            Dictionary with counts and stats
        """
        with self._transaction() as cursor:
            cursor.execute("SELECT COUNT(*) as total FROM faces")
            total_faces = cursor.fetchone()["total"]
            
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM faces GROUP BY status
            """)
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT face_type, COUNT(*) as count 
                FROM faces GROUP BY face_type
            """)
            type_counts = {row["face_type"]: row["count"] for row in cursor.fetchall()}
            
            cursor.execute("SELECT COUNT(DISTINCT person_name) as count FROM faces")
            unique_persons = cursor.fetchone()["count"]
        
        return {
            "total_faces": total_faces,
            "unique_persons": unique_persons,
            "by_status": status_counts,
            "by_type": type_counts,
        }
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy embedding to bytes for storage."""
        return embedding.astype(np.float32).tobytes()
    
    def _deserialize_embedding(self, data: Optional[bytes]) -> Optional[np.ndarray]:
        """Deserialize bytes to numpy embedding."""
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32)
    
    def _row_to_face_record(
        self, 
        row: sqlite3.Row, 
        include_embedding: bool = True
    ) -> FaceRecord:
        """Convert database row to FaceRecord."""
        embedding = None
        if include_embedding and row["embedding"]:
            embedding = self._deserialize_embedding(row["embedding"])
        
        return FaceRecord(
            id=row["id"],
            person_name=row["person_name"],
            face_type=FaceType(row["face_type"]),
            status=FaceStatus(row["status"]),
            image_path=row["image_path"],
            embedding=embedding,
            embedding_size=row["embedding_size"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error_message=row["error_message"],
        )
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# =============================================================================
# PROCESSING QUEUE (from processing.py)
# =============================================================================

"""Face processing queue module.

This module provides a background worker for processing face images,
extracting embeddings, and notifying clients of progress via callbacks.
Designed for integration with the API layer for async processing.
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class ProcessingEventType(Enum):
    """Types of processing events for callbacks."""
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    FACE_PROCESSING = "face_processing"
    FACE_COMPLETED = "face_completed"
    FACE_FAILED = "face_failed"


@dataclass
class ProcessingEvent:
    """Event data for processing callbacks."""
    
    event_type: ProcessingEventType
    job_id: str
    face_id: Optional[int] = None
    progress: float = 0.0
    total_faces: int = 0
    processed_faces: int = 0
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "job_id": self.job_id,
            "face_id": self.face_id,
            "progress": self.progress,
            "total_faces": self.total_faces,
            "processed_faces": self.processed_faces,
            "message": self.message,
            "error": self.error,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# Type alias for event callback function
EventCallback = Callable[[ProcessingEvent], None]


class FaceProcessingQueue:
    """Background worker for processing face images and extracting embeddings.
    
    Supports:
    - Queueing multiple processing jobs
    - Progress tracking with callbacks
    - Thread-safe operation
    - Graceful shutdown
    """
    
    def __init__(
        self,
        database: FaceDatabase,
        recognizer: "FaceRecognizer",
        max_workers: int = 1,  # Single worker for RPi resource constraints
        auto_start: bool = True,
    ):
        """Initialize processing queue.
        
        Args:
            database: Face database instance
            recognizer: Face recognizer for embedding extraction
            max_workers: Number of worker threads (default: 1 for RPi)
            auto_start: Whether to start workers immediately
        """
        self.database = database
        self.recognizer = recognizer
        self.max_workers = max_workers
        
        # Job queue
        self._job_queue: queue.Queue[str] = queue.Queue()
        
        # Active jobs tracking
        self._active_jobs: Dict[str, ProcessingJob] = {}
        self._jobs_lock = threading.Lock()
        
        # Event callbacks (job_id -> list of callbacks)
        self._callbacks: Dict[str, List[EventCallback]] = {}
        self._global_callbacks: List[EventCallback] = []
        self._callbacks_lock = threading.Lock()
        
        # Worker threads
        self._workers: List[threading.Thread] = []
        self._running = False
        self._shutdown_event = threading.Event()
        
        if auto_start:
            self.start()
    
    def start(self):
        """Start worker threads."""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"FaceProcessingWorker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self.max_workers} face processing worker(s)")
    
    def stop(self, timeout: float = 5.0):
        """Stop worker threads gracefully.
        
        Args:
            timeout: Maximum time to wait for workers to finish
        """
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=timeout)
        
        self._workers.clear()
        logger.info("Stopped face processing workers")
    
    def submit_job(
        self,
        face_ids: Optional[List[int]] = None,
        callback: Optional[EventCallback] = None,
    ) -> str:
        """Submit a new processing job.
        
        Args:
            face_ids: List of face IDs to process. If None, processes all pending faces.
            callback: Optional callback for job events
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())[:8]
        
        # If no face_ids specified, get all pending faces
        if face_ids is None:
            pending_faces = self.database.get_faces_by_status(FaceStatus.PENDING)
            face_ids = [face.id for face in pending_faces if face.id is not None]
        
        if not face_ids:
            logger.warning(f"No faces to process for job {job_id}")
            return job_id
        
        # Create job in database
        job = self.database.create_job(job_id, face_ids)
        
        with self._jobs_lock:
            self._active_jobs[job_id] = job
        
        # Register callback if provided
        if callback:
            self.add_callback(job_id, callback)
        
        # Queue the job
        self._job_queue.put(job_id)
        
        logger.info(f"Submitted job {job_id} with {len(face_ids)} faces")
        
        return job_id
    
    def add_callback(self, job_id: str, callback: EventCallback):
        """Add a callback for job events.
        
        Args:
            job_id: Job ID to listen for, or "*" for all jobs
            callback: Callback function
        """
        with self._callbacks_lock:
            if job_id == "*":
                self._global_callbacks.append(callback)
            else:
                if job_id not in self._callbacks:
                    self._callbacks[job_id] = []
                self._callbacks[job_id].append(callback)
    
    def remove_callback(self, job_id: str, callback: EventCallback):
        """Remove a callback.
        
        Args:
            job_id: Job ID or "*" for global callbacks
            callback: Callback to remove
        """
        with self._callbacks_lock:
            if job_id == "*":
                if callback in self._global_callbacks:
                    self._global_callbacks.remove(callback)
            elif job_id in self._callbacks:
                if callback in self._callbacks[job_id]:
                    self._callbacks[job_id].remove(callback)
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get current status of a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            ProcessingJob or None if not found
        """
        # Check active jobs first
        with self._jobs_lock:
            if job_id in self._active_jobs:
                return self._active_jobs[job_id]
        
        # Check database
        return self.database.get_job(job_id)
    
    def get_pending_count(self) -> int:
        """Get number of pending faces to process."""
        return len(self.database.get_faces_by_status(FaceStatus.PENDING))
    
    def get_queue_size(self) -> int:
        """Get number of jobs in queue."""
        return self._job_queue.qsize()
    
    def _worker_loop(self):
        """Main worker loop for processing jobs."""
        logger.debug(f"Worker {threading.current_thread().name} started")
        
        while self._running:
            try:
                # Get next job from queue with timeout
                try:
                    job_id = self._job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the job
                self._process_job(job_id)
                
                # Mark job as done in queue
                self._job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
        
        logger.debug(f"Worker {threading.current_thread().name} stopped")
    
    def _process_job(self, job_id: str):
        """Process a single job.
        
        Args:
            job_id: Job ID to process
        """
        job = self.database.get_job(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found in database")
            return
        
        # Update job status
        self.database.update_job_progress(job_id, 0, status="processing")
        
        # Emit job started event
        self._emit_event(ProcessingEvent(
            event_type=ProcessingEventType.JOB_STARTED,
            job_id=job_id,
            total_faces=job.total_faces,
            processed_faces=0,
            progress=0.0,
            message=f"Starting processing of {job.total_faces} faces",
        ))
        
        # Process each face
        processed = 0
        failed = 0
        
        for face_id in job.face_ids:
            if self._shutdown_event.is_set():
                logger.info(f"Job {job_id} interrupted by shutdown")
                break
            
            # Emit face processing event
            self._emit_event(ProcessingEvent(
                event_type=ProcessingEventType.FACE_PROCESSING,
                job_id=job_id,
                face_id=face_id,
                total_faces=job.total_faces,
                processed_faces=processed,
                progress=processed / job.total_faces,
                message=f"Processing face {face_id}",
            ))
            
            # Update face status to processing
            self.database.update_face_status(face_id, FaceStatus.PROCESSING)
            
            # Process the face
            success = self._process_face(face_id, job_id)
            
            if success:
                processed += 1
                self._emit_event(ProcessingEvent(
                    event_type=ProcessingEventType.FACE_COMPLETED,
                    job_id=job_id,
                    face_id=face_id,
                    total_faces=job.total_faces,
                    processed_faces=processed,
                    progress=processed / job.total_faces,
                    message=f"Face {face_id} processed successfully",
                ))
            else:
                failed += 1
                self._emit_event(ProcessingEvent(
                    event_type=ProcessingEventType.FACE_FAILED,
                    job_id=job_id,
                    face_id=face_id,
                    total_faces=job.total_faces,
                    processed_faces=processed,
                    progress=processed / job.total_faces,
                    error=f"Failed to process face {face_id}",
                ))
            
            # Update job progress
            self.database.update_job_progress(job_id, processed)
            
            # Emit progress event
            self._emit_event(ProcessingEvent(
                event_type=ProcessingEventType.JOB_PROGRESS,
                job_id=job_id,
                total_faces=job.total_faces,
                processed_faces=processed,
                progress=processed / job.total_faces,
                message=f"Processed {processed}/{job.total_faces} faces",
            ))
        
        # Complete the job
        final_status = "completed" if failed == 0 else "completed_with_errors"
        error_msg = f"{failed} faces failed" if failed > 0 else None
        
        self.database.complete_job(job_id, status=final_status, error_message=error_msg)
        
        # Remove from active jobs
        with self._jobs_lock:
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
        
        # Emit completion event
        event_type = ProcessingEventType.JOB_COMPLETED if failed == 0 else ProcessingEventType.JOB_FAILED
        self._emit_event(ProcessingEvent(
            event_type=event_type,
            job_id=job_id,
            total_faces=job.total_faces,
            processed_faces=processed,
            progress=1.0,
            message=f"Job completed: {processed} processed, {failed} failed",
            error=error_msg,
        ))
        
        # Clean up callbacks for this job
        with self._callbacks_lock:
            if job_id in self._callbacks:
                del self._callbacks[job_id]
        
        logger.info(f"Job {job_id} completed: {processed} processed, {failed} failed")
    
    def _process_face(self, face_id: int, job_id: str) -> bool:
        """Process a single face - load image and extract embedding.
        
        Args:
            face_id: Face ID to process
            job_id: Parent job ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get face record
            face = self.database.get_face(face_id)
            if face is None:
                logger.error(f"Face {face_id} not found")
                return False
            
            # Load image
            image_path = Path(face.image_path)
            if not image_path.exists():
                error_msg = f"Image not found: {image_path}"
                logger.error(error_msg)
                self.database.update_face_status(face_id, FaceStatus.FAILED, error_msg)
                return False
            
            image = cv2.imread(str(image_path))
            if image is None:
                error_msg = f"Failed to load image: {image_path}"
                logger.error(error_msg)
                self.database.update_face_status(face_id, FaceStatus.FAILED, error_msg)
                return False
            
            # Extract embedding
            try:
                embedding = self.recognizer.get_embedding(image)
            except Exception as e:
                error_msg = f"Embedding extraction failed: {e}"
                logger.error(error_msg)
                self.database.update_face_status(face_id, FaceStatus.FAILED, error_msg)
                return False
            
            # Update database with embedding
            self.database.update_face_embedding(face_id, embedding, FaceStatus.READY)
            
            logger.debug(f"Processed face {face_id}: embedding size {len(embedding)}")
            return True
            
        except Exception as e:
            error_msg = f"Unexpected error processing face {face_id}: {e}"
            logger.error(error_msg, exc_info=True)
            self.database.update_face_status(face_id, FaceStatus.FAILED, error_msg)
            return False
    
    def _emit_event(self, event: ProcessingEvent):
        """Emit an event to registered callbacks.
        
        Args:
            event: Processing event to emit
        """
        with self._callbacks_lock:
            # Job-specific callbacks
            callbacks = self._callbacks.get(event.job_id, []).copy()
            # Global callbacks
            callbacks.extend(self._global_callbacks.copy())
        
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}", exc_info=True)


# =============================================================================
# SERVICE (from service.py)
# =============================================================================

"""Face recognizer service module.

This module provides a thread-safe service layer that integrates
FaceRecognizer, FaceDatabase, and FaceProcessingQueue for unified
face management operations.
"""

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .face import (
    FaceRecognizer,
    FaceCategory,
    RecognitionResult,
    EMBEDDING_BACKENDS,
)

logger = logging.getLogger(__name__)


class FaceRecognizerService:
    """Thread-safe service for face recognition with database persistence.
    
    Integrates:
    - FaceRecognizer: Embedding extraction and matching
    - FaceDatabase: SQLite persistence for faces and embeddings
    - FaceProcessingQueue: Background processing with status callbacks
    
    Provides a unified API for:
    - Adding/removing faces
    - Processing face embeddings
    - Real-time face recognition
    - Hot-reload of embeddings when processing completes
    """
    
    def __init__(
        self,
        database_path: Union[str, Path],
        upload_dir: Union[str, Path],
        embedding_backend: str = "dlib",
        similarity_threshold: float = 0.6,
        tflite_model_path: Optional[Union[str, Path]] = None,
        auto_load_embeddings: bool = True,
        auto_start_processing: bool = True,
    ):
        """Initialize face recognizer service.
        
        Args:
            database_path: Path to SQLite database file
            upload_dir: Directory for uploaded face images
            embedding_backend: Backend for embedding extraction (dlib, tflite)
            similarity_threshold: Threshold for face matching
            tflite_model_path: Path to TFLite model (required for tflite backend)
            auto_load_embeddings: Load existing embeddings on startup
            auto_start_processing: Start processing queue on startup
        """
        self.database_path = Path(database_path)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize components
        self.database = FaceDatabase(self.database_path)
        
        self.recognizer = FaceRecognizer(
            model=embedding_backend,
            threshold=similarity_threshold,
            tflite_model_path=tflite_model_path,
        )
        
        self.processing_queue = FaceProcessingQueue(
            database=self.database,
            recognizer=self.recognizer,
            auto_start=auto_start_processing,
        )
        
        # Register callback for hot-reload when processing completes
        self.processing_queue.add_callback("*", self._on_processing_event)
        
        # Load existing embeddings
        if auto_load_embeddings:
            self._load_embeddings_from_database()
        
        logger.info(
            f"FaceRecognizerService initialized: "
            f"backend={embedding_backend}, threshold={similarity_threshold}"
        )
    
    def _load_embeddings_from_database(self):
        """Load all ready embeddings from database into recognizer."""
        with self._lock:
            self.recognizer.clear_embeddings()
            
            embeddings = self.database.get_ready_embeddings()
            count = 0
            
            for person_name, face_entries in embeddings.items():
                for embedding, face_type, face_id in face_entries:
                    # All registered faces are on watch list
                    self.recognizer.enroll_embedding(
                        name=person_name,
                        embedding=embedding,
                        category=FaceCategory.WATCH_LIST,
                        face_id=face_id,
                    )
                    count += 1
            
            logger.info(f"Loaded {count} embeddings from database")
    
    def _on_processing_event(self, event: ProcessingEvent):
        """Handle processing events for hot-reload."""
        # from .processing import ProcessingEventType
        
        if event.event_type == ProcessingEventType.FACE_COMPLETED:
            # Reload this specific face's embedding
            face = self.database.get_face(event.face_id)
            if face and face.embedding is not None:
                category = (
                    FaceCategory.WATCH_LIST 
                    if face.face_type == FaceType.WATCH_LIST 
                    else FaceCategory.WATCH_LIST
                )
                with self._lock:
                    self.recognizer.enroll_embedding(
                        name=face.person_name,
                        embedding=face.embedding,
                        category=category,
                        face_id=face.id,
                    )
                logger.debug(f"Hot-loaded embedding for face {face.id}")
        
        elif event.event_type == ProcessingEventType.JOB_COMPLETED:
            # Full reload after job completes
            logger.info(f"Job {event.job_id} completed, reloading all embeddings")
            self._load_embeddings_from_database()
    
    # -------------------------------------------------------------------------
    # Face Management
    # -------------------------------------------------------------------------
    
    def add_face(
        self,
        person_name: str,
        image_path: str,
        face_type: str = "watch_list",
    ) -> int:
        """Add a new face to the watch list.
        
        Args:
            person_name: Name/identifier for the person to watch
            image_path: Path to the face image
            face_type: Deprecated, always uses WATCH_LIST
            
        Returns:
            ID of the created face record
        """
        # All faces are on watch list (people we want alerts for)
        ft = FaceType.WATCH_LIST
        
        with self._lock:
            face_id = self.database.add_face(
                person_name=person_name,
                image_path=image_path,
                face_type=ft,
            )
        
        return face_id
    
    def get_face(self, face_id: int) -> Optional[FaceRecord]:
        """Get a face by ID.
        
        Args:
            face_id: Face record ID
            
        Returns:
            FaceRecord or None if not found
        """
        return self.database.get_face(face_id)
    
    def list_faces(
        self,
        person_name: Optional[str] = None,
        face_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[FaceRecord]:
        """List faces with optional filtering.
        
        Args:
            person_name: Filter by person name
            face_type: Filter by face type
            status: Filter by status
            
        Returns:
            List of FaceRecords
        """
        if person_name:
            faces = self.database.get_faces_by_person(person_name)
        elif face_type:
            # All faces are WATCH_LIST, but keep filter for API compatibility
            ft = FaceType.WATCH_LIST
            faces = self.database.get_faces_by_type(ft)
        elif status:
            fs = FaceStatus(status)
            faces = self.database.get_faces_by_status(fs)
        else:
            faces = self.database.get_all_faces(include_embeddings=False)
        
        return faces
    
    def delete_face(self, face_id: int, delete_image: bool = True) -> bool:
        """Delete a face by ID.
        
        Args:
            face_id: Face record ID
            delete_image: Also delete the image file
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            # Get face record first to get image path
            face = self.database.get_face(face_id)
            if face is None:
                return False
            
            # Remove from recognizer
            self.recognizer.remove_face(face.person_name, face_id)
            
            # Delete from database
            success = self.database.delete_face(face_id)
            
            # Delete image file
            if success and delete_image:
                try:
                    image_path = Path(face.image_path)
                    if image_path.exists():
                        image_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete image {face.image_path}: {e}")
        
        return success
    
    def delete_faces_by_person(
        self, 
        person_name: str, 
        delete_images: bool = True
    ) -> int:
        """Delete all faces for a person.
        
        Args:
            person_name: Person name/identifier
            delete_images: Also delete image files
            
        Returns:
            Number of faces deleted
        """
        with self._lock:
            # Get faces first to get image paths
            faces = self.database.get_faces_by_person(person_name)
            
            # Remove from recognizer
            self.recognizer.remove_face(person_name)
            
            # Delete from database
            count = self.database.delete_faces_by_person(person_name)
            
            # Delete image files
            if delete_images:
                for face in faces:
                    try:
                        image_path = Path(face.image_path)
                        if image_path.exists():
                            image_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete image {face.image_path}: {e}")
        
        return count
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def start_processing(
        self,
        face_ids: Optional[List[int]] = None,
        callback: Optional[EventCallback] = None,
    ) -> Tuple[str, int]:
        """Start processing faces (extract embeddings).
        
        Args:
            face_ids: List of face IDs to process. If None, processes all pending.
            callback: Optional callback for processing events
            
        Returns:
            Tuple of (job_id, total_faces)
        """
        # Get pending faces if not specified
        if face_ids is None:
            pending = self.database.get_faces_by_status(FaceStatus.PENDING)
            face_ids = [f.id for f in pending if f.id is not None]
        
        if not face_ids:
            return ("", 0)
        
        job_id = self.processing_queue.submit_job(face_ids, callback)
        
        return (job_id, len(face_ids))
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get processing job status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ProcessingJob or None if not found
        """
        return self.processing_queue.get_job_status(job_id)
    
    def add_processing_callback(self, job_id: str, callback: EventCallback):
        """Add a callback for processing events.
        
        Args:
            job_id: Job ID to listen for, or "*" for all jobs
            callback: Callback function
        """
        self.processing_queue.add_callback(job_id, callback)
    
    def remove_processing_callback(self, job_id: str, callback: EventCallback):
        """Remove a processing callback.
        
        Args:
            job_id: Job ID or "*" for global callbacks
            callback: Callback to remove
        """
        self.processing_queue.remove_callback(job_id, callback)
    
    # -------------------------------------------------------------------------
    # Recognition
    # -------------------------------------------------------------------------
    
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize a face.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            RecognitionResult with identity, confidence, and category
        """
        with self._lock:
            return self.recognizer.recognize(face_image)
    
    def recognize_batch(
        self, 
        face_images: List[np.ndarray]
    ) -> List[RecognitionResult]:
        """Recognize multiple faces.
        
        Args:
            face_images: List of face images
            
        Returns:
            List of RecognitionResults
        """
        with self._lock:
            return [self.recognizer.recognize(img) for img in face_images]
    
    # -------------------------------------------------------------------------
    # Statistics & Info
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> dict:
        """Get service statistics.
        
        Returns:
            Dictionary with counts and stats
        """
        db_stats = self.database.get_stats()
        
        return {
            "total_faces": db_stats["total_faces"],
            "unique_persons": db_stats["unique_persons"],
            "by_status": db_stats["by_status"],
            "by_type": db_stats["by_type"],
            "enrolled_in_recognizer": self.recognizer.get_enrolled_count(),
            "pending_in_queue": self.processing_queue.get_pending_count(),
            "queue_size": self.processing_queue.get_queue_size(),
        }
    
    def get_enrolled_persons(self) -> List[str]:
        """Get list of enrolled person names.
        
        Returns:
            List of person names with embeddings in recognizer
        """
        return self.recognizer.get_enrolled_names()
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def reload_embeddings(self):
        """Reload all embeddings from database."""
        logger.info("Reloading embeddings from database")
        self._load_embeddings_from_database()
    
    def shutdown(self):
        """Shutdown the service gracefully."""
        logger.info("Shutting down FaceRecognizerService")
        self.processing_queue.stop()
        self.database.close()
