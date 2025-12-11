"""FastAPI routes and schemas for face management API.

Provides endpoints for:
- Face upload and management (CRUD)
- Processing trigger with job tracking
- Real-time status via Server-Sent Events
- Health checks
"""

import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .face_service import FaceRecognizerService

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Schemas (formerly schemas.py)
# =============================================================================

class FaceTypeEnum(str, Enum):
    """Face type for API - all faces are on watch list (people to alert about)."""
    watch_list = "watch_list"


class FaceStatusEnum(str, Enum):
    """Face processing status for API."""
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


# Request Models
class FaceUploadRequest(BaseModel):
    person_name: str = Field(..., min_length=1, max_length=100, description="Name of person to watch")
    face_type: FaceTypeEnum = Field(default=FaceTypeEnum.watch_list, description="Face type (always watch_list)")


class ProcessFacesRequest(BaseModel):
    face_ids: Optional[List[int]] = Field(default=None)


class UpdateFaceRequest(BaseModel):
    person_name: Optional[str] = Field(None, min_length=1, max_length=100)
    face_type: Optional[FaceTypeEnum] = None


# Response Models
class FaceResponse(BaseModel):
    id: int
    person_name: str
    face_type: FaceTypeEnum
    status: FaceStatusEnum
    image_path: str
    embedding_size: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    class Config:
        from_attributes = True


class FaceListResponse(BaseModel):
    faces: List[FaceResponse]
    total: int


class FaceUploadResponse(BaseModel):
    message: str
    face_ids: List[int]
    total_uploaded: int


class ProcessingJobResponse(BaseModel):
    job_id: str
    status: str
    total_faces: int
    processed_faces: int
    progress: float = Field(..., ge=0.0, le=1.0)
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ProcessingStartResponse(BaseModel):
    message: str
    job_id: str
    total_faces: int


class ProcessingEventResponse(BaseModel):
    event_type: str
    job_id: str
    face_id: Optional[int] = None
    progress: float
    total_faces: int
    processed_faces: int
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


class StatsResponse(BaseModel):
    total_faces: int
    unique_persons: int
    by_status: dict
    by_type: dict


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_backend: str
    pending_faces: int
    ready_faces: int


# =============================================================================
# Dependency Injection (formerly dependencies.py)
# =============================================================================

_face_service: Optional["FaceRecognizerService"] = None


def set_face_service(service: "FaceRecognizerService"):
    """Set the global face service instance."""
    global _face_service
    _face_service = service
    logger.info("Face service registered with API")


def get_face_service() -> "FaceRecognizerService":
    """Get the face service instance (FastAPI dependency)."""
    if _face_service is None:
        raise RuntimeError(
            "Face service not initialized. "
            "Call set_face_service() before starting the API."
        )
    return _face_service


def get_upload_dir() -> Path:
    """Get the upload directory for face images."""
    if _face_service is None:
        return Path("data/raw/faces/uploads")
    return _face_service.upload_dir


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["faces"])


# -----------------------------------------------------------------------------
# Health & Status Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and get basic statistics",
)
async def health_check(service=Depends(get_face_service)):
    """Health check endpoint."""
    stats = service.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        embedding_backend=service.recognizer.model_name,
        pending_faces=stats.get("by_status", {}).get("pending", 0),
        ready_faces=stats.get("by_status", {}).get("ready", 0),
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get statistics",
    description="Get detailed database statistics",
)
async def get_stats(service=Depends(get_face_service)):
    """Get database statistics."""
    return service.get_stats()


# -----------------------------------------------------------------------------
# Face CRUD Endpoints
# -----------------------------------------------------------------------------

@router.post(
    "/faces/upload",
    response_model=FaceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload face images",
    description="Upload one or more face images for a person",
)
async def upload_faces(
    person_name: str = Form(..., description="Name/identifier for the person"),
    face_type: FaceTypeEnum = Form(FaceTypeEnum.watch_list, description="Face type"),
    files: List[UploadFile] = File(..., description="Face image files"),
    service=Depends(get_face_service),
):
    """Upload face images for processing."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded",
        )
    
    # Validate file types
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    face_ids = []
    
    for file in files:
        if not file.filename:
            continue
        
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {ext}. Allowed: {allowed_extensions}",
            )
    
    # Save files and create database records
    upload_dir = service.upload_dir / person_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        if not file.filename:
            continue
        
        # Generate unique filename
        ext = Path(file.filename).suffix.lower()
        unique_name = f"{uuid.uuid4().hex[:8]}{ext}"
        file_path = upload_dir / unique_name
        
        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {e}",
            )
        
        # Create database record
        try:
            face_id = service.add_face(
                person_name=person_name,
                image_path=str(file_path),
                face_type=face_type.value,
            )
            face_ids.append(face_id)
        except Exception as e:
            logger.error(f"Failed to create face record: {e}")
            # Clean up saved file
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create face record: {e}",
            )
    
    logger.info(f"Uploaded {len(face_ids)} faces for {person_name}")
    
    return FaceUploadResponse(
        message=f"Successfully uploaded {len(face_ids)} face(s)",
        face_ids=face_ids,
        total_uploaded=len(face_ids),
    )


@router.get(
    "/faces",
    response_model=FaceListResponse,
    summary="List all faces",
    description="Get all faces with optional filtering",
)
async def list_faces(
    person_name: Optional[str] = Query(None, description="Filter by person name"),
    face_type: Optional[FaceTypeEnum] = Query(None, description="Filter by face type"),
    status_filter: Optional[FaceStatusEnum] = Query(None, alias="status", description="Filter by status"),
    service=Depends(get_face_service),
):
    """List all faces with optional filtering."""
    faces = service.list_faces(
        person_name=person_name,
        face_type=face_type.value if face_type else None,
        status=status_filter.value if status_filter else None,
    )
    
    return FaceListResponse(
        faces=[_face_record_to_response(f) for f in faces],
        total=len(faces),
    )


@router.get(
    "/faces/{face_id}",
    response_model=FaceResponse,
    summary="Get face by ID",
    description="Get details for a specific face",
)
async def get_face(
    face_id: int,
    service=Depends(get_face_service),
):
    """Get a face by ID."""
    face = service.get_face(face_id)
    
    if face is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face {face_id} not found",
        )
    
    return _face_record_to_response(face)


@router.delete(
    "/faces/{face_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete face",
    description="Delete a face by ID",
)
async def delete_face(
    face_id: int,
    delete_image: bool = Query(True, description="Also delete the image file"),
    service=Depends(get_face_service),
):
    """Delete a face by ID."""
    success = service.delete_face(face_id, delete_image=delete_image)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face {face_id} not found",
        )


@router.delete(
    "/faces/person/{person_name}",
    status_code=status.HTTP_200_OK,
    summary="Delete all faces for a person",
    description="Delete all faces for a specific person",
)
async def delete_faces_by_person(
    person_name: str,
    delete_images: bool = Query(True, description="Also delete image files"),
    service=Depends(get_face_service),
):
    """Delete all faces for a person."""
    count = service.delete_faces_by_person(person_name, delete_images=delete_images)
    
    return {"message": f"Deleted {count} face(s) for {person_name}", "deleted_count": count}


# -----------------------------------------------------------------------------
# Processing Endpoints
# -----------------------------------------------------------------------------

@router.post(
    "/faces/process",
    response_model=ProcessingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start processing faces",
    description="Trigger embedding extraction for pending faces",
)
async def start_processing(
    request: ProcessFacesRequest = None,
    service=Depends(get_face_service),
):
    """Start processing pending faces."""
    face_ids = request.face_ids if request else None
    
    job_id, total_faces = service.start_processing(face_ids=face_ids)
    
    if total_faces == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending faces to process",
        )
    
    return ProcessingStartResponse(
        message=f"Processing started for {total_faces} face(s)",
        job_id=job_id,
        total_faces=total_faces,
    )


@router.get(
    "/faces/process/{job_id}",
    response_model=ProcessingJobResponse,
    summary="Get processing job status",
    description="Get the status of a processing job",
)
async def get_job_status(
    job_id: str,
    service=Depends(get_face_service),
):
    """Get processing job status."""
    job = service.get_job_status(job_id)
    
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    
    return ProcessingJobResponse(
        job_id=job.id,
        status=job.status,
        total_faces=job.total_faces,
        processed_faces=job.processed_faces,
        progress=job.processed_faces / job.total_faces if job.total_faces > 0 else 0,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.get(
    "/faces/process/{job_id}/stream",
    summary="Stream processing status (SSE)",
    description="Stream real-time processing status via Server-Sent Events",
)
async def stream_job_status(
    job_id: str,
    service=Depends(get_face_service),
):
    """Stream processing status via Server-Sent Events."""
    
    async def event_generator():
        """Generate SSE events for processing status."""
        event_queue = asyncio.Queue()
        
        def callback(event):
            """Callback to receive processing events."""
            asyncio.get_event_loop().call_soon_threadsafe(
                event_queue.put_nowait, event
            )
        
        # Register callback
        service.add_processing_callback(job_id, callback)
        
        try:
            # Send initial status
            job = service.get_job_status(job_id)
            if job:
                initial_event = {
                    "event_type": "status",
                    "job_id": job_id,
                    "status": job.status,
                    "progress": job.processed_faces / job.total_faces if job.total_faces > 0 else 0,
                    "total_faces": job.total_faces,
                    "processed_faces": job.processed_faces,
                }
                yield f"data: {json.dumps(initial_event)}\n\n"
            
            # Stream events
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event.to_dict())}\n\n"
                    
                    # Stop streaming when job is complete
                    if event.event_type.value in ("job_completed", "job_failed"):
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f": heartbeat\n\n"
                    
                    # Check if job still exists
                    job = service.get_job_status(job_id)
                    if job is None or job.status in ("completed", "failed", "completed_with_errors"):
                        break
                        
        finally:
            # Unregister callback
            service.remove_processing_callback(job_id, callback)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _face_record_to_response(face) -> FaceResponse:
    """Convert FaceRecord to FaceResponse."""
    return FaceResponse(
        id=face.id,
        person_name=face.person_name,
        face_type=FaceTypeEnum(face.face_type.value),
        status=FaceStatusEnum(face.status.value),
        image_path=face.image_path,
        embedding_size=face.embedding_size,
        created_at=face.created_at,
        updated_at=face.updated_at,
        error_message=face.error_message,
    )


# -----------------------------------------------------------------------------
# Application Factory
# -----------------------------------------------------------------------------

def create_app(
    service: "FaceRecognizerService" = None,  # noqa: F821
    cors_origins: List[str] = None,
    debug: bool = False,
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        service: FaceRecognizerService instance
        cors_origins: List of allowed CORS origins
        debug: Enable debug mode
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="IoT Home Security - Face Management API",
        description="API for managing face recognition database and processing",
        version="1.0.0",
        debug=debug,
    )
    
    # Add CORS middleware
    if cors_origins is None:
        cors_origins = ["*"]  # Allow all for local network
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register service
    if service is not None:
        set_face_service(service)
    
    # Include router
    app.include_router(router)
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        return {
            "message": "IoT Home Security - Face Management API",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    return app
