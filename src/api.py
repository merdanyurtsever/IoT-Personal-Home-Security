"""FastAPI routes for face detection and recognition API.

Simplified API using the new modular structure.
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Schemas
# =============================================================================

class FaceUploadResponse(BaseModel):
    message: str
    person_name: str
    files_saved: int


class RecognizeResponse(BaseModel):
    faces_detected: int
    results: List[dict]


class IdentityResponse(BaseModel):
    identities: List[str]
    total: int


class HealthResponse(BaseModel):
    status: str
    version: str
    detection_backend: str
    embedding_backend: str
    registered_identities: int


class ErrorResponse(BaseModel):
    detail: str


# =============================================================================
# Service Layer
# =============================================================================

# Global recognizer instance
_recognizer = None
_upload_dir = Path("data/raw/faces/uploads")


def get_recognizer():
    """Get the global face recognizer instance."""
    global _recognizer
    if _recognizer is None:
        from .visual import FaceRecognizer
        _recognizer = FaceRecognizer(
            embedding_backend="opencv_dnn",
            detection_backend="opencv_dnn",
            similarity_threshold=0.6,
        )
    return _recognizer


def set_recognizer(recognizer):
    """Set the global face recognizer instance."""
    global _recognizer
    _recognizer = recognizer


def set_upload_dir(path: str):
    """Set the upload directory."""
    global _upload_dir
    _upload_dir = Path(path)
    _upload_dir.mkdir(parents=True, exist_ok=True)


# Legacy compatibility
def set_face_service(service):
    """Legacy function for backward compatibility."""
    logger.warning("set_face_service is deprecated, use set_recognizer instead")
    

def get_face_service():
    """Legacy function for backward compatibility."""
    logger.warning("get_face_service is deprecated, use get_recognizer instead")
    return get_recognizer()


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["faces"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check():
    """Health check endpoint."""
    recognizer = get_recognizer()
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        detection_backend="opencv_dnn",
        embedding_backend=recognizer.embedding_backend,
        registered_identities=len(recognizer.get_registered_identities()),
    )


@router.get(
    "/identities",
    response_model=IdentityResponse,
    summary="List registered identities",
)
async def list_identities():
    """Get list of all registered identities."""
    recognizer = get_recognizer()
    identities = recognizer.get_registered_identities()
    
    return IdentityResponse(
        identities=identities,
        total=len(identities),
    )


@router.post(
    "/faces/upload",
    response_model=FaceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload face images",
)
async def upload_faces(
    person_name: str = Form(..., description="Name/identifier for the person"),
    files: List[UploadFile] = File(..., description="Face image files"),
):
    """Upload face images and register them."""
    import cv2
    import numpy as np
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded",
        )
    
    recognizer = get_recognizer()
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    saved_count = 0
    
    # Ensure upload directory exists
    person_dir = _upload_dir / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        if not file.filename:
            continue
        
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            continue
        
        # Save file
        unique_name = f"{uuid.uuid4().hex[:8]}{ext}"
        file_path = person_dir / unique_name
        
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Load and register the face
            nparr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                if recognizer.register_face(person_name, image, detect=True):
                    saved_count += 1
                    logger.info(f"Registered face for {person_name} from {file.filename}")
                else:
                    logger.warning(f"No face detected in {file.filename}")
                    file_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            file_path.unlink(missing_ok=True)
    
    if saved_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No faces could be detected in the uploaded images",
        )
    
    return FaceUploadResponse(
        message=f"Successfully registered {saved_count} face(s)",
        person_name=person_name,
        files_saved=saved_count,
    )


@router.post(
    "/faces/recognize",
    response_model=RecognizeResponse,
    summary="Recognize faces in image",
)
async def recognize_faces(
    file: UploadFile = File(..., description="Image file to analyze"),
):
    """Recognize faces in an uploaded image."""
    import cv2
    import numpy as np
    
    recognizer = get_recognizer()
    
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode image",
            )
        
        results = recognizer.recognize(image, detect_attributes=False)
        
        return RecognizeResponse(
            faces_detected=len(results),
            results=[
                {
                    "identity": r.identity,
                    "confidence": round(r.confidence, 3),
                    "category": r.category.value,
                    "is_alert": r.should_alert,
                }
                for r in results
            ],
        )
        
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recognition failed: {e}",
        )


@router.delete(
    "/identities/{person_name}",
    status_code=status.HTTP_200_OK,
    summary="Remove an identity",
)
async def remove_identity(
    person_name: str,
    delete_images: bool = Query(True, description="Also delete image files"),
):
    """Remove an identity from the recognizer."""
    recognizer = get_recognizer()
    
    if not recognizer.remove_identity(person_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Identity '{person_name}' not found",
        )
    
    # Optionally delete images
    if delete_images:
        person_dir = _upload_dir / person_name
        if person_dir.exists():
            shutil.rmtree(person_dir)
    
    return {"message": f"Removed identity: {person_name}"}


@router.post(
    "/register-directory",
    summary="Register faces from directory",
)
async def register_from_directory(
    directory: str = Form(..., description="Path to directory with person subdirectories"),
):
    """Register faces from a directory structure."""
    recognizer = get_recognizer()
    
    dir_path = Path(directory)
    if not dir_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {directory}",
        )
    
    results = recognizer.register_from_directory(directory)
    
    return {
        "message": f"Registered {sum(results.values())} faces for {len(results)} people",
        "details": results,
    }


# =============================================================================
# Application Factory
# =============================================================================

def create_app(
    recognizer=None,
    upload_dir: str = "data/raw/faces/uploads",
    cors_origins: List[str] = None,
    debug: bool = False,
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        recognizer: Optional FaceRecognizer instance
        upload_dir: Directory for uploaded images
        cors_origins: List of allowed CORS origins
        debug: Enable debug mode
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="IoT Home Security - Face API",
        description="API for face detection and recognition",
        version="2.0.0",
        debug=debug,
    )
    
    # Add CORS middleware
    if cors_origins is None:
        cors_origins = ["*"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Set up service
    if recognizer is not None:
        set_recognizer(recognizer)
    set_upload_dir(upload_dir)
    
    # Include router
    app.include_router(router)
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        return {
            "message": "IoT Home Security - Face API",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    return app
