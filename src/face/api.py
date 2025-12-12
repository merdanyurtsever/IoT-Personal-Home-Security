"""Standalone Face Recognition API for mobile app integration.

This API can run independently from the main security system.
Start with: python -m src.face api --port 8000

Endpoints:
    GET  /health              - Health check
    GET  /identities          - List registered identities
    POST /faces/upload        - Upload face images for registration
    POST /faces/recognize     - Recognize faces in an image
    DELETE /identities/{name} - Remove an identity
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

try:
    from fastapi import (
        APIRouter,
        FastAPI,
        File,
        Form,
        HTTPException,
        UploadFile,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Schemas
# =============================================================================

if FASTAPI_AVAILABLE:
    class HealthResponse(BaseModel):
        status: str
        version: str
        detection_backend: str
        embedding_backend: str
        registered_identities: int

    class IdentityResponse(BaseModel):
        identities: List[str]
        total: int

    class FaceUploadResponse(BaseModel):
        message: str
        person_name: str
        files_saved: int

    class RecognizeResponse(BaseModel):
        faces_detected: int
        results: List[dict]


# =============================================================================
# Global State
# =============================================================================

_recognizer = None
_upload_dir = Path("data/raw/faces/uploads")


def get_recognizer():
    """Get or create the face recognizer instance."""
    global _recognizer
    if _recognizer is None:
        from .recognition import FaceRecognizer
        _recognizer = FaceRecognizer(
            embedding_backend="opencv_dnn",
            detection_backend="opencv_dnn",
            similarity_threshold=0.6,
        )
        
        # Load watch list if exists
        watch_list = Path("data/raw/faces/watch_list")
        if watch_list.exists():
            results = _recognizer.register_from_directory(str(watch_list))
            logger.info(f"Loaded {sum(results.values())} faces from watch list")
    
    return _recognizer


# =============================================================================
# API Routes
# =============================================================================

def create_app() -> "FastAPI":
    """Create the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Face Recognition API",
        description="Standalone face detection and recognition API for IoT Security",
        version="2.0.0",
    )
    
    # CORS for mobile app access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    router = APIRouter(prefix="/api/v1", tags=["faces"])
    
    @router.get("/health", response_model=HealthResponse)
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
    
    @router.get("/identities", response_model=IdentityResponse)
    async def list_identities():
        """Get list of all registered identities."""
        recognizer = get_recognizer()
        identities = recognizer.get_registered_identities()
        return IdentityResponse(identities=identities, total=len(identities))
    
    @router.post("/faces/upload", response_model=FaceUploadResponse)
    async def upload_faces(
        person_name: str = Form(...),
        files: List[UploadFile] = File(...),
    ):
        """Upload face images for a person."""
        _upload_dir.mkdir(parents=True, exist_ok=True)
        person_dir = _upload_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        recognizer = get_recognizer()
        
        for file in files:
            # Save file
            file_id = str(uuid.uuid4())[:8]
            file_path = person_dir / f"{file_id}_{file.filename}"
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Register face
            image = cv2.imread(str(file_path))
            if image is not None and recognizer.register_face(person_name, image):
                saved_count += 1
        
        return FaceUploadResponse(
            message=f"Registered {saved_count} face(s)",
            person_name=person_name,
            files_saved=saved_count,
        )
    
    @router.post("/faces/recognize", response_model=RecognizeResponse)
    async def recognize_faces(file: UploadFile = File(...)):
        """Recognize faces in an uploaded image."""
        # Read image from upload
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        recognizer = get_recognizer()
        results = recognizer.recognize_faces(image)
        
        return RecognizeResponse(
            faces_detected=len(results),
            results=[
                {
                    "identity": r.identity or "Unknown",
                    "confidence": round(r.confidence, 3),
                    "category": r.category.value,
                }
                for r in results
            ],
        )
    
    @router.delete("/identities/{name}")
    async def remove_identity(name: str):
        """Remove a registered identity."""
        recognizer = get_recognizer()
        
        if name not in recognizer.get_registered_identities():
            raise HTTPException(status_code=404, detail=f"Identity '{name}' not found")
        
        recognizer.remove_identity(name)
        
        # Also remove uploaded files
        person_dir = _upload_dir / name
        if person_dir.exists():
            shutil.rmtree(person_dir)
        
        return {"message": f"Removed identity: {name}"}
    
    app.include_router(router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Face Recognition API",
            "version": "2.0.0",
            "docs": "/docs",
        }
    
    return app


# For running with uvicorn directly: uvicorn src.face.api:app
app = None
if FASTAPI_AVAILABLE:
    try:
        app = create_app()
    except Exception:
        pass
