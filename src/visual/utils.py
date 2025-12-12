"""Face image processing utilities."""

from typing import Tuple

import cv2
import numpy as np


def crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2,
) -> np.ndarray:
    """Crop face from image with margin.
    
    Args:
        image: Full image
        bbox: Bounding box (x, y, w, h)
        margin: Margin around face as fraction of size
        
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image.shape[1], x + w + margin_w)
    y2 = min(image.shape[0], y + h + margin_h)
    return image[y1:y2, x1:x2]


def align_face(
    face_image: np.ndarray,
    landmarks: dict[str, tuple[int, int]],
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Align face using eye landmarks.
    
    Args:
        face_image: Cropped face image
        landmarks: Dictionary with 'left_eye' and 'right_eye' coordinates
        output_size: Output image size
        
    Returns:
        Aligned face image
    """
    # TODO: Implement face alignment
    return cv2.resize(face_image, output_size)


def preprocess_face(
    face_image: np.ndarray,
    target_size: Tuple[int, int] = (160, 160),
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess face image for model input.
    
    Args:
        face_image: Face image
        target_size: Target size for resizing
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed face image
    """
    face = cv2.resize(face_image, target_size)
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    if normalize:
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
    return face


def compute_face_quality(face_image: np.ndarray) -> float:
    """Compute face image quality score.
    
    Args:
        face_image: Face image
        
    Returns:
        Quality score between 0 and 1
    """
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 500.0, 1.0)
    brightness = gray.mean() / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2
    return float((sharpness + brightness_score) / 2)
