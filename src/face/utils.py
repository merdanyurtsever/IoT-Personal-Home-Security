"""Face image processing utilities."""

from typing import Optional, Tuple

import cv2
import numpy as np

from ..constants import get_face_processing_config


def crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: Optional[float] = None,
) -> np.ndarray:
    """Crop face from image with margin.
    
    Args:
        image: Full image
        bbox: Bounding box (x, y, w, h)
        margin: Margin around face as fraction of size (uses config default if None)
        
    Returns:
        Cropped face image
    """
    if margin is None:
        margin = get_face_processing_config().crop_margin_ratio
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
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Align face using eye landmarks.
    
    Args:
        face_image: Cropped face image
        landmarks: Dictionary with 'left_eye' and 'right_eye' coordinates
        output_size: Output image size (uses config default if None)
        
    Returns:
        Aligned face image
    """
    if output_size is None:
        output_size = get_face_processing_config().alignment_output_size
    # TODO: Implement face alignment
    return cv2.resize(face_image, output_size)


def preprocess_face(
    face_image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess face image for model input.
    
    Args:
        face_image: Face image
        target_size: Target size for resizing (uses config default if None)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed face image
    """
    config = get_face_processing_config()
    if target_size is None:
        target_size = config.preprocess_target_size
    face = cv2.resize(face_image, target_size)
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    if normalize:
        face = face.astype(np.float32) / config.pixel_max_value
        face = (face - config.normalization_mean) / config.normalization_std
    return face


def compute_face_quality(face_image: np.ndarray) -> float:
    """Compute face image quality score.
    
    Args:
        face_image: Face image
        
    Returns:
        Quality score between 0 and 1
    """
    config = get_face_processing_config()
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / config.sharpness_divisor, 1.0)
    brightness = gray.mean() / config.pixel_max_value
    brightness_score = 1.0 - abs(brightness - config.brightness_target) * config.brightness_scale
    return float((sharpness + brightness_score) / 2)
