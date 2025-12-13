"""Camera capture implementation."""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generator, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraBackend(Enum):
    """Available camera backends."""
    OPENCV = "opencv"
    PICAMERA = "picamera"
    RTSP = "rtsp"


@dataclass
class CameraConfig:
    """Camera configuration."""
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: CameraBackend = CameraBackend.OPENCV
    device: Union[int, str] = 0
    buffer_size: int = 1
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0


@dataclass
class Frame:
    """A captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int
    
    @property
    def shape(self) -> tuple:
        return self.image.shape
    
    @property
    def height(self) -> int:
        return self.image.shape[0]
    
    @property
    def width(self) -> int:
        return self.image.shape[1]


class Camera:
    """Unified camera interface for different backends."""
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """Initialize camera.
        
        Args:
            config: Camera configuration (uses defaults if None)
        """
        self.config = config or CameraConfig()
        
        self._capture = None
        self._is_open = False
        self._frame_count = 0
        self._lock = threading.Lock()
        
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._current_frame: Optional[Frame] = None
    
    def open(self) -> bool:
        """Open camera for capture.
        
        Returns:
            True if camera opened successfully
        """
        with self._lock:
            if self._is_open:
                return True
            
            if self.config.backend == CameraBackend.PICAMERA:
                return self._open_picamera()
            else:
                return self._open_opencv()
    
    def _open_opencv(self) -> bool:
        """Open camera using OpenCV."""
        try:
            device = self.config.device
            
            if isinstance(device, str) and (
                device.startswith("rtsp://") or 
                device.startswith("http://") or
                device.startswith("https://")
            ):
                self._capture = cv2.VideoCapture(device)
            else:
                if isinstance(device, str):
                    device = int(device)
                self._capture = cv2.VideoCapture(device)
            
            if not self._capture.isOpened():
                logger.error(f"Failed to open camera device {device}")
                return False
            
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            self._is_open = True
            logger.info(f"Opened OpenCV camera: {device}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening OpenCV camera: {e}")
            return False
    
    def _open_picamera(self) -> bool:
        """Open camera using PiCamera2."""
        try:
            from picamera2 import Picamera2
            
            self._capture = Picamera2()
            
            config = self._capture.create_preview_configuration(
                main={
                    "size": (self.config.width, self.config.height),
                    "format": "RGB888",
                }
            )
            self._capture.configure(config)
            self._capture.start()
            
            self._is_open = True
            logger.info("Opened PiCamera2")
            return True
            
        except ImportError:
            logger.error("picamera2 not installed. Install with: pip install picamera2")
            return False
        except Exception as e:
            logger.error(f"Error opening PiCamera: {e}")
            return False
    
    def close(self):
        """Close camera and release resources."""
        self.stop_stream()
        
        with self._lock:
            if self._capture is not None:
                if self.config.backend == CameraBackend.PICAMERA:
                    try:
                        self._capture.stop()
                        self._capture.close()
                    except Exception:
                        pass
                else:
                    self._capture.release()
                
                self._capture = None
            
            self._is_open = False
            logger.info("Camera closed")
    
    def read(self) -> Optional[Frame]:
        """Read a single frame from the camera.
        
        Returns:
            Frame object or None if read failed
        """
        if not self._is_open:
            if not self.open():
                return None
        
        with self._lock:
            return self._read_frame()
    
    def _read_frame(self) -> Optional[Frame]:
        """Internal frame read (assumes lock is held)."""
        try:
            if self.config.backend == CameraBackend.PICAMERA:
                image = self._capture.capture_array()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                ret, image = self._capture.read()
                if not ret or image is None:
                    if self.config.auto_reconnect:
                        self._handle_reconnect()
                    return None
            
            self._frame_count += 1
            
            return Frame(
                image=image,
                timestamp=time.time(),
                frame_number=self._frame_count,
            )
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
    
    def _handle_reconnect(self):
        """Handle camera reconnection."""
        logger.warning("Camera disconnected, attempting reconnect...")
        
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        
        self._is_open = False
        time.sleep(self.config.reconnect_delay)
        
        if self._open_opencv():
            logger.info("Camera reconnected successfully")
    
    def stream(self, max_retries: int = 5) -> Generator[Frame, None, None]:
        """Generator that yields frames continuously.
        
        Args:
            max_retries: Maximum number of consecutive failures before giving up
        
        Yields:
            Frame objects
        """
        consecutive_failures = 0
        
        while True:
            frame = self.read()
            if frame is not None:
                consecutive_failures = 0  # Reset on success
                yield frame
            else:
                consecutive_failures += 1
                
                if consecutive_failures >= max_retries:
                    logger.error(f"Camera failed after {max_retries} attempts. Stopping.")
                    break
                
                if not self.config.auto_reconnect:
                    break
                    
                # Add delay before retry to prevent crash loop
                time.sleep(0.5)
    
    def start_stream(self, callback: Callable[[Frame], None]):
        """Start background streaming with callback.
        
        Args:
            callback: Function called with each frame
        """
        if self._streaming:
            return
        
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback,),
            daemon=True,
        )
        self._stream_thread.start()
        logger.info("Started camera stream")
    
    def _stream_loop(self, callback: Callable[[Frame], None]):
        """Background streaming loop."""
        while self._streaming:
            frame = self.read()
            if frame is not None:
                self._current_frame = frame
                try:
                    callback(frame)
                except Exception as e:
                    logger.error(f"Stream callback error: {e}")
    
    def stop_stream(self):
        """Stop background streaming."""
        self._streaming = False
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        logger.info("Stopped camera stream")
    
    def get_current_frame(self) -> Optional[Frame]:
        """Get most recent frame from background stream."""
        return self._current_frame
    
    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open
    
    @property
    def is_streaming(self) -> bool:
        """Check if background streaming is active."""
        return self._streaming
    
    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
