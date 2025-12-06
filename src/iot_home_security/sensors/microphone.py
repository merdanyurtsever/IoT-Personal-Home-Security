"""Microphone interface module.

This module provides microphone access for audio capture
using PyAudio or other audio backends.
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional

import numpy as np


class BaseMicrophoneInterface(ABC):
    """Abstract base class for microphone interfaces."""
    
    @abstractmethod
    def start(self) -> None:
        """Start audio capture."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop audio capture."""
        pass
    
    @abstractmethod
    def read(self, duration: float) -> np.ndarray:
        """Read audio for specified duration.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Audio samples as numpy array
        """
        pass
    
    @abstractmethod
    def stream(self, chunk_duration: float) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks.
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            
        Yields:
            Audio chunks as numpy arrays
        """
        pass


class PyAudioMicrophone(BaseMicrophoneInterface):
    """Microphone interface using PyAudio."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        """Initialize PyAudio microphone.
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            chunk_size: Chunk size for streaming
            device_index: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        self.audio = None
        self.stream = None
    
    def start(self) -> None:
        """Start audio capture."""
        try:
            import pyaudio
            
            self.audio = pyaudio.PyAudio()
            self._stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )
        except ImportError:
            raise ImportError(
                "PyAudio is required for microphone access. "
                "Install with: pip install pyaudio"
            )
    
    def stop(self) -> None:
        """Stop audio capture."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None
    
    def read(self, duration: float) -> np.ndarray:
        """Read audio for specified duration."""
        if self._stream is None:
            raise RuntimeError("Microphone not started")
        
        num_samples = int(self.sample_rate * duration)
        num_chunks = num_samples // self.chunk_size + 1
        
        frames = []
        for _ in range(num_chunks):
            data = self._stream.read(self.chunk_size)
            frames.append(np.frombuffer(data, dtype=np.float32))
        
        audio = np.concatenate(frames)[:num_samples]
        return audio
    
    def stream(self, chunk_duration: float = 0.1) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks."""
        if self._stream is None:
            self.start()
        
        chunk_samples = int(self.sample_rate * chunk_duration)
        
        try:
            while True:
                data = self._stream.read(chunk_samples)
                yield np.frombuffer(data, dtype=np.float32)
        finally:
            self.stop()


class MicrophoneInterface:
    """Unified microphone interface."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        """Initialize microphone interface.
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of channels
            chunk_size: Chunk size for streaming
            device_index: Device index (None for default)
        """
        self.microphone = PyAudioMicrophone(
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            device_index=device_index,
        )
        
        self.sample_rate = sample_rate
        self.is_running = False
    
    def start(self) -> None:
        """Start audio capture."""
        self.microphone.start()
        self.is_running = True
    
    def stop(self) -> None:
        """Stop audio capture."""
        self.microphone.stop()
        self.is_running = False
    
    def read(self, duration: float) -> np.ndarray:
        """Read audio for specified duration."""
        return self.microphone.read(duration)
    
    def stream(self, chunk_duration: float = 0.1) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks."""
        yield from self.microphone.stream(chunk_duration)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    @staticmethod
    def list_devices() -> list:
        """List available audio input devices.
        
        Returns:
            List of device info dictionaries
        """
        try:
            import pyaudio
            
            audio = pyaudio.PyAudio()
            devices = []
            
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append({
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                    })
            
            audio.terminate()
            return devices
        except ImportError:
            return []
