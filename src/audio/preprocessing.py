"""Audio preprocessing utilities."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..constants import get_audio_processing_config

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = None
    duration: float = None
    mono: bool = True
    normalize: bool = True
    trim_silence: bool = False
    silence_threshold: float = None
    
    def __post_init__(self):
        """Set defaults from global config if not specified."""
        audio_config = get_audio_processing_config()
        if self.sample_rate is None:
            self.sample_rate = audio_config.sample_rate
        if self.duration is None:
            self.duration = audio_config.duration
        if self.silence_threshold is None:
            self.silence_threshold = audio_config.silence_threshold


class AudioPreprocessor:
    """Audio preprocessing: loading, resampling, normalization."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize audio preprocessor.
        
        Args:
            config: Audio configuration (uses defaults if None)
        """
        self.config = config or AudioConfig()
        self._librosa_available = self._check_librosa()
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available."""
        try:
            import librosa
            return True
        except ImportError:
            logger.warning("librosa not installed. Install with: pip install librosa")
            return False
    
    def load(
        self,
        audio_path: Union[str, Path],
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """Load audio file.
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (uses config if None)
            duration: Maximum duration in seconds
            offset: Start time in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        sr = sr or self.config.sample_rate
        duration = duration or self.config.duration
        
        if self._librosa_available:
            return self._load_librosa(audio_path, sr, duration, offset)
        else:
            return self._load_scipy(audio_path, sr, duration, offset)
    
    def _load_librosa(
        self,
        audio_path: Union[str, Path],
        sr: int,
        duration: float,
        offset: float,
    ) -> Tuple[np.ndarray, int]:
        """Load audio using librosa."""
        import librosa
        
        audio, sample_rate = librosa.load(
            str(audio_path),
            sr=sr,
            mono=self.config.mono,
            duration=duration,
            offset=offset,
        )
        
        if self.config.normalize:
            audio = self.normalize(audio)
        
        if self.config.trim_silence:
            audio = self.trim_silence(audio)
        
        return audio, sample_rate
    
    def _load_scipy(
        self,
        audio_path: Union[str, Path],
        sr: int,
        duration: float,
        offset: float,
    ) -> Tuple[np.ndarray, int]:
        """Load audio using scipy (WAV only)."""
        from scipy.io import wavfile
        
        original_sr, audio = wavfile.read(str(audio_path))
        audio_config = get_audio_processing_config()
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / audio_config.int16_max
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / audio_config.int32_max
        
        if len(audio.shape) > 1 and self.config.mono:
            audio = np.mean(audio, axis=1)
        
        if original_sr != sr:
            audio = self.resample(audio, original_sr, sr)
        
        start_sample = int(offset * sr)
        end_sample = start_sample + int(duration * sr)
        audio = audio[start_sample:end_sample]
        
        if self.config.normalize:
            audio = self.normalize(audio)
        
        return audio, sr
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate.
        
        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        
        if self._librosa_available:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples)
    
    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Remove silence from beginning and end of audio.
        
        Args:
            audio: Audio data
            threshold: Silence threshold
            
        Returns:
            Trimmed audio
        """
        threshold = threshold or self.config.silence_threshold
        
        if self._librosa_available:
            import librosa
            trimmed, _ = librosa.effects.trim(audio, top_db=get_audio_processing_config().trim_top_db)
            return trimmed
        else:
            non_silent = np.abs(audio) > threshold
            if not np.any(non_silent):
                return audio
            
            start = np.argmax(non_silent)
            end = len(audio) - np.argmax(non_silent[::-1])
            return audio[start:end]
    
    def pad_or_truncate(
        self,
        audio: np.ndarray,
        target_length: int,
        pad_mode: str = "constant",
    ) -> np.ndarray:
        """Pad or truncate audio to target length.
        
        Args:
            audio: Audio data
            target_length: Target number of samples
            pad_mode: Numpy pad mode ("constant", "reflect", "edge")
            
        Returns:
            Audio of target length
        """
        if len(audio) > target_length:
            return audio[:target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            return np.pad(audio, (0, padding), mode=pad_mode)
        return audio
    
    def preprocess(
        self,
        audio: np.ndarray,
        target_length: Optional[int] = None,
    ) -> np.ndarray:
        """Apply full preprocessing pipeline.
        
        Args:
            audio: Raw audio data
            target_length: Optional target length in samples
            
        Returns:
            Preprocessed audio
        """
        if self.config.normalize:
            audio = self.normalize(audio)
        
        if self.config.trim_silence:
            audio = self.trim_silence(audio)
        
        if target_length is not None:
            audio = self.pad_or_truncate(audio, target_length)
        
        return audio
