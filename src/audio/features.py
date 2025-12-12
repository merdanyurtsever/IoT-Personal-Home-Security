"""Audio feature extraction."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: float = 0.0
    fmax: Optional[float] = None
    power: float = 2.0


class FeatureExtractor:
    """Extract audio features for classification."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature extractor.
        
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        self._librosa_available = self._check_librosa()
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available."""
        try:
            import librosa
            return True
        except ImportError:
            logger.warning("librosa not installed for feature extraction")
            return False
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        sr: int,
        n_mfcc: Optional[int] = None,
    ) -> np.ndarray:
        """Extract MFCC features.
        
        Args:
            audio: Audio data
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features (n_mfcc x time)
        """
        n_mfcc = n_mfcc or self.config.n_mfcc
        
        if self._librosa_available:
            import librosa
            return librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
            )
        else:
            return self._simple_mfcc(audio, sr, n_mfcc)
    
    def _simple_mfcc(
        self,
        audio: np.ndarray,
        sr: int,
        n_mfcc: int,
    ) -> np.ndarray:
        """Simple MFCC implementation without librosa."""
        from scipy.fftpack import dct
        
        frame_size = self.config.n_fft
        hop = self.config.hop_length
        
        num_frames = 1 + (len(audio) - frame_size) // hop
        frames = np.zeros((num_frames, frame_size))
        
        for i in range(num_frames):
            start = i * hop
            frames[i] = audio[start:start + frame_size] * np.hanning(frame_size)
        
        power_spectrum = np.abs(np.fft.rfft(frames, n=frame_size)) ** 2
        
        n_mels = self.config.n_mels
        mel_min = 2595 * np.log10(1 + self.config.fmin / 700)
        mel_max = 2595 * np.log10(1 + (self.config.fmax or sr/2) / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((frame_size + 1) * hz_points / sr).astype(int)
        
        filterbank = np.zeros((n_mels, frame_size // 2 + 1))
        for i in range(n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                filterbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                filterbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
        
        mel_spectrum = np.dot(power_spectrum, filterbank.T)
        log_mel = np.log(mel_spectrum + 1e-10)
        
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
        
        return mfcc.T
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int,
        n_mels: Optional[int] = None,
        to_db: bool = True,
    ) -> np.ndarray:
        """Extract Mel spectrogram.
        
        Args:
            audio: Audio data
            sr: Sample rate
            n_mels: Number of Mel bands
            to_db: Convert to decibels
            
        Returns:
            Mel spectrogram (n_mels x time)
        """
        n_mels = n_mels or self.config.n_mels
        
        if not self._librosa_available:
            logger.error("librosa required for Mel spectrogram")
            return np.zeros((n_mels, 1))
        
        import librosa
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            power=self.config.power,
        )
        
        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def extract_spectral_features(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict:
        """Extract various spectral features.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dict of feature name to feature array
        """
        if not self._librosa_available:
            logger.error("librosa required for spectral features")
            return {}
        
        import librosa
        
        features = {}
        
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.config.hop_length
        )[0]
        
        features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=self.config.hop_length
        )[0]
        
        features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.config.hop_length
        )[0]
        
        features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.config.hop_length
        )[0]
        
        features["rms"] = librosa.feature.rms(
            y=audio, hop_length=self.config.hop_length
        )[0]
        
        return features
    
    def extract_all(
        self,
        audio: np.ndarray,
        sr: int,
        aggregate: bool = True,
    ) -> np.ndarray:
        """Extract all features and optionally aggregate.
        
        Args:
            audio: Audio data
            sr: Sample rate
            aggregate: If True, compute mean/std statistics
            
        Returns:
            Feature vector
        """
        mfcc = self.extract_mfcc(audio, sr)
        
        if not aggregate:
            return mfcc
        
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        features = [mfcc_mean, mfcc_std]
        
        if self._librosa_available:
            spectral = self.extract_spectral_features(audio, sr)
            for name, feat in spectral.items():
                features.append([np.mean(feat), np.std(feat)])
        
        return np.concatenate([f.flatten() for f in features])
