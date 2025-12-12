import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

class AudioFeatureExtractor:
    
    def __init__(self, sr=16000, n_mfcc=13, n_chroma=12, n_mels=128):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_mels = n_mels
        
    def extract_features(self, audio, sr=None):
        if sr is None:
            sr = self.sr
            
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=sr)
        
        min_length = int(0.1 * sr)  
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        features = np.zeros(70, dtype=np.float32)
        
        try:
            idx = 0
            
            # 1. MFCC özellikleri (26 özellik: 13 mean + 13 std)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            if mfcc.size > 0:
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                features[idx:idx+13] = mfcc_mean[:13]
                idx += 13
                features[idx:idx+13] = mfcc_std[:13]
                idx += 13
            else:
                idx += 26
            
            # 2. Spektral özellikler (6 özellik)
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                features[idx] = np.mean(spectral_centroids)
                features[idx+1] = np.std(spectral_centroids)
            except:
                pass
            idx += 2
            
            try:
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
                features[idx] = np.mean(spectral_rolloff)
                features[idx+1] = np.std(spectral_rolloff)
            except:
                pass
            idx += 2
            
            try:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
                features[idx] = np.mean(spectral_bandwidth)
                features[idx+1] = np.std(spectral_bandwidth)
            except:
                pass
            idx += 2
            
            # 3. Zero Crossing Rate (2 özellik)
            try:
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                features[idx] = np.mean(zcr)
                features[idx+1] = np.std(zcr)
            except:
                pass
            idx += 2
            
            # 4. RMS Energy (2 özellik)
            try:
                rms = librosa.feature.rms(y=audio)[0]
                features[idx] = np.mean(rms)
                features[idx+1] = np.std(rms)
            except:
                pass
            idx += 2
            
            # 5. Chroma özellikleri (12 özellik)
            try:
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
                if chroma.size > 0:
                    chroma_mean = np.mean(chroma, axis=1)
                    features[idx:idx+12] = chroma_mean[:12]
            except:
                pass
            idx += 12
            
            # 6. Mel-spectrogram (20 özellik)
            try:
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=20)
                if mel_spec.size > 0:
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    mel_mean = np.mean(mel_spec_db, axis=1)
                    features[idx:idx+20] = mel_mean[:20]
            except:
                pass
            idx += 20
            
        except Exception as e:
            print(f"Özellik çıkarım hatası: {e}")
            # Hata durumunda sıfır vektörü döndür
            features = np.zeros(70, dtype=np.float32)
            
        # NaN ve inf değerlerini temizle
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

def extract_mfcc(audio, sr=16000, n_mfcc=20):
    """Eski fonksiyon - geriye dönük uyumluluk için"""
    extractor = AudioFeatureExtractor(sr=sr, n_mfcc=n_mfcc)
    features = extractor.extract_features(audio, sr)
    return features
