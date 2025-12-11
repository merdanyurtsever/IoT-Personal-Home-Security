import os
import numpy as np
import librosa
import soundfile as sf

from .extract_features import AudioFeatureExtractor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "sounds")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

CLASSES = ["man_scream", "woman_scream", "noise"]  

extractor = AudioFeatureExtractor(sr=16000)

def preprocess_and_extract():
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    X = []
    y = []

    print("Starting preprocessing...\n")

    for label, class_name in enumerate(CLASSES):
        raw_folder = os.path.join(RAW_PATH, class_name)
        processed_folder = os.path.join(PROCESSED_PATH, class_name)

        os.makedirs(processed_folder, exist_ok=True)

        print(f"Processing: {class_name}")

        if not os.path.exists(raw_folder):
            print(f"{raw_folder} bulunamadı, atlanıyor...")
            continue

        for file in os.listdir(raw_folder):
            if file.endswith(".wav"):
                raw_file_path = os.path.join(raw_folder, file)
                processed_file_path = os.path.join(processed_folder, file)

                try:
                    audio, sr = librosa.load(raw_file_path, sr=16000)
                    sf.write(processed_file_path, audio, sr)

                    features = extractor.extract_features(audio, sr)

                    X.append(features)
                    y.append(label)

                except Exception as e:
                    print(f"Hata: {file} işlenemedi → {e}")

    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(PROCESSED_PATH, "X.npy"), X)
    np.save(os.path.join(PROCESSED_PATH, "y.npy"), y)

    print(f"\nPreprocessing completed!")
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1] if len(X) > 0 else 0}")
    print(f"Saved to: {PROCESSED_PATH}")
    
    unique, counts = np.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"{CLASSES[class_idx]}: {count} samples")


if __name__ == "__main__":
    preprocess_and_extract()
