"""# Gerçek zamanlı test
python -m src.audio_processing.realtime_test bunu çalıştırıyorum ama o çalışmazsa
önce debug ı çalıştır

# Model debug
python -m src.audio_processing.debug_model

# Veri yeniden işleme
python -m src.audio_processing.preprocessing

# Model yeniden eğitimi  
python -m src.audio_processing.train_model

# Sadece bu komutu çalıştırın:
python -m src.audio_processing.realtime_test
"""


import sounddevice as sd
import numpy as np
import pickle
import time
import os
import sys
import json
from datetime import datetime

print("Kullanılabilir Ses Cihazları:")
print(sd.query_devices())  
print("\n" + "="*50 + "\n")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_features import AudioFeatureExtractor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "svm_audio_model.pkl")

SAMPLE_RATE = 16000
DURATION = 1.0  
VOLUME_THRESHOLD = 0.01  

#  ses özellik çıkarm
extractor = AudioFeatureExtractor(sr=SAMPLE_RATE)

def listen_and_predict():
    print("📌 Model yükleniyor...")
    print(f"Model yolu: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        return
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        classes = model_data['classes']
        
        print(f" Model başarıyla yüklendi!")
        print(f"Sınıflar: {classes}")
        print(f"Model accuracy: {model_data['accuracy']:.3f}")
        print(f"Özellik boyutu: {model_data['feature_size']}")
        print(f" Scaler kullanılıyor: {'Evet' if scaler else 'Hayır'}")
        
    except Exception as e:
        print(f" Model yükleme hatası: {e}")
        return

    print("\n Gerçek zamanlı dinleme başladı!")
    print(" Mikrofon açık, ses analiz ediliyor...")
    print("Kapatmak için CTRL + C\n")

    try:
        while True:
            print("Ses alınıyor...")

            # Ses kaydı
            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE,
                          channels=1,
                          dtype='float32')
            sd.wait()

            audio = audio.flatten()
            
            # Ses seviyesi kontrolü
            volume = np.sqrt(np.mean(audio**2))
            
            if volume > VOLUME_THRESHOLD:
                try:
                    # Özellik çıkarımı
                    features = extractor.extract_features(audio, SAMPLE_RATE)
                    features = features.reshape(1, -1)
                    
                    # Normalleştirme 
                    if scaler is not None:
                        features = scaler.transform(features)

                    
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0]

                    predicted_class = classes[pred]
                    confidence = prob[pred]
                    
                    print(f"Tespit: {predicted_class} (güven: {confidence:.3f})")
                    print(f"Ses seviyesi: {volume:.4f}")
                    
                    # ACİL DURUM KONTROLÜ
                    if predicted_class == 'woman_scream' and confidence > 0.4:
                        print("\n" + "=" * 50)
                        print("ACİL DURUM TESPİT EDİLDİ!")
                        print("KADIN ÇIĞLIĞI ALGILANDI!")
                        print("=" * 50 + "\n")
                        
                        with open("emergency.json", "w") as f:
                            json.dump({"emergency": True, "time": datetime.now().isoformat(), "confidence": confidence}, f)
                        
                    elif predicted_class == 'man_scream' and confidence > 0.5:
                        print("\nERKEK ÇIĞLIĞI TESPİT EDİLDİ!")
                        print("DİKKAT GEREKTİRİYOR!\n")
                    
                    print(f"Detay:")
                    for i, (class_name, probability) in enumerate(zip(classes, prob)):
                        print(f"   {class_name}: {probability:.3f}")
                    
                except Exception as e:
                    print(f"Ses işleme hatası: {e}")
            else:
                print("Çok sessiz - atlanıyor...")
            
            print("-" * 50)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nGerçek zamanlı dinleme durduruldu.")
        print("Program kapatıldı.")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")

if __name__ == "__main__":
    listen_and_predict()
