"""
Veri Dengeleme Scripti - Her sınıftan 125 örnek alır
"""
import numpy as np
from sklearn.utils import resample
import os

def balance_dataset():
    
    print(" Veri dengeleme başlıyor...")
    
    # Veri yükle
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
    
    X_file = os.path.join(DATA_PATH, "X.npy")
    y_file = os.path.join(DATA_PATH, "y.npy")
    
    if not os.path.exists(X_file) or not os.path.exists(y_file):
        print(" Veri dosyaları bulunamadı! Önce preprocessing yapın.")
        return False
    
    X = np.load(X_file)
    y = np.load(y_file)
    
    classes = ['man_scream', 'woman_scream', 'noise']
    
    print("\n Orijinal Veri Dağılımı:")
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    for cls_idx, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"{classes[cls_idx]}: {count} örnek ({percentage:.1f}%)")
    
    # Hedef boyut: 125 (en küçük sınıf)
    target_size = 125
    print(f"\n Hedef boyut: {target_size} örnek/sınıf")
    
    # Her sınıfı ayrı ayrı işliyoz
    X_balanced = []
    y_balanced = []
    
    for class_idx in range(len(classes)):
        # Bu sınıfa ait örnekleri alcaz
        class_mask = (y == class_idx)
        X_class = X[class_mask]
        y_class = y[class_mask]
        
        print(f"\n{classes[class_idx]}:")
        print(f"  Öncesi: {len(X_class)} örnek")
        
        if len(X_class) >= target_size:
            # Rastgele örnekleme ile 125'e düşürdük
            X_resampled, y_resampled = resample(
                X_class, y_class,
                n_samples=target_size,
                random_state=42,
                replace=False 
            )
            print(f"  Sonrası: {len(X_resampled)} örnek (downsampled)")
        else:
            X_resampled = X_class
            y_resampled = y_class
            print(f"  Sonrası: {len(X_resampled)} örnek (değiştirilmedi)")
        
        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)
    
    X_final = np.vstack(X_balanced)
    y_final = np.hstack(y_balanced)
    
    from sklearn.utils import shuffle
    X_final, y_final = shuffle(X_final, y_final, random_state=42)
    
    print(f"\n Dengelenmiş Veri:")
    unique, counts = np.unique(y_final, return_counts=True)
    total = len(y_final)
    for cls_idx, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"{classes[cls_idx]}: {count} örnek ({percentage:.1f}%)")
    
    print(f"\nToplam: {total} örnek")
    print(f"Dengesizlik oranı: {max(counts)/min(counts):.1f}:1 ")
    
    X_balanced_file = os.path.join(DATA_PATH, "X_balanced.npy")
    y_balanced_file = os.path.join(DATA_PATH, "y_balanced.npy")
    
    np.save(X_balanced_file, X_final)
    np.save(y_balanced_file, y_final)
    
    print(f"\n Dengelenmiş veri kaydedildi:")
    print(f"   {X_balanced_file}")
    print(f"   {y_balanced_file}")

    return True

if __name__ == "__main__":
    success = balance_dataset()
    if success:
        print("\n🎉 Veri dengeleme başarılı!")
        print(" Sonraki adım: python -m src.audio_processing.train_model_balanced")
    else:
        print("\n Veri dengeleme başarısız!")
