"""
Dengelenmiş Veri ile SVM Model Eğitimi
Her sınıftan 125 örnek aldım çünkü diğer türlü kadın çığlığı da noise olarak algılanıyordu- 
"""
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models")

def load_balanced_dataset():
    """Dengelenmiş veri setini yüklüyom"""
    print(" Dengelenmiş dataset yükleniyor:", DATA_PATH)
    
    X_file = os.path.join(DATA_PATH, "X_balanced.npy")
    y_file = os.path.join(DATA_PATH, "y_balanced.npy")
    
    if not os.path.exists(X_file) or not os.path.exists(y_file):
        print(" Dengelenmiş veri dosyaları bulunamadı!")
        print("Önce şu komutu çalıştırın: python -m src.audio_processing.balance_dataset")
        return np.array([]), np.array([]), []
    
    try:
        X = np.load(X_file)
        y = np.load(y_file)
        
        print(f" X_balanced.npy yüklendi: {X.shape}")
        print(f" y_balanced.npy yüklendi: {y.shape}")
        
        class_names = ["man_scream", "woman_scream", "noise"]
        
        unique_classes, counts = np.unique(y, return_counts=True)
        print("\n Sınıf Dağılımı:")
        for cls, count in zip(unique_classes, counts):
            class_name = class_names[int(cls)]
            percentage = (count / len(y)) * 100
            print(f"   {cls} ({class_name}): {count} örnek ({percentage:.1f}%)")
        
        return X, y, class_names
        
    except Exception as e:
        print(f" Dosya yükleme hatası: {e}")
        return np.array([]), np.array([]), []

def main():
    print(" Dengelenmiş SVM Model Eğitimi Başlıyor...")
    print("=" * 60)

    X, y, class_names = load_balanced_dataset()

    if len(X) == 0:
        print(" HATA: Dataset boş!")
        return

    print(f"\n Dengelenmiş Dataset:")
    print(f"   Örnekler: {X.shape[0]}")
    print(f"   Özellikler: {X.shape[1]}")
    print(f"   Sınıflar: {class_names}")

    # Özellik normalleştirme uygulıcaz
    print("\n Özellik normalleştirme uygulanıyor...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #train-test ksımımız
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n Eğitim/Test Bölünmesi:")
    print(f"   Eğitim: {X_train.shape[0]} örnek")
    print(f"   Test: {X_test.shape[0]} örnek")
    
    print(f"\n Test setindeki sınıf dağılımı:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for cls, count in zip(unique_test, counts_test):
        print(f"   {class_names[int(cls)]}: {count} örnek")

    print("\n⚡ SVM Hiperparametre Optimizasyonu Başlıyor...")

    # Dengelenmiş veri için daha geniş parametre aralığı uygulrızki en iyi sonucu bulalım
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    #cllas_weigth
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,  
        scoring='f1_macro',  
        n_jobs=-1,
        verbose=1
    )
    
    print(" En iyi parametreler aranıyor...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n En iyi parametreler: {grid_search.best_params_}")
    print(f" En iyi CV F1-skoru: {grid_search.best_score_:.3f}")

    clf = grid_search.best_estimator_
    
    print(f"\n Kullanılan SVM Parametreleri:")
    print(f"   Kernel: {clf.kernel}")
    print(f"   C: {clf.C}")
    print(f"   Gamma: {clf.gamma}")

    # Test 
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n SVM Model Performansı:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   F1-Score: {grid_search.best_score_:.3f}")

    print(f"\n Sınıf Bazında Performans:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_predictions = y_pred[class_mask]
            class_accuracy = np.sum(class_predictions == i) / len(class_predictions)
            test_count = np.sum(class_mask)
            predicted_count = np.sum(y_pred == i)
            print(f"   {class_name}:")
            print(f"     Doğruluk: {class_accuracy:.3f}")
            print(f"     Test örnekleri: {test_count}")
            print(f"     Tahmin edilen: {predicted_count}")
    
    print(f"\n Detaylı Sınıflandırma Raporu:")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    print(f"\n Confusion Matrix:")
    print("=" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print("Gerçek \\ Tahmin\t", "\t".join(class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]}\t\t", "\t".join(map(str, row)))

    print(f"\n SVM Model kaydediliyor...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_file = os.path.join(MODEL_PATH, "svm_audio_model.pkl")
    
    model_data = {
        'model': clf,
        'scaler': scaler,
        'classes': class_names,
        'class_mapping': {i: name for i, name in enumerate(class_names)},
        'feature_size': X_scaled.shape[1],
        'accuracy': accuracy,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'class_distribution': {class_names[i]: np.sum(y == i) for i in range(len(class_names))},
        'training_info': {
            'total_samples': len(y),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'balanced': True,
            'feature_normalized': True
        }
    }
    
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n SVM Model başarıyla kaydedildi!")
    print(f"    Dosya: {model_file}")
    print(f"    Test Accuracy: {accuracy:.3f}")
    print(f"    CV F1-Score: {grid_search.best_score_:.3f}")
    print(f"    Sınıflar: {class_names}")
    print(f"    Veri dengelendi: Evet (125'er örnek)")
    print(f"    En iyi kernel: {clf.kernel}")
    
    # Kadın çığlığı tespiti için özel kontrol aç
    woman_scream_mask = y_test == 1  
    if np.sum(woman_scream_mask) > 0:
        woman_predictions = y_pred[woman_scream_mask]
        woman_accuracy = np.sum(woman_predictions == 1) / len(woman_predictions)
        woman_test_count = np.sum(woman_scream_mask)
        print(f"\n KADIN ÇIĞLIĞI TESPİT PERFORMANSI:")
        print(f"    Doğruluk: {woman_accuracy:.3f} ({woman_accuracy*100:.1f}%)")
        print(f"    Test örnekleri: {woman_test_count}")
    
    print(f"\n Model eğitimi tamamlandı!")
    print(f" Test etmek için: python -m src.audio_processing.realtime_test")

if __name__ == "__main__":
    main()