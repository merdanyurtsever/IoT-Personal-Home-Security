# Ses Veri Klasörleri

Bu klasör ses sınıflandırma modeli için ses verilerini içerir.

## 📁 Klasör Yapısı

### 🔊 scream/ 
Acil durum ve çığlık sesleri:
- Çığlık sesleri 
- Yardım çağrıları
- Bebek ağlaması
- Bağırma sesleri

**Desteklenen formatlar:** `.wav`, `.mp3`, `.m4a`, `.flac`

### 🏠 normal/
Normal ev sesleri:
- Kapı çalma
- Ayak sesleri
- Normal konuşma
- Köpek havlaması (normal)
- Günlük ev sesleri

**Desteklenen formatlar:** `.wav`, `.mp3`, `.m4a`, `.flac`

### ⚠️ noise/
Gürültü ve tehlike sesleri:
- Cam kırılması
- Siren sesleri
- Silah sesi
- Araç kornası
- Metal çarpışma
- Patlama sesleri

**Desteklenen formatlar:** `.wav`, `.mp3`, `.m4a`, `.flac`

## 📋 Veri Yükleme Rehberi

1. Ses dosyalarınızı uygun kategorilere yerleştirin
2. Dosya isimlerinde Türkçe karakter kullanmaktan kaçının
3. Ses dosyaları 3-10 saniye arası olmalı
4. Her kategori için en az 20-30 örnek bulunmalı
5. Yüksek kaliteli ses dosyaları tercih edin (16kHz+)

## 🔧 Önerilen Dosya İsimlendirme

```
scream/
├── scream_001.wav
├── scream_002.wav
├── baby_cry_001.wav
└── shouting_001.wav

normal/
├── door_knock_001.wav
├── footsteps_001.wav  
├── conversation_001.wav
└── dog_bark_normal_001.wav

noise/
├── glass_break_001.wav
├── siren_001.wav
├── gunshot_001.wav
└── car_horn_001.wav
```

## ⚡ Hızlı Başlangıç

Ses dosyalarınızı yükledikten sonra:

1. `notebooks/03_sound_classification_training.ipynb` notebook'unu açın
2. Veri yollarını güncelleyin  
3. Model eğitimini çalıştırın
4. Test edin ve dağıtın
