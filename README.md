# IoT Personal Home Security System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4%2F5-red.svg)](https://www.raspberrypi.org/)

An intelligent home security system combining **face recognition** and **environmental sound classification** for comprehensive threat detection. Designed to run on Raspberry Pi for edge inference.

## 🎯 Features

- **Face Detection & Recognition** - Identify known vs unknown individuals
- **Sound Classification** - Detect security-relevant sounds (glass breaking, door knocks, alarms)
- **Motion Detection** - PIR sensor integration for motion-triggered alerts
- **Multi-channel Alerts** - Local alarm, push notifications, SMS, MQTT
- **Edge Deployment** - Optimized for Raspberry Pi with TensorFlow Lite
- **Modular Architecture** - Easy to extend and customize

## 🏗️ Project Structure

```
IoT-Personal-Home-Security/
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   └── logging.yaml           # Logging configuration
│
├── data/                       # Data directories
│   ├── raw/faces/             # Face images for training
│   ├── processed/             # Preprocessed data
│   └── models/                # Trained models
│
├── docs/                       # Documentation
│   ├── architecture.md        # System architecture
│   ├── hardware_setup.md      # Hardware wiring guide
│   ├── model_training.md      # Training guide
│   └── deployment.md          # Deployment guide
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_face_detection_analysis.ipynb
│   ├── 02_face_recognition_training.ipynb
│   ├── 03_sound_classification_training.ipynb
│   └── 04_model_optimization.ipynb
│
├── src/iot_home_security/      # Main Python package
│   ├── face/                  # Face detection & recognition
│   ├── audio/                 # Sound classification
│   ├── sensors/               # Sensor interfaces
│   └── alerts/                # Notification system
│
├── raspberry_pi/               # Raspberry Pi deployment
│   ├── main.py               # Main entry point
│   ├── config/               # RPi-specific config
│   ├── scripts/              # Installation scripts
│   └── systemd/              # Service files
│
├── tests/                      # Unit tests
│
├── ESC-50-master/              # Sound classification dataset
│
└── documents/                  # Project documents
```

## 🚀 Quick Start

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/IoT-Personal-Home-Security.git
cd IoT-Personal-Home-Security

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Training Models

1. **Face Recognition:**
   - Add face images to `data/raw/faces/known/<person_name>/`
   - Run `notebooks/02_face_recognition_training.ipynb`

2. **Sound Classification:**
   - Download ESC-50 audio files to `ESC-50-master/audio/`
   - Run `notebooks/03_sound_classification_training.ipynb`

3. **Optimize for Raspberry Pi:**
   - Run `notebooks/04_model_optimization.ipynb`
   - Export TFLite models to `data/models/`

### Raspberry Pi Deployment

```bash
# Transfer to Raspberry Pi
rsync -avz . pi@raspberrypi:/home/pi/security/

# On Raspberry Pi
cd /home/pi/security
chmod +x raspberry_pi/scripts/install.sh
./raspberry_pi/scripts/install.sh

# Start the service
sudo systemctl start security-system
```

See [Deployment Guide](docs/deployment.md) for detailed instructions.

## 🔧 Hardware Requirements

### Minimum
- Raspberry Pi 4 (2GB RAM)
- Pi Camera Module v2
- USB Microphone
- MicroSD Card (32GB)

### Recommended
- Raspberry Pi 4/5 (4GB+ RAM)
- Pi Camera Module v3
- USB/I2S MEMS Microphone
- PIR Motion Sensor (HC-SR501)
- Buzzer & LED indicators
- Coral USB Accelerator (for faster inference)

See [Hardware Setup Guide](docs/hardware_setup.md) for wiring diagrams.

## 📊 Performance

| Model | Accuracy | Inference Time (RPi 4) |
|-------|----------|------------------------|
| Face Detection (Haar Cascade) | 85% | ~30ms |
| Face Recognition (FaceNet) | 92% | ~80ms |
| Sound Classification (CNN) | 75% | ~150ms |

*Performance varies based on configuration and optimization settings.*

## 🔒 Security Events Detected

### Face Recognition
- ✅ Known person detected (welcome)
- ⚠️ Unknown person detected (alert)
- 📸 Face logged with timestamp

### Sound Classification
- 🔊 Glass breaking
- 🚪 Door knock
- 🐕 Dog barking
- 🚨 Sirens
- 👶 Baby crying
- 👣 Footsteps

## 📝 Configuration

Main configuration in `config/config.yaml`:

```yaml
face_detection:
  enabled: true
  model: "haar_cascade"
  confidence_threshold: 0.8

sound_classification:
  enabled: true
  confidence_threshold: 0.7
  target_classes:
    - "glass_breaking"
    - "door_wood_knock"
    - "siren"

alerts:
  local_alarm:
    enabled: true
    gpio_pin: 18
  push_notifications:
    enabled: false
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_face_detection.py
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Hardware Setup Guide](docs/hardware_setup.md)
- [Model Training Guide](docs/model_training.md)
- [Deployment Guide](docs/deployment.md)
- [Raspberry Pi README](raspberry_pi/README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) for environmental sound classification
- OpenCV for computer vision capabilities
- TensorFlow/TensorFlow Lite for machine learning
- Raspberry Pi Foundation for edge computing platform

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/IoT-Personal-Home-Security](https://github.com/yourusername/IoT-Personal-Home-Security)
