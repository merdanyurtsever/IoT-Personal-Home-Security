# Project Architecture

This document describes the architecture of the IoT Personal Home Security system.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEVELOPMENT ENVIRONMENT                              │
│                    (Training & Model Development)                            │
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐    ┌──────────────────┐  │
│  │   Face Detection   │    │  Sound             │    │   Model          │  │
│  │   & Recognition    │    │  Classification    │    │   Optimization   │  │
│  │   Training         │    │  Training          │    │   & Export       │  │
│  └─────────┬──────────┘    └─────────┬──────────┘    └────────┬─────────┘  │
│            │                         │                         │            │
│            └─────────────────────────┼─────────────────────────┘            │
│                                      │                                       │
│                                      ▼                                       │
│                    ┌─────────────────────────────────┐                      │
│                    │   Optimized Models (.tflite)    │                      │
│                    │   - Face Detection              │                      │
│                    │   - Face Recognition            │                      │
│                    │   - Sound Classification        │                      │
│                    └─────────────────────────────────┘                      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   │  Model Deployment
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RASPBERRY PI EDGE DEVICE                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Sensor Layer                                  │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐ │   │
│  │  │   Camera   │   │ Microphone │   │ PIR Motion │   │   Other    │ │   │
│  │  │   Module   │   │   Module   │   │   Sensor   │   │  Sensors   │ │   │
│  │  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘ │   │
│  └────────┼────────────────┼────────────────┼────────────────┼────────┘   │
│           │                │                │                │            │
│           ▼                ▼                ▼                ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Processing Layer                                 │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │   │
│  │  │ Face Detection │  │    Sound       │  │  Motion Detection      │ │   │
│  │  │ & Recognition  │  │ Classification │  │  & Event Filtering     │ │   │
│  │  │   (TFLite)     │  │   (TFLite)     │  │                        │ │   │
│  │  └───────┬────────┘  └───────┬────────┘  └───────────┬────────────┘ │   │
│  └──────────┼───────────────────┼───────────────────────┼──────────────┘   │
│             │                   │                       │                  │
│             └───────────────────┼───────────────────────┘                  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Alert Layer                                     │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐ │   │
│  │  │   Local    │   │    Push    │   │    SMS     │   │    MQTT    │ │   │
│  │  │   Alarm    │   │   Notif.   │   │   Alerts   │   │  Messages  │ │   │
│  │  └────────────┘   └────────────┘   └────────────┘   └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules (`src/iot_home_security/`)

| Module | Description |
|--------|-------------|
| `face/` | Face detection and recognition |
| `audio/` | Sound classification and audio processing |
| `sensors/` | Hardware sensor interfaces |
| `alerts/` | Notification and alarm systems |

### Face Module

```
face/
├── __init__.py
├── detector.py      # Face detection with multiple backends
├── recognizer.py    # Face embedding and matching
└── utils.py         # Face processing utilities
```

**Key Classes:**
- `FaceDetector`: Detects faces in images using configurable backends
- `FaceRecognizer`: Identifies faces using embedding similarity
- `DetectedFace`: Data class for detection results

### Audio Module

```
audio/
├── __init__.py
├── classifier.py    # Sound classification
├── preprocessing.py # Audio preprocessing
└── features.py      # Feature extraction (MFCC, Mel spectrogram)
```

**Key Classes:**
- `SoundClassifier`: Classifies audio for security-relevant sounds
- `AudioPreprocessor`: Handles audio loading, resampling, normalization
- `FeatureExtractor`: Extracts MFCC and Mel spectrogram features

### Sensors Module

```
sensors/
├── __init__.py
├── camera.py        # Camera interface (OpenCV, PiCamera)
├── microphone.py    # Microphone interface (PyAudio)
└── motion.py        # PIR motion sensor (GPIO)
```

**Key Classes:**
- `CameraInterface`: Unified camera access for dev and RPi
- `MicrophoneInterface`: Audio capture interface
- `MotionSensor`: PIR sensor with debouncing

### Alerts Module

```
alerts/
├── __init__.py
├── notifications.py # Push, SMS, MQTT notifications
└── local_alarm.py   # Buzzer and LED alarms
```

**Key Classes:**
- `NotificationManager`: Manages multiple notification channels
- `LocalAlarm`: Controls buzzer and LED indicators

## Data Flow

### 1. Video Processing Pipeline

```
Camera Frame → Face Detection → Face Cropping → Face Recognition → Identity
      │              │                                    │
      │              └── Draw Bounding Boxes              │
      │                                                   │
      └── Motion Trigger ◄──────────────────────── Unknown Face Alert
```

### 2. Audio Processing Pipeline

```
Microphone → Audio Buffer → Preprocessing → Feature Extraction → Classification
      │                                              │
      │                                              ▼
      │                                     Security Sound Detected?
      │                                              │
      └── Continuous Monitoring ◄──────────── Trigger Alert
```

### 3. Alert Pipeline

```
Security Event → Event Classification → Alert Priority
      │                                       │
      │                    ┌──────────────────┼──────────────────┐
      │                    ▼                  ▼                  ▼
      │              Local Alarm       Push Notification    Log Event
      │                    │                  │                  │
      └── Cooldown ◄───────┴──────────────────┴──────────────────┘
```

## Configuration

Configuration is managed through YAML files:

- `config/config.yaml`: Main configuration
- `config/logging.yaml`: Logging configuration
- `raspberry_pi/config/config.yaml`: RPi-specific overrides

## Deployment Architecture

### Development to Production Flow

```
1. Train models on development machine (GPU)
   │
   ▼
2. Optimize and convert to TFLite
   │
   ▼
3. Deploy to Raspberry Pi
   │
   ▼
4. Run as systemd service
```

### Raspberry Pi Service Architecture

```
systemd service
      │
      ▼
main.py (SecuritySystem)
      │
      ├── Camera Thread
      │
      ├── Audio Thread
      │
      ├── Sensor Monitoring
      │
      └── Alert Handler
```
