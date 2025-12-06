# Hardware Setup Guide

This guide covers the hardware components and wiring for the IoT Personal Home Security system on Raspberry Pi.

## Required Components

### Core Components

| Component | Model/Specification | Purpose |
|-----------|---------------------|---------|
| Raspberry Pi | Pi 4 (2GB+) or Pi 5 | Main processing unit |
| Power Supply | 5V 3A USB-C | Powers the Pi |
| MicroSD Card | 32GB+ Class 10 | Storage |
| Pi Camera | V2 or V3 Module | Face detection |
| USB Microphone | Any USB mic | Sound classification |

### Sensors & Actuators

| Component | Model | GPIO Pin (BCM) |
|-----------|-------|----------------|
| PIR Motion Sensor | HC-SR501 | GPIO 17 |
| Buzzer | Active 5V Buzzer | GPIO 18 |
| Alert LED | 5mm Red LED | GPIO 25 |
| Status LED | 5mm Green LED | GPIO 24 |

### Optional Components

- Coral USB Accelerator (for faster inference)
- I2S MEMS Microphone (better audio quality)
- Door/Window magnetic sensors
- Relay module for siren integration

## Wiring Diagram

### GPIO Pinout Reference

```
                    Raspberry Pi GPIO
    ┌─────────────────────────────────────────────┐
    │    3.3V (1)  ○ ○  (2)  5V                   │
    │   GPIO2 (3)  ○ ○  (4)  5V                   │
    │   GPIO3 (5)  ○ ○  (6)  GND                  │
    │   GPIO4 (7)  ○ ○  (8)  GPIO14               │
    │     GND (9)  ○ ○  (10) GPIO15               │
    │  GPIO17 (11) ○ ○  (12) GPIO18               │ ◄── PIR & Buzzer
    │  GPIO27 (13) ○ ○  (14) GND                  │
    │  GPIO22 (15) ○ ○  (16) GPIO23               │
    │    3.3V (17) ○ ○  (18) GPIO24               │ ◄── Status LED
    │  GPIO10 (19) ○ ○  (20) GND                  │
    │   GPIO9 (21) ○ ○  (22) GPIO25               │ ◄── Alert LED
    │  GPIO11 (23) ○ ○  (24) GPIO8                │
    │     GND (25) ○ ○  (26) GPIO7                │
    │   GPIO0 (27) ○ ○  (28) GPIO1                │
    │   GPIO5 (29) ○ ○  (30) GND                  │
    │   GPIO6 (31) ○ ○  (32) GPIO12               │
    │  GPIO13 (33) ○ ○  (34) GND                  │
    │  GPIO19 (35) ○ ○  (36) GPIO16               │
    │  GPIO26 (37) ○ ○  (38) GPIO20               │
    │     GND (39) ○ ○  (40) GPIO21               │
    └─────────────────────────────────────────────┘
```

### Component Wiring

#### PIR Motion Sensor (HC-SR501)

```
HC-SR501          Raspberry Pi
┌─────────┐       
│  VCC    │ ────► Pin 2 (5V)
│  OUT    │ ────► Pin 11 (GPIO17)
│  GND    │ ────► Pin 6 (GND)
└─────────┘
```

**Adjustment:**
- Sensitivity: Adjust potentiometer for detection range
- Time Delay: Set to minimum for faster response

#### Active Buzzer

```
Buzzer            Raspberry Pi
┌─────────┐       
│    +    │ ────► Pin 12 (GPIO18) via 100Ω resistor
│    -    │ ────► Pin 6 (GND)
└─────────┘
```

#### LEDs

```
Alert LED (Red)   Raspberry Pi
┌─────────┐
│  Anode  │ ────► Pin 22 (GPIO25) via 220Ω resistor
│ Cathode │ ────► Pin 25 (GND)
└─────────┘

Status LED (Green)
┌─────────┐
│  Anode  │ ────► Pin 18 (GPIO24) via 220Ω resistor
│ Cathode │ ────► Pin 25 (GND)
└─────────┘
```

#### Pi Camera

Connect the camera ribbon cable to the camera port (CSI) on the Raspberry Pi:

1. Lift the camera port latch
2. Insert ribbon cable (blue side facing the USB ports)
3. Push latch down to secure

### Complete Wiring Diagram

```
                                    ┌──────────────────────┐
                                    │    Raspberry Pi 4    │
                                    │                      │
    ┌─────────────────┐            │   ┌──────────────┐  │
    │   PIR Sensor    │            │   │   Camera     │  │
    │   HC-SR501      │            │   │   Module     │  │
    │  ┌───┬───┬───┐  │            │   └──────────────┘  │
    │  │VCC│OUT│GND│  │            │                      │
    └──┼───┼───┼───┼──┘            │                      │
       │   │   │                   │                      │
       │   │   └───────────────────┼──► GND (Pin 6)       │
       │   └───────────────────────┼──► GPIO17 (Pin 11)   │
       └───────────────────────────┼──► 5V (Pin 2)        │
                                   │                      │
    ┌─────────────────┐            │                      │
    │   USB Mic       │────────────┼──► USB Port          │
    └─────────────────┘            │                      │
                                   │                      │
    ┌─────────────────┐            │                      │
    │   Buzzer        │            │                      │
    │  ┌─────┬─────┐  │            │                      │
    │  │  +  │  -  │  │            │                      │
    └──┼─────┼─────┼──┘            │                      │
       │     │                     │                      │
       │     └─────────────────────┼──► GND               │
       └──[100Ω]───────────────────┼──► GPIO18 (Pin 12)   │
                                   │                      │
    ┌─────────────────┐            │                      │
    │   Alert LED     │            │                      │
    │   (Red)         │            │                      │
    └──[220Ω]─────────┼────────────┼──► GPIO25 (Pin 22)   │
       │              │            │                      │
       └──────────────┼────────────┼──► GND               │
                                   │                      │
    ┌─────────────────┐            │                      │
    │   Status LED    │            │                      │
    │   (Green)       │            │                      │
    └──[220Ω]─────────┼────────────┼──► GPIO24 (Pin 18)   │
       │              │            │                      │
       └──────────────┼────────────┼──► GND               │
                                   │                      │
                                   └──────────────────────┘
```

## Camera Setup

### Enable Camera Interface

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
# Reboot
```

### Test Camera

```bash
# Test with libcamera
libcamera-hello

# Capture test image
libcamera-still -o test.jpg
```

## Microphone Setup

### List Audio Devices

```bash
arecord -l
```

### Test Microphone

```bash
# Record 5 seconds of audio
arecord -d 5 -f S16_LE -r 44100 test.wav

# Play back
aplay test.wav
```

### Adjust Volume

```bash
alsamixer
# Press F4 for capture devices
# Adjust mic gain
```

## Troubleshooting

### Camera Not Detected

```bash
# Check if camera is detected
vcgencmd get_camera

# Check camera connection
libcamera-hello --list-cameras
```

### GPIO Permission Issues

```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Logout and login again
```

### Audio Issues

```bash
# Install pulseaudio
sudo apt install pulseaudio

# List audio sources
pactl list sources
```
