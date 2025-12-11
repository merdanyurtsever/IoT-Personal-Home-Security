"""Unified alerts module.

Combines local alarms (buzzer, LED) and remote notifications (Firebase, Twilio, MQTT).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional
import json
import threading
import time


# =============================================================================
# Security Event
# =============================================================================

@dataclass
class SecurityEvent:
    """Represents a security event to be notified."""

    event_type: str  # face_detected, sound_detected, motion_detected
    timestamp: datetime
    confidence: float
    details: Dict
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "details": self.details,
            "image_path": self.image_path,
            "audio_path": self.audio_path,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# =============================================================================
# Base Classes
# =============================================================================

class BaseAlarm(ABC):
    """Abstract base class for local alarms."""

    @abstractmethod
    def activate(self, duration: Optional[float] = None) -> None:
        pass

    @abstractmethod
    def deactivate(self) -> None:
        pass

    @abstractmethod
    def is_active(self) -> bool:
        pass


class BaseNotifier(ABC):
    """Abstract base class for notification services."""

    @abstractmethod
    def send(self, event: SecurityEvent) -> bool:
        pass


# =============================================================================
# Local Alarms
# =============================================================================

class GPIOBuzzer(BaseAlarm):
    """Buzzer alarm using Raspberry Pi GPIO."""

    def __init__(self, pin: int = 18):
        self.pin = pin
        self.buzzer = None
        self._active = False
        self._timer = None
        self._setup_gpio()

    def _setup_gpio(self) -> None:
        try:
            from gpiozero import Buzzer
            self.buzzer = Buzzer(self.pin)
        except ImportError:
            print(f"Warning: gpiozero not available. Buzzer on pin {self.pin} simulated.")

    def activate(self, duration: Optional[float] = None) -> None:
        self._active = True
        if self.buzzer is not None:
            self.buzzer.on()
        else:
            print("🔊 BUZZER ON (simulated)")
        if duration is not None:
            self._timer = threading.Timer(duration, self.deactivate)
            self._timer.start()

    def deactivate(self) -> None:
        self._active = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.buzzer is not None:
            self.buzzer.off()
        else:
            print("🔇 BUZZER OFF (simulated)")

    def is_active(self) -> bool:
        return self._active

    def beep(self, on_time: float = 0.5, off_time: float = 0.5, count: int = 3) -> None:
        for _ in range(count):
            self.activate()
            time.sleep(on_time)
            self.deactivate()
            time.sleep(off_time)


class GPIOLed(BaseAlarm):
    """LED alarm indicator using Raspberry Pi GPIO."""

    def __init__(self, pin: int = 25):
        self.pin = pin
        self.led = None
        self._active = False
        self._setup_gpio()

    def _setup_gpio(self) -> None:
        try:
            from gpiozero import LED
            self.led = LED(self.pin)
        except ImportError:
            print(f"Warning: gpiozero not available. LED on pin {self.pin} simulated.")

    def activate(self, duration: Optional[float] = None) -> None:
        self._active = True
        if self.led is not None:
            self.led.on()
        else:
            print("💡 LED ON (simulated)")
        if duration is not None:
            threading.Timer(duration, self.deactivate).start()

    def deactivate(self) -> None:
        self._active = False
        if self.led is not None:
            self.led.off()
        else:
            print("⚫ LED OFF (simulated)")

    def is_active(self) -> bool:
        return self._active

    def blink(self, on_time: float = 0.5, off_time: float = 0.5) -> None:
        if self.led is not None:
            self.led.blink(on_time=on_time, off_time=off_time)
        else:
            print("💡 LED BLINKING (simulated)")


class MockAlarm(BaseAlarm):
    """Mock alarm for testing without hardware."""

    def __init__(self, name: str = "MockAlarm"):
        self.name = name
        self._active = False

    def activate(self, duration: Optional[float] = None) -> None:
        self._active = True
        print(f"🚨 {self.name} ACTIVATED")
        if duration is not None:
            threading.Timer(duration, self.deactivate).start()

    def deactivate(self) -> None:
        self._active = False
        print(f"✓ {self.name} DEACTIVATED")

    def is_active(self) -> bool:
        return self._active


class LocalAlarm:
    """Unified local alarm interface."""

    def __init__(
        self,
        buzzer_pin: Optional[int] = 18,
        led_pin: Optional[int] = 25,
        default_duration: float = 10.0,
        use_mock: bool = False,
    ):
        self.default_duration = default_duration
        self.alarms: List[BaseAlarm] = []

        if use_mock:
            if buzzer_pin is not None:
                self.alarms.append(MockAlarm("Buzzer"))
            if led_pin is not None:
                self.alarms.append(MockAlarm("LED"))
        else:
            try:
                if buzzer_pin is not None:
                    self.alarms.append(GPIOBuzzer(buzzer_pin))
                if led_pin is not None:
                    self.alarms.append(GPIOLed(led_pin))
            except Exception:
                print("Falling back to mock alarms")
                if buzzer_pin is not None:
                    self.alarms.append(MockAlarm("Buzzer"))
                if led_pin is not None:
                    self.alarms.append(MockAlarm("LED"))

    def trigger(self, duration: Optional[float] = None) -> None:
        if duration is None:
            duration = self.default_duration
        for alarm in self.alarms:
            alarm.activate(duration)

    def stop(self) -> None:
        for alarm in self.alarms:
            alarm.deactivate()

    def is_active(self) -> bool:
        return any(alarm.is_active() for alarm in self.alarms)


# =============================================================================
# Remote Notifiers
# =============================================================================

class FirebaseNotifier(BaseNotifier):
    """Push notifications via Firebase Cloud Messaging."""

    def __init__(self, credentials_path: str, topic: str = "security_alerts"):
        self.credentials_path = credentials_path
        self.topic = topic
        self.initialized = False
        self._init_firebase()

    def _init_firebase(self) -> None:
        try:
            import firebase_admin
            from firebase_admin import credentials

            cred = credentials.Certificate(self.credentials_path)
            firebase_admin.initialize_app(cred)
            self.initialized = True
        except ImportError:
            print("Warning: firebase-admin not installed")
        except Exception as e:
            print(f"Warning: Failed to initialize Firebase: {e}")

    def send(self, event: SecurityEvent) -> bool:
        if not self.initialized:
            return False
        try:
            from firebase_admin import messaging

            message = messaging.Message(
                notification=messaging.Notification(
                    title=f"Security Alert: {event.event_type}",
                    body=f"Detected at {event.timestamp.strftime('%H:%M:%S')} "
                    f"with {event.confidence:.0%} confidence",
                ),
                data={k: str(v) for k, v in event.to_dict().items()},
                topic=self.topic,
            )
            response = messaging.send(message)
            print(f"Firebase notification sent: {response}")
            return True
        except Exception as e:
            print(f"Failed to send Firebase notification: {e}")
            return False


class TwilioNotifier(BaseNotifier):
    """SMS notifications via Twilio."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        to_numbers: List[str],
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers
        self.client = None
        self._init_twilio()

    def _init_twilio(self) -> None:
        try:
            from twilio.rest import Client

            self.client = Client(self.account_sid, self.auth_token)
        except ImportError:
            print("Warning: twilio not installed")

    def send(self, event: SecurityEvent) -> bool:
        if self.client is None:
            return False
        message_body = (
            f"🚨 Security Alert: {event.event_type}\n"
            f"Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Confidence: {event.confidence:.0%}"
        )
        success = True
        for number in self.to_numbers:
            try:
                self.client.messages.create(
                    body=message_body,
                    from_=self.from_number,
                    to=number,
                )
            except Exception as e:
                print(f"Failed to send SMS to {number}: {e}")
                success = False
        return success


class MQTTNotifier(BaseNotifier):
    """Notifications via MQTT for IoT hub integration."""

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "home/security/alerts",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.client = None
        self._init_mqtt()

    def _init_mqtt(self) -> None:
        try:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client()
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except ImportError:
            print("Warning: paho-mqtt not installed")
        except Exception as e:
            print(f"Warning: Failed to connect to MQTT broker: {e}")

    def send(self, event: SecurityEvent) -> bool:
        if self.client is None:
            return False
        try:
            result = self.client.publish(self.topic, event.to_json(), qos=1)
            return result.rc == 0
        except Exception as e:
            print(f"Failed to publish MQTT message: {e}")
            return False


# =============================================================================
# Notification Manager
# =============================================================================

class NotificationManager:
    """Manages multiple notification channels."""

    def __init__(self):
        self.notifiers: List[BaseNotifier] = []

    def add_notifier(self, notifier: BaseNotifier) -> None:
        self.notifiers.append(notifier)

    def notify(self, event: SecurityEvent) -> Dict[str, bool]:
        results = {}
        for notifier in self.notifiers:
            notifier_name = notifier.__class__.__name__
            try:
                results[notifier_name] = notifier.send(event)
            except Exception as e:
                print(f"Error in {notifier_name}: {e}")
                results[notifier_name] = False
        return results

    def notify_all(self, events: List[SecurityEvent]) -> None:
        for event in events:
            self.notify(event)
