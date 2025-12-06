"""Notification manager module.

This module provides notification capabilities including
push notifications, SMS, and MQTT messaging.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import json


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


class BaseNotifier(ABC):
    """Abstract base class for notification services."""
    
    @abstractmethod
    def send(self, event: SecurityEvent) -> bool:
        """Send notification for security event.
        
        Args:
            event: Security event to notify about
            
        Returns:
            True if notification sent successfully
        """
        pass


class FirebaseNotifier(BaseNotifier):
    """Push notifications via Firebase Cloud Messaging."""
    
    def __init__(
        self,
        credentials_path: str,
        topic: str = "security_alerts",
    ):
        """Initialize Firebase notifier.
        
        Args:
            credentials_path: Path to Firebase credentials JSON
            topic: FCM topic to send notifications to
        """
        self.credentials_path = credentials_path
        self.topic = topic
        self.initialized = False
        
        self._init_firebase()
    
    def _init_firebase(self) -> None:
        """Initialize Firebase Admin SDK."""
        try:
            import firebase_admin
            from firebase_admin import credentials, messaging
            
            cred = credentials.Certificate(self.credentials_path)
            firebase_admin.initialize_app(cred)
            self.initialized = True
        except ImportError:
            print("Warning: firebase-admin not installed")
        except Exception as e:
            print(f"Warning: Failed to initialize Firebase: {e}")
    
    def send(self, event: SecurityEvent) -> bool:
        """Send push notification via Firebase."""
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
                data=event.to_dict(),
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
        """Initialize Twilio notifier.
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: Sender phone number
            to_numbers: List of recipient phone numbers
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers
        self.client = None
        
        self._init_twilio()
    
    def _init_twilio(self) -> None:
        """Initialize Twilio client."""
        try:
            from twilio.rest import Client
            self.client = Client(self.account_sid, self.auth_token)
        except ImportError:
            print("Warning: twilio not installed")
    
    def send(self, event: SecurityEvent) -> bool:
        """Send SMS notification via Twilio."""
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
        """Initialize MQTT notifier.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            topic: Topic to publish alerts to
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.client = None
        
        self._init_mqtt()
    
    def _init_mqtt(self) -> None:
        """Initialize MQTT client."""
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
        """Publish security event to MQTT topic."""
        if self.client is None:
            return False
        
        try:
            result = self.client.publish(
                self.topic,
                event.to_json(),
                qos=1,
            )
            return result.rc == 0
        except Exception as e:
            print(f"Failed to publish MQTT message: {e}")
            return False


class NotificationManager:
    """Manages multiple notification channels."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.notifiers: List[BaseNotifier] = []
    
    def add_notifier(self, notifier: BaseNotifier) -> None:
        """Add a notification channel.
        
        Args:
            notifier: Notifier instance to add
        """
        self.notifiers.append(notifier)
    
    def notify(self, event: SecurityEvent) -> Dict[str, bool]:
        """Send notification to all channels.
        
        Args:
            event: Security event to notify about
            
        Returns:
            Dictionary of notifier type to success status
        """
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
        """Send notifications for multiple events.
        
        Args:
            events: List of security events
        """
        for event in events:
            self.notify(event)
