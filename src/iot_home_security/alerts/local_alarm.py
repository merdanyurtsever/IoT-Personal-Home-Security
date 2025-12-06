"""Local alarm module.

This module provides local alarm functionality using
buzzer, speaker, or LED indicators on Raspberry Pi.
"""

from abc import ABC, abstractmethod
from typing import Optional
import time
import threading


class BaseAlarm(ABC):
    """Abstract base class for local alarms."""
    
    @abstractmethod
    def activate(self, duration: Optional[float] = None) -> None:
        """Activate the alarm.
        
        Args:
            duration: Duration in seconds (None for indefinite)
        """
        pass
    
    @abstractmethod
    def deactivate(self) -> None:
        """Deactivate the alarm."""
        pass
    
    @abstractmethod
    def is_active(self) -> bool:
        """Check if alarm is currently active."""
        pass


class GPIOBuzzer(BaseAlarm):
    """Buzzer alarm using Raspberry Pi GPIO."""
    
    def __init__(self, pin: int = 18):
        """Initialize GPIO buzzer.
        
        Args:
            pin: GPIO pin number (BCM numbering)
        """
        self.pin = pin
        self.buzzer = None
        self._active = False
        self._timer = None
        
        self._setup_gpio()
    
    def _setup_gpio(self) -> None:
        """Setup GPIO for buzzer."""
        try:
            from gpiozero import Buzzer
            self.buzzer = Buzzer(self.pin)
        except ImportError:
            print(f"Warning: gpiozero not available. Buzzer on pin {self.pin} simulated.")
    
    def activate(self, duration: Optional[float] = None) -> None:
        """Activate the buzzer."""
        self._active = True
        
        if self.buzzer is not None:
            self.buzzer.on()
        else:
            print(f"🔊 BUZZER ON (simulated)")
        
        if duration is not None:
            self._timer = threading.Timer(duration, self.deactivate)
            self._timer.start()
    
    def deactivate(self) -> None:
        """Deactivate the buzzer."""
        self._active = False
        
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        
        if self.buzzer is not None:
            self.buzzer.off()
        else:
            print(f"🔇 BUZZER OFF (simulated)")
    
    def is_active(self) -> bool:
        """Check if buzzer is active."""
        return self._active
    
    def beep(self, on_time: float = 0.5, off_time: float = 0.5, count: int = 3) -> None:
        """Beep the buzzer multiple times.
        
        Args:
            on_time: Duration of each beep
            off_time: Duration between beeps
            count: Number of beeps
        """
        for _ in range(count):
            self.activate()
            time.sleep(on_time)
            self.deactivate()
            time.sleep(off_time)


class GPIOLed(BaseAlarm):
    """LED alarm indicator using Raspberry Pi GPIO."""
    
    def __init__(self, pin: int = 25):
        """Initialize GPIO LED.
        
        Args:
            pin: GPIO pin number (BCM numbering)
        """
        self.pin = pin
        self.led = None
        self._active = False
        self._blink_thread = None
        
        self._setup_gpio()
    
    def _setup_gpio(self) -> None:
        """Setup GPIO for LED."""
        try:
            from gpiozero import LED
            self.led = LED(self.pin)
        except ImportError:
            print(f"Warning: gpiozero not available. LED on pin {self.pin} simulated.")
    
    def activate(self, duration: Optional[float] = None) -> None:
        """Turn on the LED."""
        self._active = True
        
        if self.led is not None:
            self.led.on()
        else:
            print(f"💡 LED ON (simulated)")
        
        if duration is not None:
            threading.Timer(duration, self.deactivate).start()
    
    def deactivate(self) -> None:
        """Turn off the LED."""
        self._active = False
        
        if self.led is not None:
            self.led.off()
        else:
            print(f"⚫ LED OFF (simulated)")
    
    def is_active(self) -> bool:
        """Check if LED is active."""
        return self._active
    
    def blink(self, on_time: float = 0.5, off_time: float = 0.5) -> None:
        """Start blinking the LED.
        
        Args:
            on_time: Duration LED is on
            off_time: Duration LED is off
        """
        if self.led is not None:
            self.led.blink(on_time=on_time, off_time=off_time)
        else:
            print(f"💡 LED BLINKING (simulated)")


class MockAlarm(BaseAlarm):
    """Mock alarm for testing without hardware."""
    
    def __init__(self, name: str = "MockAlarm"):
        """Initialize mock alarm.
        
        Args:
            name: Name for logging
        """
        self.name = name
        self._active = False
    
    def activate(self, duration: Optional[float] = None) -> None:
        """Activate mock alarm."""
        self._active = True
        print(f"🚨 {self.name} ACTIVATED")
        
        if duration is not None:
            threading.Timer(duration, self.deactivate).start()
    
    def deactivate(self) -> None:
        """Deactivate mock alarm."""
        self._active = False
        print(f"✓ {self.name} DEACTIVATED")
    
    def is_active(self) -> bool:
        """Check if alarm is active."""
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
        """Initialize local alarm system.
        
        Args:
            buzzer_pin: GPIO pin for buzzer (None to disable)
            led_pin: GPIO pin for LED (None to disable)
            default_duration: Default alarm duration in seconds
            use_mock: Force use of mock alarms
        """
        self.default_duration = default_duration
        self.alarms = []
        
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
        """Trigger all alarms.
        
        Args:
            duration: Duration in seconds (uses default if None)
        """
        if duration is None:
            duration = self.default_duration
        
        for alarm in self.alarms:
            alarm.activate(duration)
    
    def stop(self) -> None:
        """Stop all alarms."""
        for alarm in self.alarms:
            alarm.deactivate()
    
    def is_active(self) -> bool:
        """Check if any alarm is active."""
        return any(alarm.is_active() for alarm in self.alarms)
