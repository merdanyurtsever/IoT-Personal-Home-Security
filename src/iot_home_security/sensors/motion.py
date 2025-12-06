"""Motion sensor (PIR) interface module.

This module provides interface for PIR motion sensors
connected to Raspberry Pi GPIO pins.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import time


class BaseMotionSensor(ABC):
    """Abstract base class for motion sensors."""
    
    @abstractmethod
    def read(self) -> bool:
        """Read motion sensor state.
        
        Returns:
            True if motion detected, False otherwise
        """
        pass
    
    @abstractmethod
    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        """Wait for motion to be detected.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if motion detected, False if timeout
        """
        pass
    
    @abstractmethod
    def on_motion(self, callback: Callable[[], None]) -> None:
        """Register callback for motion detection.
        
        Args:
            callback: Function to call when motion detected
        """
        pass


class GPIOMotionSensor(BaseMotionSensor):
    """PIR motion sensor using Raspberry Pi GPIO."""
    
    def __init__(
        self,
        pin: int = 17,
        pull_up: bool = False,
        bounce_time: float = 0.3,
    ):
        """Initialize GPIO motion sensor.
        
        Args:
            pin: GPIO pin number (BCM numbering)
            pull_up: Use internal pull-up resistor
            bounce_time: Debounce time in seconds
        """
        self.pin = pin
        self.pull_up = pull_up
        self.bounce_time = bounce_time
        
        self.gpio = None
        self.sensor = None
        self._callback = None
        
        self._setup_gpio()
    
    def _setup_gpio(self) -> None:
        """Setup GPIO pin for motion sensor."""
        try:
            from gpiozero import MotionSensor as GPIOZeroMotionSensor
            
            self.sensor = GPIOZeroMotionSensor(
                self.pin,
                pull_up=self.pull_up,
            )
        except ImportError:
            # Not running on Raspberry Pi - use mock
            self.sensor = None
            print(
                f"Warning: gpiozero not available. "
                f"Motion sensor on pin {self.pin} will be simulated."
            )
    
    def read(self) -> bool:
        """Read current motion sensor state."""
        if self.sensor is None:
            return False
        
        return self.sensor.motion_detected
    
    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        """Wait for motion detection."""
        if self.sensor is None:
            return False
        
        return self.sensor.wait_for_motion(timeout=timeout)
    
    def on_motion(self, callback: Callable[[], None]) -> None:
        """Register motion detection callback."""
        self._callback = callback
        
        if self.sensor is not None:
            self.sensor.when_motion = callback
    
    def on_no_motion(self, callback: Callable[[], None]) -> None:
        """Register callback for when motion stops."""
        if self.sensor is not None:
            self.sensor.when_no_motion = callback


class MockMotionSensor(BaseMotionSensor):
    """Mock motion sensor for testing without hardware."""
    
    def __init__(self, pin: int = 17):
        """Initialize mock motion sensor.
        
        Args:
            pin: Simulated GPIO pin
        """
        self.pin = pin
        self._motion_detected = False
        self._callback = None
    
    def read(self) -> bool:
        """Read motion sensor state."""
        return self._motion_detected
    
    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        """Wait for motion (simulated)."""
        start = time.time()
        
        while True:
            if self._motion_detected:
                return True
            
            if timeout and (time.time() - start) >= timeout:
                return False
            
            time.sleep(0.1)
    
    def on_motion(self, callback: Callable[[], None]) -> None:
        """Register motion callback."""
        self._callback = callback
    
    def simulate_motion(self) -> None:
        """Simulate motion detection (for testing)."""
        self._motion_detected = True
        if self._callback:
            self._callback()
    
    def simulate_no_motion(self) -> None:
        """Simulate motion stopped (for testing)."""
        self._motion_detected = False


class MotionSensor:
    """Unified motion sensor interface with automatic backend selection."""
    
    def __init__(
        self,
        pin: int = 17,
        cooldown: float = 5.0,
        use_mock: bool = False,
    ):
        """Initialize motion sensor.
        
        Args:
            pin: GPIO pin number
            cooldown: Cooldown time between detections in seconds
            use_mock: Force use of mock sensor
        """
        self.pin = pin
        self.cooldown = cooldown
        self.last_motion_time = 0
        
        if use_mock:
            self.sensor = MockMotionSensor(pin)
        else:
            try:
                self.sensor = GPIOMotionSensor(pin)
            except Exception:
                print("Falling back to mock motion sensor")
                self.sensor = MockMotionSensor(pin)
    
    def read(self) -> bool:
        """Read motion sensor state with cooldown."""
        current_time = time.time()
        
        if current_time - self.last_motion_time < self.cooldown:
            return False
        
        motion = self.sensor.read()
        
        if motion:
            self.last_motion_time = current_time
        
        return motion
    
    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        """Wait for motion detection."""
        return self.sensor.wait_for_motion(timeout)
    
    def on_motion(self, callback: Callable[[], None]) -> None:
        """Register motion detection callback with cooldown."""
        def cooldown_callback():
            current_time = time.time()
            if current_time - self.last_motion_time >= self.cooldown:
                self.last_motion_time = current_time
                callback()
        
        self.sensor.on_motion(cooldown_callback)
