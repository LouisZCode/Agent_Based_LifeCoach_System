"""
Abstract base class for audio capture.
Defines the interface that platform-specific implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional


class AudioCapturer(ABC):
    """
    Abstract base class for capturing audio from system devices.

    Implementations:
    - MacAudioCapturer: Uses sounddevice + BlackHole aggregate device
    - WindowsAudioCapturer: (Future) Uses WASAPI loopback
    """

    @abstractmethod
    def get_available_devices(self) -> list[dict]:
        """
        Get list of available audio input devices.

        Returns:
            List of dicts with keys: 'id', 'name', 'channels', 'sample_rate'
        """
        pass

    @abstractmethod
    def start_recording(self, device_id: int, output_path: str,
                        sample_rate: int = 48000, channels: int = 2) -> None:
        """
        Start recording audio from the specified device.

        Args:
            device_id: ID of the input device to record from
            output_path: Path where the audio file will be saved
            sample_rate: Sample rate in Hz (default: 44100)
            channels: Number of audio channels (default: 2 for stereo)
        """
        pass

    @abstractmethod
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save the audio file.

        Returns:
            Path to the saved audio file, or None if no recording was active
        """
        pass

    @abstractmethod
    def is_recording(self) -> bool:
        """
        Check if recording is currently active.

        Returns:
            True if recording, False otherwise
        """
        pass

    @abstractmethod
    def get_recording_duration(self) -> float:
        """
        Get the duration of the current recording in seconds.

        Returns:
            Duration in seconds, or 0 if not recording
        """
        pass
