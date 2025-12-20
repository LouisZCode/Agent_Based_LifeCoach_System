"""
Audio Capture Module

Cross-platform audio recording with system audio capture support.

Usage:
    from audio_capture import get_audio_capturer

    capturer = get_audio_capturer()
    devices = capturer.get_available_devices()
    capturer.start_recording(device_id=0, output_path="recording.wav")
    # ... recording ...
    capturer.stop_recording()
"""

import sys

from .base import AudioCapturer


def get_audio_capturer() -> AudioCapturer:
    """
    Factory function to get the appropriate audio capturer for the current platform.

    Returns:
        AudioCapturer instance for the current platform

    Raises:
        NotImplementedError: If the platform is not supported
    """
    if sys.platform == "darwin":
        from .mac_capture import MacAudioCapturer
        return MacAudioCapturer()

    elif sys.platform == "win32":
        # Future: Windows implementation using WASAPI loopback
        raise NotImplementedError(
            "Windows audio capture not yet implemented. "
            "Coming soon with WASAPI loopback support."
        )

    else:
        raise NotImplementedError(
            f"Audio capture not supported on platform: {sys.platform}"
        )


def get_platform_info() -> dict:
    """
    Get information about audio capture support on the current platform.

    Returns:
        Dict with 'platform', 'supported', 'requirements' keys
    """
    if sys.platform == "darwin":
        return {
            "platform": "macOS",
            "supported": True,
            "requirements": [
                "BlackHole 2ch installed",
                "Aggregate device created in Audio MIDI Setup",
                "Multi-output device for hearing audio while recording"
            ]
        }
    elif sys.platform == "win32":
        return {
            "platform": "Windows",
            "supported": False,
            "requirements": ["WASAPI loopback (coming soon)"]
        }
    else:
        return {
            "platform": sys.platform,
            "supported": False,
            "requirements": ["Platform not supported"]
        }


# Convenience exports
__all__ = [
    "get_audio_capturer",
    "get_platform_info",
    "AudioCapturer",
]
