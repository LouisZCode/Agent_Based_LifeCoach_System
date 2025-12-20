"""
Mac implementation of audio capture using sounddevice and BlackHole.

Requirements:
- BlackHole 2ch installed (https://existential.audio/blackhole/)
- Aggregate device created in Audio MIDI Setup (mic + BlackHole)
- sounddevice and soundfile packages
"""

import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from .base import AudioCapturer


class MacAudioCapturer(AudioCapturer):
    """
    Mac audio capture using sounddevice.

    Designed to work with BlackHole aggregate device to capture
    both microphone input and system audio simultaneously.

    For aggregate devices with 3+ channels (mic + BlackHole stereo),
    automatically mixes down to stereo with both sources centered.
    """

    def __init__(self):
        self._recording = False
        self._audio_data: list[np.ndarray] = []
        self._output_path: Optional[str] = None
        self._sample_rate: int = 48000
        self._channels: int = 2
        self._raw_channels: int = 2  # Actual channels being recorded
        self._start_time: Optional[float] = None
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def get_available_devices(self) -> list[dict]:
        """Get list of available audio input devices."""
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            # Only include input devices (max_input_channels > 0)
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': int(device['default_samplerate'])
                })

        return input_devices

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        """Callback function called for each audio block."""
        if status:
            print(f"Audio callback status: {status}")

        with self._lock:
            if self._recording:
                # Make a copy of the data
                self._audio_data.append(indata.copy())

    def start_recording(self, device_id: int, output_path: str,
                        sample_rate: int = 48000, channels: int = 2) -> None:
        """
        Start recording audio from the specified device.

        For aggregate devices (mic + BlackHole), records all available channels
        and mixes down to stereo on save. Uses device's native sample rate.
        """
        if self._recording:
            raise RuntimeError("Recording already in progress")

        # Get device info for native sample rate and channel count
        device_info = sd.query_devices(device_id)
        device_sample_rate = int(device_info['default_samplerate'])
        device_channels = device_info['max_input_channels']

        self._output_path = output_path
        self._sample_rate = device_sample_rate  # Use device's native rate
        self._channels = channels  # Output channels (stereo)
        self._raw_channels = device_channels  # Record all device channels
        self._audio_data = []
        self._recording = True
        self._start_time = time.time()

        print(f"Recording: {device_info['name']}")
        print(f"  Sample rate: {device_sample_rate} Hz")
        print(f"  Input channels: {device_channels}")

        # Create and start the input stream (record ALL channels)
        self._stream = sd.InputStream(
            device=device_id,
            channels=device_channels,  # Record all available channels
            samplerate=device_sample_rate,
            callback=self._audio_callback,
            dtype=np.float32
        )
        self._stream.start()

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save the audio file."""
        if not self._recording:
            return None

        self._recording = False

        # Stop and close the stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Combine all audio chunks
        with self._lock:
            if not self._audio_data:
                return None

            audio_array = np.concatenate(self._audio_data, axis=0)

        # Mix down channels to stereo
        audio_array = self._mix_to_stereo(audio_array)

        # Save to file
        sf.write(self._output_path, audio_array, self._sample_rate)

        # Reset state
        saved_path = self._output_path
        self._output_path = None
        self._audio_data = []
        self._start_time = None

        return saved_path

    def _mix_to_stereo(self, audio: np.ndarray) -> np.ndarray:
        """
        Mix multi-channel audio to stereo.

        For 3-channel aggregate devices (mic + BlackHole stereo):
        - Channel 0: Microphone (mono)
        - Channel 1: BlackHole Left
        - Channel 2: BlackHole Right

        Mixes so mic is centered and system audio stays stereo.
        """
        num_channels = audio.shape[1] if len(audio.shape) > 1 else 1

        if num_channels == 1:
            # Mono: duplicate to stereo
            return np.column_stack([audio, audio])

        elif num_channels == 2:
            # Already stereo, return as-is
            return audio

        elif num_channels == 3:
            # Aggregate device: mic (ch0) + BlackHole stereo (ch1, ch2)
            mic = audio[:, 0]
            blackhole_left = audio[:, 1]
            blackhole_right = audio[:, 2]

            # Mix: center the mic, keep system audio stereo
            left = (mic + blackhole_left) / 2
            right = (mic + blackhole_right) / 2

            print(f"Mixed 3 channels → stereo (mic centered)")
            return np.column_stack([left, right])

        else:
            # More than 3 channels: take first 2 as fallback
            print(f"Warning: {num_channels} channels, using first 2")
            return audio[:, :2]

    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        return self._recording

    def get_recording_duration(self) -> float:
        """Get the duration of the current recording in seconds."""
        if not self._recording or self._start_time is None:
            return 0.0
        return time.time() - self._start_time


def find_aggregate_device(name_contains: str = "Session Capture") -> Optional[int]:
    """
    Find an aggregate device by name.

    Args:
        name_contains: Substring to search for in device names

    Returns:
        Device ID if found, None otherwise
    """
    devices = sd.query_devices()

    for i, device in enumerate(devices):
        if name_contains.lower() in device['name'].lower():
            if device['max_input_channels'] > 0:
                return i

    return None
