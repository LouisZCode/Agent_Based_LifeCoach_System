"""
Mac implementation of audio capture using sounddevice and BlackHole.

Supports:
- Local recording to MP3 file (converted from WAV via FFmpeg)
- Optional real-time streaming callback for live transcription

Requirements:
- BlackHole 2ch installed (https://existential.audio/blackhole/)
- Aggregate device created in Audio MIDI Setup (mic + BlackHole)
- sounddevice and soundfile packages
- FFmpeg installed (brew install ffmpeg)
"""

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

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
        # Optional callback for streaming audio chunks (for live transcription)
        self._on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None
        # Audio levels for visualization (RMS values 0.0-1.0)
        self._mic_level: float = 0.0
        self._system_level: float = 0.0

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
                audio_copy = indata.copy()
                self._audio_data.append(audio_copy)

                # Calculate audio levels for visualization
                self._update_audio_levels(audio_copy)

                # Stream to callback if provided (for live transcription)
                if self._on_audio_chunk is not None:
                    try:
                        self._on_audio_chunk(audio_copy)
                    except Exception as e:
                        print(f"Audio chunk callback error: {e}")

    def _update_audio_levels(self, audio: np.ndarray) -> None:
        """
        Update audio level meters from audio chunk.

        For aggregate devices (3 channels):
        - Channel 0: Microphone
        - Channels 1-2: System audio (BlackHole stereo)
        """
        num_channels = audio.shape[1] if len(audio.shape) > 1 else 1

        # Calculate RMS (root mean square) for each source
        if num_channels >= 3:
            # Aggregate device: mic (ch0) + system (ch1, ch2)
            mic_rms = np.sqrt(np.mean(audio[:, 0] ** 2))
            system_rms = np.sqrt(np.mean((audio[:, 1] ** 2 + audio[:, 2] ** 2) / 2))
        elif num_channels == 2:
            # Stereo: treat as system audio, no mic
            mic_rms = 0.0
            system_rms = np.sqrt(np.mean(audio ** 2))
        else:
            # Mono: treat as mic only
            mic_rms = np.sqrt(np.mean(audio ** 2))
            system_rms = 0.0

        # Smooth the levels (exponential moving average)
        smoothing = 0.3
        self._mic_level = smoothing * mic_rms + (1 - smoothing) * self._mic_level
        self._system_level = smoothing * system_rms + (1 - smoothing) * self._system_level

    def start_recording(
        self,
        device_id: int,
        output_path: str,
        sample_rate: int = 48000,
        channels: int = 2,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None
    ) -> None:
        """
        Start recording audio from the specified device.

        For aggregate devices (mic + BlackHole), records all available channels
        and mixes down to stereo on save. Uses device's native sample rate.

        Args:
            device_id: ID of the audio input device.
            output_path: Path to save the WAV file.
            sample_rate: Desired sample rate (device's native rate is used).
            channels: Number of output channels (default 2 for stereo).
            on_audio_chunk: Optional callback for streaming audio chunks.
                Called with each audio chunk (numpy float32 array) for
                real-time processing like live transcription.
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
        self._on_audio_chunk = on_audio_chunk  # Store chunk callback

        print(f"Recording: {device_info['name']}")
        print(f"  Sample rate: {device_sample_rate} Hz")
        print(f"  Input channels: {device_channels}")
        if on_audio_chunk:
            print(f"  Live streaming: enabled")

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
        """Stop recording and save the audio file as MP3."""
        if not self._recording:
            return None

        self._recording = False
        self._on_audio_chunk = None  # Clear callback

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

        # Determine output path (convert .wav to .mp3 if needed)
        output_path = Path(self._output_path)
        if output_path.suffix.lower() == '.wav':
            mp3_path = output_path.with_suffix('.mp3')
        else:
            mp3_path = output_path.with_suffix('.mp3')

        # Save to temporary WAV first
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_wav = tmp.name

        sf.write(temp_wav, audio_array, self._sample_rate)

        # Convert to MP3 with FFmpeg (128kbps - good for speech)
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', temp_wav, '-codec:a', 'libmp3lame',
             '-b:a', '128k', str(mp3_path)],
            capture_output=True, text=True
        )

        # Clean up temp WAV
        os.unlink(temp_wav)

        if result.returncode != 0:
            print(f"[AudioCapture] FFmpeg error: {result.stderr}")
            # Fallback: save as WAV if MP3 conversion fails
            sf.write(str(output_path), audio_array, self._sample_rate)
            saved_path = str(output_path)
        else:
            saved_path = str(mp3_path)
            print(f"[AudioCapture] Saved as MP3: {mp3_path.name}")

        # Reset state
        self._output_path = None
        self._audio_data = []
        self._start_time = None

        return saved_path

    def get_sample_rate(self) -> int:
        """Get the current recording sample rate."""
        return self._sample_rate

    def get_device_sample_rate(self, device_id: int) -> int:
        """Get the native sample rate for a device."""
        device_info = sd.query_devices(device_id)
        return int(device_info['default_samplerate'])

    def get_audio_levels(self) -> tuple[float, float]:
        """
        Get current audio levels for visualization.

        Returns:
            tuple: (mic_level, system_level) - RMS values from 0.0 to ~1.0
                   Values above 0.1 indicate active audio.
        """
        return (self._mic_level, self._system_level)

    def get_channel_count(self) -> int:
        """Get the number of channels being recorded."""
        return self._raw_channels

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
