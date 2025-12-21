"""
Transcription functions using Deepgram Nova-3 API.
Fast cloud-based transcription with built-in speaker diarization.

Batch transcription: transcribe_with_deepgram()
Live streaming: DeepgramLiveTranscriber class

Requires:
    - DEEPGRAM_API_KEY in .env
    - deepgram-sdk installed (uv add deepgram-sdk)
"""

import os
import subprocess
import threading
import queue
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from deepgram import DeepgramClient
from deepgram.core.events import EventType


def is_deepgram_available() -> bool:
    """
    Check if Deepgram API is available (API key configured).

    Returns:
        bool: True if DEEPGRAM_API_KEY is set in environment.
    """
    return bool(os.getenv("DEEPGRAM_API_KEY"))


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def convert_wav_to_mp3(wav_path: Path) -> Path:
    """
    Convert WAV to MP3 for faster uploads.

    Large WAV files (100MB+) can timeout during upload.
    MP3 at 128kbps is ~10x smaller and sufficient for speech.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Path to the MP3 file (created in same directory).

    Raises:
        RuntimeError: If FFmpeg conversion fails.
    """
    mp3_path = wav_path.with_suffix('.mp3')

    # Skip if MP3 already exists and is newer than WAV
    if mp3_path.exists() and mp3_path.stat().st_mtime > wav_path.stat().st_mtime:
        return mp3_path

    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(wav_path), '-codec:a', 'libmp3lame',
         '-b:a', '128k', str(mp3_path)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    return mp3_path


def format_diarized_transcript(response) -> str:
    """
    Format Deepgram response into speaker-labeled transcript.

    Uses utterances if available (grouped by speaker), otherwise
    falls back to word-level speaker assignment.

    Args:
        response: Deepgram API response object.

    Returns:
        Formatted transcript with speaker labels.
    """
    results = response.results if hasattr(response, 'results') else response

    # Try utterances first (cleaner output)
    if hasattr(results, 'utterances') and results.utterances:
        lines = []
        for utterance in results.utterances:
            speaker_num = int(utterance.speaker) if utterance.speaker is not None else 0
            speaker = f"Speaker {speaker_num:02d}"
            text = utterance.transcript.strip() if utterance.transcript else ""
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n\n".join(lines)

    # Fallback: use channels/alternatives with word-level diarization
    if hasattr(results, 'channels') and results.channels:
        channel = results.channels[0]
        if hasattr(channel, 'alternatives') and channel.alternatives:
            alt = channel.alternatives[0]

            if hasattr(alt, 'words') and alt.words:
                lines = []
                current_speaker = None
                current_text = []

                for word in alt.words:
                    speaker = getattr(word, 'speaker', None)

                    if speaker != current_speaker:
                        if current_speaker is not None and current_text:
                            speaker_num = int(current_speaker)
                            lines.append(f"Speaker {speaker_num:02d}: {' '.join(current_text)}")
                        current_speaker = speaker
                        current_text = [word.word]
                    else:
                        current_text.append(word.word)

                if current_speaker is not None and current_text:
                    speaker_num = int(current_speaker)
                    lines.append(f"Speaker {speaker_num:02d}: {' '.join(current_text)}")

                return "\n\n".join(lines)

            return alt.transcript if alt.transcript else "No transcription available"

    return "No transcription available"


def transcribe_with_deepgram(
    audio_path: str,
    progress_callback=None
) -> tuple[str, bool]:
    """
    Transcribe audio using Deepgram Nova-3 with speaker diarization.

    Fast cloud-based transcription (~58x realtime speed).
    Cost: $0.0043/min (~$0.26/hour).

    Args:
        audio_path: Path to the audio file.
        progress_callback: Optional callback(progress, stage) for UI updates.
            Note: Deepgram is very fast, so progress may jump quickly.

    Returns:
        tuple: (transcription_text, diarization_used)
            - transcription_text: Formatted transcript with speaker labels
            - diarization_used: Always True (built-in to Deepgram)

    Raises:
        RuntimeError: If API key not configured or API call fails.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not found in .env")

    audio_path = Path(audio_path)

    # Report progress: starting
    if progress_callback:
        progress_callback(0.1, "preparing")

    # Convert WAV to MP3 for faster upload (100MB WAV -> ~10MB MP3)
    upload_path = audio_path
    if audio_path.suffix.lower() == '.wav':
        if progress_callback:
            progress_callback(0.2, "converting")
        upload_path = convert_wav_to_mp3(audio_path)

    # Create Deepgram client with extended timeout for large files
    client = DeepgramClient(api_key=api_key, timeout=300.0)

    # Read the audio file
    with open(upload_path, "rb") as audio_file:
        buffer_data = audio_file.read()

    # Report progress: uploading
    if progress_callback:
        progress_callback(0.4, "uploading")

    # Transcribe using Deepgram Nova-3
    response = client.listen.v1.media.transcribe_file(
        request=buffer_data,
        model="nova-3",
        smart_format=True,      # Smart punctuation and formatting
        diarize=True,           # Speaker diarization (built-in)
        punctuate=True,         # Add punctuation
        utterances=True,        # Group into utterances
        language="en",          # Language
    )

    # Report progress: formatting
    if progress_callback:
        progress_callback(0.9, "formatting")

    # Format transcription with speaker labels
    formatted_text = format_diarized_transcript(response)

    # Report progress: done
    if progress_callback:
        progress_callback(1.0, "complete")

    # Diarization is always used with Deepgram
    return formatted_text, True


def convert_audio_to_deepgram_format(audio_data: np.ndarray) -> bytes:
    """
    Convert sounddevice float32 audio to Deepgram linear16 format.

    Args:
        audio_data: numpy array of float32 audio samples (can be mono or multi-channel)

    Returns:
        bytes: int16 mono audio data suitable for Deepgram streaming
    """
    # Handle multi-channel audio - mix to mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        mono = np.mean(audio_data, axis=1)
    else:
        mono = audio_data.flatten()

    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    int16_data = (mono * 32767).clip(-32768, 32767).astype(np.int16)

    return int16_data.tobytes()


class DeepgramLiveTranscriber:
    """
    Real-time live transcription using Deepgram WebSocket streaming.

    Streams audio chunks to Deepgram and accumulates transcript with speaker labels.
    Cost: ~$0.0077/min (~$0.46/hour).

    Usage:
        transcriber = DeepgramLiveTranscriber(sample_rate=48000)
        transcriber.on_transcript = lambda text, speaker, is_final: print(text)
        transcriber.start()

        # In audio callback:
        transcriber.send_audio(audio_chunk)

        # When done:
        transcript = transcriber.stop()
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize the live transcriber.

        Args:
            sample_rate: Audio sample rate in Hz (must match audio source).
        """
        self.sample_rate = sample_rate
        self.client = None
        self.connection = None
        self.is_connected = False

        # Threading
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._sender_thread = None
        self._listen_thread = None
        self._context_manager = None

        # Transcript accumulation
        self._transcript_parts = []  # List of (speaker, text) tuples

        # Debug metrics
        self._chunks_queued = 0
        self._chunks_sent = 0
        self._bytes_sent = 0

        # Callbacks
        self.on_transcript: Optional[Callable[[str, Optional[int], bool], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connected: Optional[Callable[[], None]] = None
        self.on_disconnected: Optional[Callable[[], None]] = None

        # Get API key
        self._api_key = os.getenv("DEEPGRAM_API_KEY")

    def _on_message(self, message):
        """Handle transcription results from Deepgram."""
        # Log message type for debugging
        msg_type = getattr(message, 'type', 'unknown')
        if msg_type not in ('Results',):  # Don't spam with normal results
            print(f"[DeepgramLive] Message type: {msg_type}")

        if hasattr(message, 'channel'):
            channel = message.channel
            if hasattr(channel, 'alternatives') and channel.alternatives:
                alt = channel.alternatives[0]
                text = getattr(alt, 'transcript', '')

                if text:
                    is_final = getattr(message, 'is_final', False)

                    # Get speaker if available
                    speaker = None
                    if hasattr(alt, 'words') and alt.words:
                        speaker = getattr(alt.words[0], 'speaker', None)

                    # Store final results
                    if is_final:
                        self._transcript_parts.append((speaker, text))

                    # Notify callback
                    if self.on_transcript:
                        self.on_transcript(text, speaker, is_final)

    def _on_open(self, event):
        """Handle WebSocket open."""
        print(f"[DeepgramLive] WebSocket opened")
        if self.on_connected:
            self.on_connected()

    def _on_close(self, event):
        """Handle WebSocket close."""
        # Log close details
        close_code = getattr(event, 'code', 'unknown')
        close_reason = getattr(event, 'reason', 'unknown')
        print(f"[DeepgramLive] WebSocket closed - code={close_code}, reason={close_reason}")
        print(f"[DeepgramLive] Close event details: {event}")
        self.is_connected = False
        if self.on_disconnected:
            self.on_disconnected()

    def _on_error(self, error):
        """Handle WebSocket error."""
        print(f"[DeepgramLive] WebSocket error: {error}")
        if self.on_error:
            self.on_error(error)

    def _sender_loop(self):
        """Background thread to send audio chunks to Deepgram."""
        error_count = 0
        max_errors = 5  # Allow some errors before giving up
        last_status_time = time.time()

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if self.is_connected and self.connection:
                    self.connection.send_media(chunk)
                    self._chunks_sent += 1
                    self._bytes_sent += len(chunk)
                    error_count = 0  # Reset on success

                    # Log status every 5 seconds
                    if time.time() - last_status_time > 5:
                        print(f"[DeepgramLive] Status: sent={self._chunks_sent}, bytes={self._bytes_sent}")
                        last_status_time = time.time()
                else:
                    print(f"[DeepgramLive] Chunk dropped - not connected (connected={self.is_connected})")
            except queue.Empty:
                continue
            except Exception as e:
                error_count += 1
                print(f"[DeepgramLive] Send error #{error_count}: {type(e).__name__}: {e}")
                if not self._stop_event.is_set():
                    if self.on_error:
                        self.on_error(e)
                    # Only stop after too many consecutive errors
                    if error_count >= max_errors:
                        print(f"[DeepgramLive] Too many errors ({error_count}), stopping sender")
                        self.is_connected = False
                        break

    def _listen_loop(self):
        """Background thread to receive messages from Deepgram."""
        try:
            print("[DeepgramLive] Starting listen loop...")
            self.connection.start_listening()
            print("[DeepgramLive] start_listening() returned normally")
        except Exception as e:
            print(f"[DeepgramLive] Listen loop exception: {type(e).__name__}: {e}")
            if not self._stop_event.is_set() and self.on_error:
                self.on_error(e)
        finally:
            # When listen ends (WebSocket closes), mark as disconnected
            if not self._stop_event.is_set():
                print("[DeepgramLive] WebSocket connection ended unexpectedly")
            self.is_connected = False

    def start(self) -> bool:
        """
        Connect to Deepgram and start streaming.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        if not self._api_key:
            if self.on_error:
                self.on_error(RuntimeError("DEEPGRAM_API_KEY not found"))
            return False

        try:
            self.client = DeepgramClient(api_key=self._api_key)

            # Generate request ID for tracking in Deepgram console
            request_id = str(uuid.uuid4())[:8]
            print(f"[DeepgramLive] Request ID: {request_id}")

            # Create context manager with proper parameter types
            self._context_manager = self.client.listen.v1.connect(
                model="nova-3",
                encoding="linear16",
                sample_rate=self.sample_rate,  # int, not string
                channels=1,                     # int, not string
                diarize=True,                   # bool, not string
                smart_format=True,              # bool, not string
                punctuate=True,                 # bool, not string
                interim_results=True,           # bool, not string
                language="en",
            )

            # Enter context to establish connection
            self.connection = self._context_manager.__enter__()

            # Register event handlers BEFORE starting threads
            self.connection.on(EventType.OPEN, self._on_open)
            self.connection.on(EventType.CLOSE, self._on_close)
            self.connection.on(EventType.ERROR, self._on_error)
            self.connection.on(EventType.MESSAGE, self._on_message)

            print(f"[DeepgramLive] Connected! Sample rate: {self.sample_rate}Hz")

            # Start listen thread FIRST (must be running before we can send)
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()

            # Small delay to ensure WebSocket is ready
            time.sleep(0.2)

            # Now mark as connected and start sender
            self.is_connected = True
            self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._sender_thread.start()

            return True

        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False

    def send_audio(self, audio_data: np.ndarray):
        """
        Send audio chunk to Deepgram for transcription.

        Args:
            audio_data: numpy array of float32 audio from sounddevice callback.
        """
        if self.is_connected:
            chunk_bytes = convert_audio_to_deepgram_format(audio_data)
            self._audio_queue.put(chunk_bytes)
            self._chunks_queued += 1
        else:
            # Log occasionally to avoid spam
            if self._chunks_queued % 100 == 0:
                print(f"[DeepgramLive] Audio chunk ignored - not connected")

    def send_audio_bytes(self, audio_bytes: bytes):
        """
        Send raw audio bytes to Deepgram.

        Args:
            audio_bytes: Raw int16 mono audio bytes.
        """
        if self.is_connected:
            self._audio_queue.put(audio_bytes)

    def stop(self) -> str:
        """
        Stop streaming and disconnect from Deepgram.

        Returns:
            str: Full transcript with speaker labels.
        """
        # Print debug stats
        print(f"[DeepgramLive] Stopping - Stats: queued={self._chunks_queued}, sent={self._chunks_sent}, bytes={self._bytes_sent}, transcripts={len(self._transcript_parts)}")

        self._stop_event.set()
        self.is_connected = False

        # Send close signal
        if self.connection:
            try:
                self.connection._send({"type": "CloseStream"})
            except:
                pass

        # Exit context manager
        if self._context_manager:
            try:
                self._context_manager.__exit__(None, None, None)
            except:
                pass

        # Wait for threads
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=2)
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2)

        return self.get_transcript()

    def get_transcript(self) -> str:
        """
        Get the accumulated transcript with speaker labels.

        Returns:
            str: Formatted transcript with "Speaker N:" labels.
        """
        lines = []
        for speaker, text in self._transcript_parts:
            if speaker is not None:
                lines.append(f"Speaker {speaker:02d}: {text}")
            else:
                lines.append(text)
        return "\n\n".join(lines)

    def get_transcript_parts(self) -> list:
        """
        Get raw transcript parts.

        Returns:
            list: List of (speaker, text) tuples.
        """
        return self._transcript_parts.copy()
