"""
Test Deepgram WebSocket Live Streaming Transcription.

This script tests real-time transcription using the Deepgram WebSocket API.
It captures audio from a device and streams it to Deepgram for live transcription.

Usage:
    # List available devices
    uv run python test_deepgram_live.py --list-devices

    # Run test with default device for 10 seconds
    uv run python test_deepgram_live.py

    # Run test with specific device for 30 seconds
    uv run python test_deepgram_live.py --device 3 --duration 30

Requires:
    - DEEPGRAM_API_KEY in .env
    - sounddevice installed
    - BlackHole + Aggregate device for system audio capture
"""

import os
import sys
import time
import argparse
import threading
import queue
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from parent directory
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.core.events import EventType


def list_devices():
    """List all available audio input devices."""
    print("\n=== Available Audio Input Devices ===\n")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']}")
            print(f"      Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
    print()


def convert_to_deepgram_format(audio_data: np.ndarray) -> bytes:
    """
    Convert sounddevice float32 audio to Deepgram linear16 format.

    Args:
        audio_data: numpy array of float32 audio samples

    Returns:
        bytes: int16 mono audio data
    """
    # Handle multi-channel audio - mix to mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        mono = np.mean(audio_data, axis=1)
    else:
        mono = audio_data.flatten()

    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    int16_data = (mono * 32767).clip(-32768, 32767).astype(np.int16)

    return int16_data.tobytes()


def run_test(device_id: int, duration: int):
    """Run live transcription test."""

    # Get device info
    if device_id is not None:
        device_info = sd.query_devices(device_id)
    else:
        device_info = sd.query_devices(kind='input')
        device_id = sd.default.device[0]

    sample_rate = int(device_info['default_samplerate'])
    channels = device_info['max_input_channels']

    print(f"\n{'='*60}")
    print("DEEPGRAM LIVE STREAMING TEST")
    print('='*60)
    print(f"Device: {device_info['name']}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Duration: {duration} seconds")
    print('='*60)

    # Get API key
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("[ERROR] DEEPGRAM_API_KEY not found in .env")
        return

    # Create audio buffer queue
    audio_queue = queue.Queue()
    transcript_parts = []
    stop_event = threading.Event()

    # Audio callback - buffers audio
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] Audio status: {status}")
        # Convert and queue the audio
        chunk_bytes = convert_to_deepgram_format(indata)
        audio_queue.put(chunk_bytes)

    # Message handler
    def on_message(message):
        if hasattr(message, 'channel'):
            channel = message.channel
            if hasattr(channel, 'alternatives') and channel.alternatives:
                alt = channel.alternatives[0]
                text = getattr(alt, 'transcript', '')

                if text:
                    is_final = getattr(message, 'is_final', False)
                    speaker = None
                    if hasattr(alt, 'words') and alt.words:
                        speaker = getattr(alt.words[0], 'speaker', None)

                    if is_final:
                        speaker_label = f"Speaker {speaker}" if speaker is not None else ""
                        if speaker_label:
                            print(f"\n[FINAL] {speaker_label}: {text}")
                        else:
                            print(f"\n[FINAL] {text}")
                        transcript_parts.append((speaker, text))
                    else:
                        print(f"\r[...] {text[:60]}{'...' if len(text) > 60 else ''}", end='', flush=True)

    # Start audio capture FIRST (before connecting)
    print(f"\n[INFO] Starting audio capture...")
    stream = sd.InputStream(
        device=device_id,
        channels=channels,
        samplerate=sample_rate,
        dtype='float32',
        blocksize=2048,
        callback=audio_callback
    )
    stream.start()
    print("[INFO] Audio capture started!")

    # Wait a moment to buffer some audio
    time.sleep(0.1)

    try:
        print(f"[INFO] Connecting to Deepgram (sample_rate={sample_rate})...")

        # Create client and connect
        client = DeepgramClient(api_key=api_key)

        # Use context manager properly
        with client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=str(sample_rate),
            channels="1",  # We convert to mono
            diarize="true",
            smart_format="true",
            punctuate="true",
            interim_results="true",
            language="en",
        ) as connection:

            print("[INFO] Connected to Deepgram!")

            # Register event handlers
            connection.on(EventType.OPEN, lambda _: print("[INFO] WebSocket opened"))
            connection.on(EventType.CLOSE, lambda _: print("[INFO] WebSocket closed"))
            connection.on(EventType.ERROR, lambda e: print(f"[ERROR] WebSocket error: {e}"))
            connection.on(EventType.MESSAGE, on_message)

            # Audio sender thread - sends buffered audio to Deepgram
            def send_audio():
                while not stop_event.is_set():
                    try:
                        # Get audio from queue with timeout
                        chunk = audio_queue.get(timeout=0.1)
                        connection.send_media(chunk)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        if not stop_event.is_set():
                            print(f"\n[ERROR] Send error: {e}")
                        break

            # Start sender thread
            sender_thread = threading.Thread(target=send_audio, daemon=True)
            sender_thread.start()

            # Start listening in background
            listen_thread = threading.Thread(
                target=connection.start_listening,
                daemon=True
            )
            listen_thread.start()

            print(f"\n[INFO] Recording for {duration} seconds... Speak now!")
            print("-" * 40)

            # Wait for duration
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(0.1)

            print("\n" + "-" * 40)
            print("[INFO] Recording complete. Waiting for final results...")

            # Stop sending
            stop_event.set()

            # Send close signal
            try:
                connection._send({"type": "CloseStream"})
            except:
                pass

            # Wait for final transcripts
            time.sleep(2)

    except Exception as e:
        print(f"\n[ERROR] Connection error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop audio capture
        stop_event.set()
        stream.stop()
        stream.close()

    # Show final transcript
    print(f"\n{'='*60}")
    print("FINAL TRANSCRIPT")
    print('='*60)

    if transcript_parts:
        lines = []
        for speaker, text in transcript_parts:
            if speaker is not None:
                lines.append(f"Speaker {speaker}: {text}")
            else:
                lines.append(text)
        transcript = "\n\n".join(lines)
        print(transcript)
    else:
        print("(No transcript received)")
        transcript = ""

    print('='*60)

    # Save transcript
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"deepgram_live_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write(f"Deepgram Live Streaming Test\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device_info['name']}\n")
        f.write(f"Sample Rate: {sample_rate} Hz\n")
        f.write(f"Duration: {duration} seconds\n")
        f.write(f"\n{'='*40}\n\n")
        f.write(transcript if transcript else "(No transcript)")

    print(f"\n[INFO] Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test Deepgram Live Streaming")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    parser.add_argument("--device", type=int, default=None, help="Audio device ID to use")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    run_test(args.device, args.duration)


if __name__ == "__main__":
    main()
