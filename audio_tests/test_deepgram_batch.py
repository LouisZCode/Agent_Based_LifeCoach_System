"""
Test script for Deepgram Nova-3 batch transcription with diarization.
Pre-recorded audio file transcription (not real-time).

Usage:
    1. Add DEEPGRAM_API_KEY to .env
    2. Drop an audio file in audio_sample/
    3. Run: python test_deepgram_batch.py
    4. Check test_results/ for output

Requirements:
    - deepgram-sdk installed (uv add deepgram-sdk)
    - DEEPGRAM_API_KEY in .env
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

from deepgram import DeepgramClient


# Pricing (as of Dec 2025)
DEEPGRAM_BATCH_PRICE_PER_MINUTE = 0.0043  # $0.0043/min for Nova-3 batch


def find_audio_file(folder: Path) -> Path | None:
    """Find the audio file in the sample folder."""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in audio_extensions:
            return file
    return None


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    import subprocess

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def convert_wav_to_mp3(wav_path: Path) -> Path:
    """Convert WAV to MP3 for faster uploads. Returns path to MP3 file."""
    import subprocess

    mp3_path = wav_path.with_suffix('.mp3')

    # Skip if MP3 already exists and is newer than WAV
    if mp3_path.exists() and mp3_path.stat().st_mtime > wav_path.stat().st_mtime:
        print(f"  Using existing MP3: {mp3_path.name}")
        return mp3_path

    print(f"  Converting WAV to MP3 for faster upload...")

    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(wav_path), '-codec:a', 'libmp3lame',
         '-b:a', '128k', str(mp3_path)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    original_size = get_file_size_mb(wav_path)
    new_size = get_file_size_mb(mp3_path)
    print(f"  Converted: {original_size:.1f}MB → {new_size:.1f}MB ({original_size/new_size:.1f}x smaller)")

    return mp3_path


def transcribe_with_deepgram(audio_path: Path) -> dict:
    """
    Transcribe audio using Deepgram Nova-3 with diarization.

    Returns dict with:
        - text: formatted transcription with speaker labels
        - duration: audio duration in seconds
        - raw_response: the raw API response
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not found in .env")

    # Create Deepgram client with extended timeout for large files
    client = DeepgramClient(api_key=api_key, timeout=300.0)

    # Convert WAV to MP3 for faster upload (100MB WAV → ~10MB MP3)
    upload_path = audio_path
    if audio_path.suffix.lower() == '.wav':
        upload_path = convert_wav_to_mp3(audio_path)

    # Read the audio file
    with open(upload_path, "rb") as audio_file:
        buffer_data = audio_file.read()

    file_size = get_file_size_mb(upload_path)
    print(f"  Uploading {file_size:.1f}MB to Deepgram API...")

    # Transcribe using v4 SDK API
    response = client.listen.v1.media.transcribe_file(
        request=buffer_data,
        model="nova-3",
        smart_format=True,      # Smart punctuation and formatting
        diarize=True,           # Speaker diarization
        punctuate=True,         # Add punctuation
        utterances=True,        # Group into utterances
        language="en",          # Language
    )

    # Extract duration from metadata
    duration = response.metadata.duration if hasattr(response, 'metadata') else get_audio_duration(audio_path)

    # Format transcription with speaker labels
    formatted_text = format_diarized_transcript(response)

    return {
        "text": formatted_text,
        "duration": duration,
        "raw_response": response
    }


def format_diarized_transcript(response) -> str:
    """
    Format Deepgram response into speaker-labeled transcript.

    Uses utterances if available (grouped by speaker), otherwise
    falls back to word-level speaker assignment.
    """
    # Get results from response
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

            # Check if we have word-level speaker info
            if hasattr(alt, 'words') and alt.words:
                lines = []
                current_speaker = None
                current_text = []

                for word in alt.words:
                    speaker = getattr(word, 'speaker', None)

                    if speaker != current_speaker:
                        if current_speaker is not None and current_text:
                            speaker_num = int(current_speaker) if current_speaker is not None else 0
                            lines.append(f"Speaker {speaker_num:02d}: {' '.join(current_text)}")
                        current_speaker = speaker
                        current_text = [word.word]
                    else:
                        current_text.append(word.word)

                # Don't forget last speaker
                if current_speaker is not None and current_text:
                    speaker_num = int(current_speaker) if current_speaker is not None else 0
                    lines.append(f"Speaker {speaker_num:02d}: {' '.join(current_text)}")

                return "\n\n".join(lines)

            # No word-level speaker info, just return transcript
            return alt.transcript if alt.transcript else "No transcription available"

    return "No transcription available"


def save_results(audio_name: str, result: dict, output_folder: Path,
                 timing: dict = None):
    """Save transcription results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"deepgram_batch_{timestamp}_{audio_name}.txt"

    duration = result.get("duration", 0)
    cost_estimate = (duration / 60) * DEEPGRAM_BATCH_PRICE_PER_MINUTE

    # Format timing info
    timing_info = ""
    if timing:
        timing_info = f"""
Processing Time:
  - Deepgram API: {timing.get('deepgram', 0):.1f}s
  - Total: {timing.get('total', 0):.1f}s
  - Speed: {duration / timing.get('total', 1):.1f}x realtime
"""

    content = f"""=== TRANSCRIPTION TEST RESULTS ===
Method: Deepgram Nova-3 (Batch)
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Audio File: {audio_name}
Audio Duration: {format_duration(duration)}
Cost Estimate: ${cost_estimate:.4f} (@ ${DEEPGRAM_BATCH_PRICE_PER_MINUTE}/min)
{timing_info}
Speaker Diarization: Yes (built-in)

Features:
  - Model: Nova-3
  - Smart formatting: Yes
  - Punctuation: Yes
  - Utterance grouping: Yes

=== TRANSCRIPTION ===

{result['text']}
"""

    output_file.write_text(content, encoding='utf-8')
    return output_file


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    sample_folder = script_dir / "audio_sample"
    results_folder = script_dir / "test_results"

    # Check API key
    if not os.getenv("DEEPGRAM_API_KEY"):
        print("Error: DEEPGRAM_API_KEY not found in .env")
        print("Sign up at https://deepgram.com and add your API key to .env")
        sys.exit(1)

    # Find audio file
    audio_file = find_audio_file(sample_folder)
    if not audio_file:
        print(f"Error: No audio file found in {sample_folder}")
        print("Supported formats: mp3, wav, m4a, flac, ogg, webm")
        sys.exit(1)

    print(f"Found audio file: {audio_file.name}")
    file_size = get_file_size_mb(audio_file)
    print(f"File size: {file_size:.1f}MB")
    print()

    try:
        timing = {}
        total_start = time.time()

        # Transcribe with Deepgram
        print("Transcribing with Deepgram Nova-3 (batch)...")
        api_start = time.time()
        result = transcribe_with_deepgram(audio_file)
        timing['deepgram'] = time.time() - api_start
        timing['total'] = time.time() - total_start

        # Save results
        output_file = save_results(audio_file.stem, result, results_folder, timing)

        duration = result['duration']
        cost = (duration / 60) * DEEPGRAM_BATCH_PRICE_PER_MINUTE

        print()
        print(f"Audio duration: {format_duration(duration)}")
        print(f"Processing time: {timing['total']:.1f}s ({duration / timing['total']:.1f}x realtime)")
        print(f"Cost estimate: ${cost:.4f}")
        print()
        print(f"Results saved to: {output_file}")
        print()
        print("=== PREVIEW ===")
        print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
