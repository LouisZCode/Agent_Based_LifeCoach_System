"""
Transcription functions using Parakeet-MLX for audio-to-text conversion.
Optional speaker diarization using Pyannote.
Optimized for Apple Silicon (M1/M2/M3).
"""

from pathlib import Path
import os

# HuggingFace model ID for Parakeet-MLX
# Model auto-downloads on first use (~600MB) and caches locally
# V3 supports 25 European languages with auto-detection
PARAKEET_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"

# Pyannote speaker diarization model (requires HuggingFace token)
# CC-BY-4.0 license, free to use
PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"


def is_model_cached() -> bool:
    """
    Check if the Parakeet model is already downloaded and cached.

    Returns:
        bool: True if model is cached, False if it needs to be downloaded.
    """
    # HuggingFace cache location
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")

    # Model folder name pattern in HF cache
    model_folder_pattern = "models--mlx-community--parakeet-tdt-0.6b-v3"
    model_path = os.path.join(hf_cache, model_folder_pattern)

    # Check if the model folder exists and has content
    if os.path.exists(model_path):
        snapshots_path = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_path) and os.listdir(snapshots_path):
            return True

    return False


def transcribe_audio(audio_path: str, progress_callback=None) -> str:
    """
    Transcribe an audio file using Parakeet-MLX model.

    Args:
        audio_path (str): Path to the audio file to transcribe.
            Supported formats: mp3, wav, m4a, flac
        progress_callback: Optional callback function(progress: float)
            where progress is 0.0 to 1.0. Used for progress bar updates.

    Returns:
        str: The transcribed text with punctuation and capitalization.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio format is not supported.
        RuntimeError: If transcription fails.
    """
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import load_audio

    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    supported_formats = {'.mp3', '.wav', '.m4a', '.flac'}
    if audio_file.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported audio format: {audio_file.suffix}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    try:
        model = from_pretrained(PARAKEET_MODEL_ID)

        # If no progress callback, use simple transcribe
        if progress_callback is None:
            result = model.transcribe(str(audio_file))
            return result.text

        # Use streaming API for progress tracking
        sample_rate = model.preprocessor_config.sample_rate
        audio_data = load_audio(str(audio_file), sample_rate)

        # Process in 1-second chunks for progress updates
        chunk_size = sample_rate  # 1 second of audio
        total_chunks = max(1, len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0))

        with model.transcribe_stream(
            context_size=(256, 256),  # (left_context, right_context) frames
        ) as transcriber:
            for i, start in enumerate(range(0, len(audio_data), chunk_size)):
                chunk = audio_data[start:start + chunk_size]
                transcriber.add_audio(chunk)

                # Update progress
                progress = min((i + 1) / total_chunks, 1.0)
                progress_callback(progress)

            # Get final result
            result = transcriber.result
            return result.text

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


def save_transcription(path: str, content: str) -> str:
    """
    Save transcription text to the session folder.

    Args:
        path (str): The session folder path where transcription will be saved.
        content (str): The transcription text to save.

    Returns:
        str: Confirmation message with the file path.

    Raises:
        OSError: If there's an issue writing the file.
    """
    file_path = Path(f"{path}/transcription.txt")
    file_path.write_text(content, encoding='utf-8')

    return f"Transcription saved to {file_path}"


def get_huggingface_token() -> str | None:
    """
    Get HuggingFace token from environment variables.

    Checks for HUGGINGFACE_TOKEN or HF_TOKEN environment variables.

    Returns:
        str | None: The token if found, None otherwise.
    """
    return os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")


def is_diarization_available() -> bool:
    """
    Check if speaker diarization is available.

    Requires:
    - pyannote.audio package installed
    - Valid HuggingFace token in environment

    Returns:
        bool: True if diarization can be used, False otherwise.
    """
    # First check if token exists (fast check, no imports)
    if not get_huggingface_token():
        return False

    # Then check if pyannote is installed (lazy import to avoid startup warnings)
    try:
        import importlib.util
        return importlib.util.find_spec("pyannote.audio") is not None
    except Exception:
        return False


def is_diarization_model_cached() -> bool:
    """
    Check if the Pyannote diarization model is already downloaded.

    Returns:
        bool: True if model is cached, False if download needed.
    """
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_folder = "models--pyannote--speaker-diarization-3.1"
    model_path = os.path.join(hf_cache, model_folder)

    if os.path.exists(model_path):
        snapshots_path = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_path) and os.listdir(snapshots_path):
            return True
    return False


def diarize_audio(audio_path: str, progress_callback=None) -> list[dict]:
    """
    Perform speaker diarization on an audio file using Pyannote.

    Identifies different speakers and when they spoke.

    Args:
        audio_path (str): Path to the audio file.
        progress_callback: Optional callback for progress updates.

    Returns:
        list[dict]: List of speaker segments with keys:
            - speaker (str): Speaker label (e.g., "SPEAKER_00")
            - start (float): Start time in seconds
            - end (float): End time in seconds

    Raises:
        RuntimeError: If diarization fails or token not available.
    """
    from pyannote.audio import Pipeline
    import torch
    import torchaudio

    token = get_huggingface_token()
    if not token:
        raise RuntimeError(
            "HuggingFace token required for speaker diarization. "
            "Set HUGGINGFACE_TOKEN environment variable."
        )

    try:
        # Load audio using torchaudio (bypasses torchcodec issues)
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if needed (pyannote expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Create audio dict for pyannote (bypasses built-in audio loading)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        # Load the diarization pipeline
        pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL_ID,
            token=token
        )

        # Run diarization
        diarization = pipeline(audio_input)

        # Extract speaker segments (new API uses .speaker_diarization)
        segments = []
        for turn, speaker in diarization.speaker_diarization:
            segments.append({
                "speaker": f"Speaker {speaker}",
                "start": turn.start,
                "end": turn.end
            })

        return segments

    except Exception as e:
        raise RuntimeError(f"Diarization failed: {str(e)}")


def transcribe_with_timestamps(audio_path: str) -> list[dict]:
    """
    Transcribe audio and return segments with timestamps.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        list[dict]: List of text segments with keys:
            - text (str): Transcribed text
            - start (float): Start time in seconds
            - end (float): End time in seconds
    """
    from parakeet_mlx import from_pretrained

    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = from_pretrained(PARAKEET_MODEL_ID)
    result = model.transcribe(str(audio_file))

    # Extract segments with timestamps from sentences
    segments = []
    for sentence in result.sentences:
        segments.append({
            "text": sentence.text.strip(),
            "start": sentence.start,
            "end": sentence.end
        })

    return segments


def align_transcription_with_speakers(
    text_segments: list[dict],
    speaker_segments: list[dict]
) -> str:
    """
    Align transcribed text segments with speaker diarization.

    Uses midpoint matching: each text segment is assigned to the speaker
    who was active at the segment's midpoint.

    Args:
        text_segments: List of dicts with text, start, end
        speaker_segments: List of dicts with speaker, start, end

    Returns:
        str: Formatted transcription with speaker labels.
    """
    if not speaker_segments:
        # No diarization, return plain text
        return "\n".join(seg["text"] for seg in text_segments)

    aligned = []
    current_speaker = None
    current_text = []

    for text_seg in text_segments:
        # Find speaker at midpoint of text segment
        midpoint = (text_seg["start"] + text_seg["end"]) / 2
        speaker = "UNKNOWN"

        for spk_seg in speaker_segments:
            if spk_seg["start"] <= midpoint <= spk_seg["end"]:
                speaker = spk_seg["speaker"]
                break

        # Group consecutive segments by same speaker
        if speaker == current_speaker:
            current_text.append(text_seg["text"])
        else:
            # Save previous speaker's text
            if current_speaker and current_text:
                aligned.append(f"{current_speaker}: {' '.join(current_text)}")
            # Start new speaker
            current_speaker = speaker
            current_text = [text_seg["text"]]

    # Don't forget the last speaker
    if current_speaker and current_text:
        aligned.append(f"{current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(aligned)


def transcribe_with_diarization(
    audio_path: str,
    progress_callback=None
) -> tuple[str, bool]:
    """
    Transcribe audio with speaker identification.

    Combines Parakeet transcription with Pyannote diarization.
    Falls back to plain transcription if diarization unavailable.

    Args:
        audio_path (str): Path to the audio file.
        progress_callback: Optional callback(progress, stage) where:
            - progress: 0.0 to 1.0
            - stage: "transcribing" or "diarizing"

    Returns:
        tuple[str, bool]: (transcription_text, diarization_used)
            - transcription_text: The transcribed text with speaker labels
            - diarization_used: Whether diarization was applied

    Raises:
        RuntimeError: If transcription fails.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    supported_formats = {'.mp3', '.wav', '.m4a', '.flac'}
    if audio_file.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported audio format: {audio_file.suffix}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    try:
        # Step 1: Transcribe with timestamps
        if progress_callback:
            progress_callback(0.0, "transcribing")

        text_segments = transcribe_with_timestamps(audio_path)

        if progress_callback:
            progress_callback(0.5, "transcribing")

        # Step 2: Try diarization if available
        diarization_used = False
        speaker_segments = []
        diarization_error = None

        if is_diarization_available():
            try:
                if progress_callback:
                    progress_callback(0.5, "diarizing")

                speaker_segments = diarize_audio(audio_path)
                diarization_used = True

                if progress_callback:
                    progress_callback(1.0, "diarizing")
            except Exception as e:
                # Diarization failed, store error for debugging
                diarization_error = str(e)
                print(f"Diarization error: {diarization_error}")

        # Step 3: Align transcription with speakers
        result = align_transcription_with_speakers(text_segments, speaker_segments)

        return result, diarization_used

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")
