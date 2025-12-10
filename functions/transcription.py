"""
Transcription functions using Parakeet-MLX for audio-to-text conversion.
Optimized for Apple Silicon (M1/M2/M3).
"""

from pathlib import Path
from config.paths import MODEL_PATH


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using Parakeet-MLX model.

    Args:
        audio_path (str): Path to the audio file to transcribe.
            Supported formats: mp3, wav, m4a, flac

    Returns:
        str: The transcribed text with punctuation and capitalization.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio format is not supported.
        RuntimeError: If transcription fails.
    """
    from parakeet_mlx import from_pretrained

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
        model = from_pretrained(MODEL_PATH)
        result = model.transcribe(str(audio_file))
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
