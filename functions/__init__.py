"""
Here you will find the agent tool functions, and will also the functions
that manage all the data behind the scenes.
"""

from .agent_tools import read_folder, read_template, save_summary, save_homework, save_session_draft, verify_document_draft, save_initial_persona
from .logger import log_separator, log_tool_call
from .transcription import (
    transcribe_audio,
    save_transcription,
    is_model_cached,
    is_diarization_available,
    is_diarization_model_cached,
    transcribe_with_diarization,
)
from .transcription_deepgram import (
    is_deepgram_available,
    transcribe_with_deepgram,
    DeepgramLiveTranscriber,
    convert_audio_to_deepgram_format,
)
from .document_orchestrator import create_document, detect_document_type
from .retriever_tool import retriever_tool
from .agent_tools import read_document


print("Function module loaded...")