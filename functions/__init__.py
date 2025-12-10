"""
Here you will find the agent tool functions, and will also the functions
that manage all the data behind the scenes.
"""

from .agent_tools import read_folder, read_template, save_summary, save_homework, save_session_draft
from .transcription import (
    transcribe_audio,
    save_transcription,
    is_model_cached,
    is_diarization_available,
    is_diarization_model_cached,
    transcribe_with_diarization,
)


print("Function module loaded...")