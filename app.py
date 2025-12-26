"""
Life Coach AI Assistant - Streamlit Application
A multi-tab interface for managing coaching clients and sessions.
"""

#run by writing:
#   streamlit run app.py


import streamlit as st
import os
from datetime import datetime
from io import BytesIO

# File parsing imports
from docx import Document as DocxDocument
import PyPDF2
from pathlib import Path

# Import agents (to be created by user)
from agents import session_agent

# Import transcription functions
from functions import (
    transcribe_audio,
    save_transcription,
    is_model_cached,
    is_diarization_available,
    is_diarization_model_cached,
    transcribe_with_diarization,
    is_deepgram_available,
    transcribe_with_deepgram,
    DeepgramLiveTranscriber,
)

# Import audio capture module
from audio_capture import get_audio_capturer, get_platform_info

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Life Coach AI Assistant",
    page_icon="🌿",
    layout="wide"
)

# Placeholder for Brand Logo
# Replace 'path/to/logo.png' with the actual file path when available
# st.logo("branding_docs/logo.png", link="https://your-website.com")

def apply_custom_styling():
    """Inject custom CSS for Premium Branding"""
    st.markdown("""
        <style>
        /* Import Font (Optional - relying on system fonts for now, but can import Google Fonts) */
        
        /* Dual Typography: Sans-Serif for Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Helvetica', 'Arial', sans-serif !important;
            color: #2E3B1F !important; /* Deep Brand Green */
        }
        
        /* Dual Typography: Serif for Body */
        .stApp, .stMarkdown, p, div {
            font-family: 'Georgia', 'Times New Roman', serif;
        }

        /* Card-Style Containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FAF9F6;
            border: 1px solid #A0522D; /* Terracotta border */
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }

        /* Button Styling - Terracotta */
        div.stButton > button:first-child {
            background-color: #A0522D;
            color: white;
            border-radius: 8px;
            border: none;
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #8a4524;
            color: white;
        }

        /* Sidebar Styling - Text Color Overrides for Dark Background */
        [data-testid="stSidebar"] * {
            color: #F4F1EA !important;
        }
        
        /* Sidebar Title Specifics - Reinforcing, though the wildcard above covers it */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #F4F1EA !important;
        }

        /* Sidebar Metric Values (Big Numbers) */
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #F4F1EA !important;
        }

        /* Input Fields in Sidebar - Light background for readability, reset text to dark for inside inputs */
        [data-testid="stSidebar"] input {
            background-color: #F4F1EA !important;
            color: #2C2C2C !important;
        }
        
        /* FIX: Reset Widget Backgrounds to Light for Contrast */
        
        /* Text Input (Client Name) */
        div[data-testid="stTextInput"] div[data-baseweb="input"] {
            background-color: #FAF9F6 !important;
            border: 1px solid #4A5D23 !important;
        }
        /* Crucial Fix: Force the input element itself to be light */
        div[data-testid="stTextInput"] input {
            background-color: #FAF9F6 !important;
            color: #2C2C2C !important;
        }

        /* Selectbox (Select Client) - Targeting the inner container */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #FAF9F6 !important;
            border: 1px solid #4A5D23 !important;
            color: #2C2C2C !important;
        }
        /* Ensure the text inside the selectbox is dark */
        div[data-testid="stSelectbox"] div[data-testid="stMarkdownContainer"] p {
            color: #2C2C2C !important;
        }
        
        /* Date Input Box */
        div[data-testid="stDateInput"] div[data-baseweb="input"] {
            background-color: #FAF9F6 !important;
            border: 1px solid #4A5D23 !important;
        }
        div[data-testid="stDateInput"] input {
            color: #2C2C2C !important;
        }

        /* File Uploader Dropzone */
        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
            background-color: #FAF9F6 !important;
            color: #2C2C2C !important;
            border: 1px dashed #4A5D23 !important;
        }
        div[data-testid="stFileUploader"] div, div[data-testid="stFileUploader"] small {
            color: #2C2C2C !important;
        }
        /* Icon color in file uploader */
        div[data-testid="stFileUploader"] svg {
            fill: #4A5D23 !important;
        }

        /* Chat Input Styling - Force Light Background */
        div[data-testid="stChatInput"] {
            background-color: #FAF9F6 !important;
            border-color: #4A5D23 !important;
        }
        div[data-testid="stChatInput"] textarea {
            background-color: #FAF9F6 !important;
            color: #2C2C2C !important;
            caret-color: #A0522D !important; /* Terracotta cursor */
        }
        div[data-testid="stChatInput"] button {
            color: #4A5D23 !important; /* Send button color */
        }

        /* General override for labels above widgets if they aren't catching the header rule */
        .stMarkdown label p {
            color: #2C2C2C !important; /* Keep labels dark on the main light background */
        }
        
        /* Chat Message Styling */
        .stChatMessage {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #E0E0E0;
        }

        /* Text Area Styling - Light Background */
        div[data-testid="stTextArea"] textarea {
            background-color: #FAF9F6 !important;
            color: #2C2C2C !important;
        }
        div[data-testid="stTextArea"] div[data-baseweb="textarea"] {
            background-color: #FAF9F6 !important;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styling()

DATA_PATH = "LifeCoach_Data"
ACTIVE_PATH = os.path.join(DATA_PATH, "Active")
INACTIVE_PATH = os.path.join(DATA_PATH, "Inactive")
UNDEFINED_PATH = os.path.join(DATA_PATH, "Undefined")

# Ensure directories exist
for path in [ACTIVE_PATH, INACTIVE_PATH, UNDEFINED_PATH]:
    os.makedirs(path, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_uploaded_file(uploaded_file) -> str:
    """Read content from uploaded file (.txt, .docx, .pdf)"""
    if uploaded_file is None:
        return ""

    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'txt':
        return uploaded_file.read().decode('utf-8')

    elif file_extension == 'docx':
        doc = DocxDocument(BytesIO(uploaded_file.read()))
        return '\n'.join([para.text for para in doc.paragraphs])

    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)

    return ""


def sanitize_client_name(name: str) -> str:
    """
    Sanitize client name for use as folder name.
    Replaces spaces with underscores and removes special characters.

    Example: "Pedro Perez" -> "Pedro_Perez"
    """
    if not name:
        return name
    # Replace spaces with underscores
    sanitized = name.strip().replace(" ", "_")
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in "_-")
    return sanitized


def get_active_clients() -> list:
    """Get list of active client folders"""
    if not os.path.exists(ACTIVE_PATH):
        return []
    return [d for d in os.listdir(ACTIVE_PATH)
            if os.path.isdir(os.path.join(ACTIVE_PATH, d))]


def get_undefined_clients() -> list:
    """Get list of undefined client folders"""
    if not os.path.exists(UNDEFINED_PATH):
        return []
    return [d for d in os.listdir(UNDEFINED_PATH)
            if os.path.isdir(os.path.join(UNDEFINED_PATH, d))]


def get_client_sessions(client_name: str) -> list:
    """Get list of existing session folders for a client, sorted newest first."""
    client_path = os.path.join(ACTIVE_PATH, client_name)
    if not os.path.exists(client_path):
        return []
    sessions = [d for d in os.listdir(client_path)
                if d.startswith("Session_") and os.path.isdir(os.path.join(client_path, d))]
    # Sort by session number descending (newest first)
    sessions.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    return sessions


def load_session_transcription(client_name: str, session_folder: str) -> str | None:
    """Load transcription.txt from a session folder if it exists."""
    transcription_path = os.path.join(ACTIVE_PATH, client_name, session_folder, "transcription.txt")
    if os.path.exists(transcription_path):
        with open(transcription_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def load_session_documents(client_name: str, session_folder: str) -> dict:
    """Load existing documents (summary, homework, next_session) from a session folder."""
    session_path = os.path.join(ACTIVE_PATH, client_name, session_folder)
    documents = {}

    doc_files = {
        "summary": "summary.txt",
        "homework": "homework.txt",
        "next_session": "next_session.txt"
    }

    for key, filename in doc_files.items():
        file_path = os.path.join(session_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents[key] = f.read()

    return documents


def get_next_session_number_for_path(client_name: str, base_path: str) -> int:
    """Get the next session number for a client in a specific path"""
    client_path = os.path.join(base_path, client_name)
    if not os.path.exists(client_path):
        return 1
    session_folders = [d for d in os.listdir(client_path)
                       if d.startswith("Session_")]
    return len(session_folders) + 1


def create_session_folder_for_transcription(client_name: str, client_type: str, session_date) -> tuple:
    """
    Create session folder based on client type.

    Undefined clients: Always Session_1 (first session for new clients)
    Active clients: Session_N+1 (next sequential number)

    Returns:
        tuple: (full_path, folder_name, sanitized_client_name)
    """
    # Sanitize client name for folder creation
    safe_client_name = sanitize_client_name(client_name)

    if client_type == "Undefined":
        base_path = UNDEFINED_PATH
        session_num = get_next_session_number_for_path(safe_client_name, base_path)
    else:  # Active
        base_path = ACTIVE_PATH
        session_num = get_next_session_number_for_path(safe_client_name, base_path)

    folder_name = f"Session_{session_num}_{session_date.strftime('%d-%m-%Y')}"
    full_path = os.path.join(base_path, safe_client_name, folder_name)
    os.makedirs(full_path, exist_ok=True)

    return full_path, folder_name, safe_client_name


def save_audio_file(session_path: str, uploaded_file) -> str:
    """
    Save original audio file to session folder.

    Returns:
        str: Path to saved audio file
    """
    original_name = Path(uploaded_file.name).stem
    extension = Path(uploaded_file.name).suffix
    audio_filename = f"audio_{original_name}{extension}"
    audio_path = os.path.join(session_path, audio_filename)

    with open(audio_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    return audio_path


def get_next_session_number(client_name: str) -> int:
    """Get the next session number for a client"""
    client_path = os.path.join(ACTIVE_PATH, client_name)
    if not os.path.exists(client_path):
        return 1

    session_folders = [d for d in os.listdir(client_path)
                       if d.startswith("Session_")]
    return len(session_folders) + 1


def save_generated_document(content: str, path: str, filename: str) -> str:
    """Save a generated document and return the full path"""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return full_path


def get_or_create_session_folder(client_name: str, session_date) -> str:
    """Get existing session folder from state or create new one"""
    # Reset if client or date changed
    if (st.session_state.current_active_client != client_name or
        st.session_state.get("current_session_date") != session_date):
        st.session_state.current_session_folder = None
        st.session_state.current_active_client = client_name
        st.session_state.current_session_date = session_date

    # Calculate new session folder if needed
    if st.session_state.current_session_folder is None:
        session_num = get_next_session_number(client_name)
        st.session_state.current_session_folder = f"Session_{session_num}_{session_date.strftime('%d-%m-%Y')}"

    return st.session_state.current_session_folder


def strip_context_tags(text: str) -> str:
    """Remove context tags like [Client: X] [Session: Y] from display text"""
    import re
    # Remove patterns like [Client: ...] [Session: ...] [Uploaded file: ...]
    cleaned = re.sub(r'\[Client:[^\]]*\]\s*', '', text)
    cleaned = re.sub(r'\[Session:[^\]]*\]\s*', '', cleaned)
    cleaned = re.sub(r'\[Uploaded file:[^\]]*\]\s*', '', cleaned)
    cleaned = re.sub(r'\[Session transcription:[^\]]*\]\s*', '', cleaned)
    return cleaned.strip()


def invoke_agent(agent, messages: list) -> str:
    """Invoke an agent with the conversation history and return response"""
    from langchain_core.messages import AIMessage, HumanMessage
    from functions import log_separator

    # Log separator for each agent call
    latest_msg = messages[-1]["content"][:50] if messages else "New Session"
    log_separator(f"Agent Call: {latest_msg}...")

    # Convert dict messages to LangChain message objects
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    response = agent.invoke(
        {"messages": lc_messages},
        {"configurable": {"thread_id": "001"}}
    )

    # Handle LangChain message response format
    if isinstance(response, dict):
        # Check for 'messages' key with list of message objects
        if "messages" in response and isinstance(response["messages"], list):
            # Find the last AIMessage in the list
            for msg in reversed(response["messages"]):
                if isinstance(msg, AIMessage):
                    return msg.content
                # Also handle dict format
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")

        # Fallback to other common keys
        if "output" in response:
            return response["output"]
        if "content" in response:
            return response["content"]

    # Handle direct AIMessage response
    if isinstance(response, AIMessage):
        return response.content

    return str(response)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Tab 1: Transcribe Audio
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = None
if "transcription_session_path" not in st.session_state:
    st.session_state.transcription_session_path = None
if "transcription_session_folder" not in st.session_state:
    st.session_state.transcription_session_folder = None

# Tab 1: Audio Recording state
if "audio_capturer" not in st.session_state:
    st.session_state.audio_capturer = None
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "recording_start_time" not in st.session_state:
    st.session_state.recording_start_time = None
if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = None

# Tab 1: Live Transcription state
if "live_transcriber" not in st.session_state:
    st.session_state.live_transcriber = None
if "recording_session_path" not in st.session_state:
    st.session_state.recording_session_path = None
if "completed_session_info" not in st.session_state:
    st.session_state.completed_session_info = None

# Tab 2: Undefined Clients
if "messages_undefined" not in st.session_state:
    st.session_state.messages_undefined = []
if "current_undefined_client" not in st.session_state:
    st.session_state.current_undefined_client = ""
if "discovery_prep_content" not in st.session_state:
    st.session_state.discovery_prep_content = None

# Tab 3: Active Clients
if "messages_active" not in st.session_state:
    st.session_state.messages_active = []
if "session_documents" not in st.session_state:
    st.session_state.session_documents = {}
if "current_session_folder" not in st.session_state:
    st.session_state.current_session_folder = None
if "is_new_session_mode" not in st.session_state:
    st.session_state.is_new_session_mode = False
if "current_active_client" not in st.session_state:
    st.session_state.current_active_client = None
if "loaded_transcription" not in st.session_state:
    st.session_state.loaded_transcription = None
if "transcription_added_to_chat" not in st.session_state:
    st.session_state.transcription_added_to_chat = False
if "documents_added_to_chat" not in st.session_state:
    st.session_state.documents_added_to_chat = False

# Tab 4: Chat Assistant
if "messages_chat" not in st.session_state:
    st.session_state.messages_chat = []


# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("🧘 Life Coach AI Assistant")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️ Transcribe Audio",
    "📋 Undefined Clients",
    "✅ Active Clients",
    "💬 Chat Assistant"
])


# ============================================================================
# TAB 1: AUDIO (RECORD & TRANSCRIBE)
# ============================================================================

with tab1:
    st.header("Audio")

    # Mode selector: Live Session or Transcribe
    audio_mode = st.radio(
        "Mode",
        options=["Have a Live Session", "Transcribe Audio"],
        horizontal=True,
        key="audio_mode"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # LIVE SESSION MODE
    # -------------------------------------------------------------------------
    if audio_mode == "Have a Live Session":
        st.caption("Live session with real-time transcription (Deepgram)")

        # Check platform support
        platform_info = get_platform_info()
        if not platform_info["supported"]:
            st.error(f"Audio recording not supported on {platform_info['platform']}")
            st.info("Requirements: " + ", ".join(platform_info["requirements"]))
        else:
            # Initialize capturer if needed
            if st.session_state.audio_capturer is None:
                try:
                    st.session_state.audio_capturer = get_audio_capturer()
                except Exception as e:
                    st.error(f"Failed to initialize audio capture: {e}")

            if st.session_state.audio_capturer:
                capturer = st.session_state.audio_capturer

                # Auto-select "Session Capture" device
                devices = capturer.get_available_devices()
                selected_device_id = None
                selected_device_name = None

                for d in devices:
                    if "session capture" in d['name'].lower():
                        selected_device_id = d['id']
                        selected_device_name = f"{d['name']} ({d['channels']}ch)"
                        break

                if selected_device_id is None:
                    st.error("⚠️ 'Session Capture' device not found. Please create the aggregate device in Audio MIDI Setup.")
                    st.info("Requirements: BlackHole 2ch + Microphone combined into an aggregate device named 'Session Capture'")
                else:
                    st.caption(f"🎧 {selected_device_name}")

                    # Check Deepgram availability (required for live sessions)
                    if not is_deepgram_available():
                        st.error("⚠️ DEEPGRAM_API_KEY not configured. Add it to .env for live transcription.")

                    # Client and date selection
                    rec_col1, rec_col2 = st.columns([1, 1])

                    with rec_col1:
                        rec_client_type = st.radio(
                            "Client Type",
                            options=["Undefined", "Active"],
                            horizontal=True,
                            key="record_client_type"
                        )

                    with rec_col2:
                        rec_date = st.date_input(
                            "Session Date",
                            value=datetime.today(),
                            key="record_date"
                        )

                    # Client selection based on type
                    if rec_client_type == "Undefined":
                        undefined_clients = get_undefined_clients()
                        rec_col_select, rec_col_new = st.columns([1, 1])

                        with rec_col_select:
                            if undefined_clients:
                                rec_selected_existing = st.selectbox(
                                    "Select Existing Client",
                                    options=["-- Create New --"] + undefined_clients,
                                    key="record_undefined_client_select"
                                )
                            else:
                                rec_selected_existing = "-- Create New --"
                                st.info("No existing undefined clients.")

                        with rec_col_new:
                            rec_new_client_name = st.text_input(
                                "Or Enter New Client Name",
                                placeholder="Enter new client name...",
                                key="record_new_undefined_client"
                            )

                        if rec_new_client_name:
                            rec_final_client = rec_new_client_name
                        elif rec_selected_existing and rec_selected_existing != "-- Create New --":
                            rec_final_client = rec_selected_existing
                        else:
                            rec_final_client = None
                    else:
                        active_clients = get_active_clients()
                        if active_clients:
                            rec_final_client = st.selectbox(
                                "Select Active Client",
                                options=active_clients,
                                key="record_active_client_select"
                            )
                        else:
                            st.warning("No active clients found.")
                            rec_final_client = None

                    st.divider()

                    # Recording controls
                    if st.session_state.is_recording:
                        # Show recording status
                        duration = capturer.get_recording_duration()
                        minutes = int(duration // 60)
                        seconds = int(duration % 60)

                        st.warning(f"🔴 Live Session... {minutes:02d}:{seconds:02d}")

                        # Audio LED-style indicators
                        mic_level, system_level = capturer.get_audio_levels()
                        channel_count = capturer.get_channel_count()

                        # Scale levels (RMS typically 0-0.3 for speech)
                        mic_scaled = min(mic_level * 5, 1.0)
                        system_scaled = min(system_level * 5, 1.0)

                        def led_dots(level):
                            """Generate LED-style dots based on audio level"""
                            dots = ["⚫", "⚫", "⚫", "⚫", "⚫"]
                            active_count = int(level * 5)
                            for i in range(active_count):
                                if i < 2:
                                    dots[i] = "🟢"
                                elif i < 4:
                                    dots[i] = "🟡"
                                else:
                                    dots[i] = "🔴"
                            return " ".join(dots)

                        # Display LED indicators based on channel count
                        if channel_count >= 3:
                            st.markdown(f"🎤 Mic: {led_dots(mic_scaled)}  &nbsp;&nbsp;&nbsp;  🔊 System: {led_dots(system_scaled)}")
                        elif channel_count == 2:
                            st.markdown(f"🎤 Input: {led_dots(mic_scaled)}  &nbsp;&nbsp;&nbsp;  🔊 Audio: {led_dots(system_scaled)}")
                        else:
                            st.markdown(f"🎤 Input: {led_dots(mic_scaled)}")

                        if st.button("⏹️ End Session", type="primary", key="stop_recording"):
                            # Stop audio capture
                            saved_path = capturer.stop_recording()
                            st.session_state.is_recording = False
                            st.session_state.recorded_audio_path = saved_path

                            # Stop live transcriber and save transcript
                            if st.session_state.live_transcriber:
                                transcript = st.session_state.live_transcriber.stop()
                                st.session_state.live_transcriber = None

                                # Save transcription to session folder
                                if transcript and st.session_state.recording_session_path:
                                    save_transcription(st.session_state.recording_session_path, transcript)
                                    # Store session info for preview (transcript comes from loaded_transcription)
                                    st.session_state.completed_session_info = {
                                        "audio_path": saved_path,
                                        "session_path": st.session_state.recording_session_path
                                    }
                                    # Reset Tab 3 state so it picks up the new session
                                    st.session_state.current_session_folder = None
                                    st.session_state.current_active_client = None
                                    st.session_state.loaded_transcription = None
                                    # Clear the selectbox widget state to force fresh selection
                                    if "session_select" in st.session_state:
                                        del st.session_state["session_select"]
                                else:
                                    st.session_state.completed_session_info = None

                                st.session_state.recording_session_path = None
                            else:
                                st.session_state.completed_session_info = None

                            st.rerun()
                        else:
                            # Auto-refresh while recording (every 0.5 seconds)
                            import time
                            time.sleep(0.5)
                            st.rerun()
                    else:
                        # Start session button
                        can_start = rec_final_client is not None and is_deepgram_available()

                        if st.button(
                            "🎙️ Start Session",
                            type="primary",
                            key="start_recording",
                            disabled=not can_start
                        ):
                            # Create session folder
                            session_path, session_folder, safe_client = create_session_folder_for_transcription(
                                rec_final_client,
                                rec_client_type,
                                rec_date
                            )

                            # Create output path (will be saved as MP3)
                            audio_filename = f"recording_{datetime.now().strftime('%H%M%S')}.mp3"
                            output_path = os.path.join(session_path, audio_filename)

                            # Initialize live transcription (always enabled)
                            sample_rate = capturer.get_device_sample_rate(selected_device_id)
                            transcriber = DeepgramLiveTranscriber(sample_rate=sample_rate)

                            # NOTE: Don't set on_transcript callback - it runs in background thread
                            # and can't access st.session_state. Instead, poll get_transcript() in UI loop.

                            # Start transcriber
                            if transcriber.start():
                                st.session_state.live_transcriber = transcriber
                                st.session_state.recording_session_path = session_path
                                audio_callback = transcriber.send_audio

                                # Clear previous session preview
                                st.session_state.completed_session_info = None

                                # Start recording with live transcription
                                capturer.start_recording(
                                    device_id=selected_device_id,
                                    output_path=output_path,
                                    on_audio_chunk=audio_callback
                                )
                                st.session_state.is_recording = True
                                st.session_state.recording_start_time = datetime.now()
                                st.rerun()
                            else:
                                st.error("Failed to connect to Deepgram. Check your API key and internet connection.")

                        if not can_start:
                            if not rec_final_client:
                                st.info("Select a client to start the session.")

                    # Show completed session with transcript preview (uses same source as LLM in Tab 3)
                    if st.session_state.completed_session_info:
                        st.divider()
                        st.subheader("✅ Session Complete")
                        st.success(f"Saved to: {st.session_state.completed_session_info['session_path']}")

                        # Audio player
                        if os.path.exists(st.session_state.completed_session_info['audio_path']):
                            st.audio(st.session_state.completed_session_info['audio_path'])

                        # Transcript preview - show what LLM sees (single source of truth)
                        transcript_to_show = st.session_state.loaded_transcription
                        if transcript_to_show:
                            with st.expander("📝 Transcription (same as LLM sees)", expanded=True):
                                st.text_area(
                                    "Full Transcript",
                                    value=transcript_to_show,
                                    height=300,
                                    disabled=True,
                                    key="completed_transcript_preview"
                                )
                        else:
                            st.info("Transcription will be visible in Active Clients tab")

                        # Clear button
                        if st.button("🗑️ Clear & Start New Session", key="clear_completed"):
                            st.session_state.completed_session_info = None
                            st.session_state.recorded_audio_path = None
                            st.rerun()

                    # Show last session recording if no completed session preview
                    elif st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                        st.divider()
                        st.subheader("Last Session")
                        st.audio(st.session_state.recorded_audio_path)
                        st.caption(f"Saved to: {st.session_state.recorded_audio_path}")

    # -------------------------------------------------------------------------
    # TRANSCRIBE AUDIO MODE
    # -------------------------------------------------------------------------
    else:
        st.caption("Transcribe session recordings and create session folders")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Client type selector
            client_type = st.radio(
                "Client Type",
                options=["Undefined", "Active"],
                horizontal=True,
                key="transcribe_client_type"
            )

        with col2:
            # Date picker
            transcribe_date = st.date_input(
                "Session Date",
                value=datetime.today(),
                key="transcribe_date"
            )

        # Client selection based on type
        if client_type == "Undefined":
            undefined_clients = get_undefined_clients()
            col_select, col_new = st.columns([1, 1])

            with col_select:
                if undefined_clients:
                    selected_existing = st.selectbox(
                        "Select Existing Client",
                        options=["-- Create New --"] + undefined_clients,
                        key="undefined_client_select"
                    )
                else:
                    selected_existing = "-- Create New --"
                    st.info("No existing undefined clients. Enter a new name below.")

            with col_new:
                new_client_name = st.text_input(
                    "Or Enter New Client Name",
                    placeholder="Enter new client name...",
                    key="new_undefined_client_name"
                )

            # Determine final client name
            if new_client_name:
                final_client_name = new_client_name
            elif selected_existing and selected_existing != "-- Create New --":
                final_client_name = selected_existing
            else:
                final_client_name = None

        else:  # Active
            active_clients = get_active_clients()
            if active_clients:
                final_client_name = st.selectbox(
                    "Select Active Client",
                    options=active_clients,
                    key="active_client_select_transcribe"
                )
            else:
                st.warning("No active clients found. Please add clients from the Undefined Clients tab first.")
                final_client_name = None

        st.divider()

        # Transcription method selector
        transcription_options = ["Locally - Slow, Free"]
        if is_deepgram_available():
            transcription_options.append("Deepgram - Fast, $0.26/hr")

        transcription_method = st.selectbox(
            "Transcription Method",
            options=transcription_options,
            key="transcription_method",
            help="Choose how to transcribe the audio"
        )

        # Show info based on selected method
        if transcription_method == "Locally - Slow, Free":
            # Speaker diarization status (always enabled for local)
            diarization_available = is_diarization_available()
            if not diarization_available:
                st.caption("⚠️ Speaker identification unavailable. Set `HUGGINGFACE_TOKEN` in .env and install `pyannote.audio`.")
            elif not is_diarization_model_cached():
                st.caption("📥 First use will download the speaker identification model (~200MB)")
            else:
                st.caption("✓ Speaker identification enabled")
        else:
            st.caption("✓ Fast cloud transcription with built-in speaker identification")

        # Tip if Deepgram not configured
        if not is_deepgram_available():
            st.caption("💡 Add `DEEPGRAM_API_KEY` to .env for faster cloud transcription")

        # Audio file uploader
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=['mp3', 'wav', 'm4a', 'flac'],
            key="audio_file_upload"
        )

        if audio_file:
            st.audio(audio_file)
            st.caption(f"File: {audio_file.name} ({audio_file.size / 1024 / 1024:.2f} MB)")

        # Transcribe button
        transcribe_col1, transcribe_col2 = st.columns([1, 3])

        with transcribe_col1:
            transcribe_button = st.button(
                "🎙️ Transcribe",
                type="primary",
                key="transcribe_button",
                disabled=not (final_client_name and audio_file)
            )

        if transcribe_button and final_client_name and audio_file:
            # Validation
            if audio_file.size == 0:
                st.error("The uploaded file appears to be empty.")
            else:
                try:
                    # Create session folder (returns sanitized client name)
                    session_path, session_folder, safe_client_name = create_session_folder_for_transcription(
                        final_client_name,
                        client_type,
                        transcribe_date
                    )

                    # Save audio file
                    audio_path = save_audio_file(session_path, audio_file)

                    # Transcribe based on selected method
                    if transcription_method.startswith("Deepgram"):
                        # Deepgram cloud transcription (fast, with built-in diarization)
                        with st.spinner("☁️ Transcribing with Deepgram..."):
                            transcription_text, diarization_used = transcribe_with_deepgram(audio_path)
                        st.success("✓ Transcription complete (Deepgram)")

                    else:
                        # Local transcription with Parakeet + Pyannote
                        model_cached = is_model_cached()
                        diarization_available = is_diarization_available()
                        diarization_model_cached = is_diarization_model_cached() if diarization_available else True

                        # Show download messages if needed
                        if not model_cached:
                            st.info("📥 First time setup: Downloading transcription model (~600MB). This only happens once...")

                        if diarization_available and not diarization_model_cached:
                            st.info("📥 Downloading speaker diarization model (~200MB). This only happens once...")

                        # Use diarization if available (always for coaching sessions)
                        if diarization_available:
                            with st.spinner("🎙️ Transcribing and identifying speakers..."):
                                transcription_text, diarization_used = transcribe_with_diarization(audio_path)

                            if diarization_used:
                                st.success("✓ Speaker identification applied")
                            else:
                                st.warning("Speaker identification failed, using plain transcription")
                        else:
                            # Fallback: plain transcription without diarization
                            with st.spinner("🎙️ Transcribing audio..."):
                                transcription_text = transcribe_audio(audio_path)
                            st.success("✓ Transcription complete")

                    # Save transcription
                    save_transcription(session_path, transcription_text)

                    # Store in session state
                    st.session_state.transcription_result = transcription_text
                    st.session_state.transcription_session_path = session_path
                    st.session_state.transcription_session_folder = session_folder

                    st.success(f"Transcription complete! Saved to: {safe_client_name}/{session_folder}/")

                except FileNotFoundError as e:
                    st.error(f"File error: {str(e)}")
                except ValueError as e:
                    st.error(f"Invalid file: {str(e)}")
                except RuntimeError as e:
                    st.error(f"Transcription failed: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

        # Display transcription result
        if st.session_state.transcription_result:
            st.divider()
            st.subheader("Transcription Result")

            if st.session_state.transcription_session_folder:
                st.caption(f"Session: {st.session_state.transcription_session_folder}")

            # Display in text area
            st.text_area(
                "Transcription",
                value=st.session_state.transcription_result,
                height=300,
                key="transcription_display",
                disabled=True
            )

            # Download button
            st.download_button(
                label="📥 Download Transcription",
                data=st.session_state.transcription_result,
                file_name="transcription.txt",
                mime="text/plain",
                key="download_transcription"
            )


# ============================================================================
# TAB 2: UNDEFINED CLIENTS
# ============================================================================

with tab2:
    st.header("New Client Discovery")

    col1, col2 = st.columns([2, 1])

    with col1:
        client_name = st.text_input(
            "Client Name",
            value=st.session_state.current_undefined_client,
            placeholder="Enter new client name...",
            key="undefined_client_name"
        )
        if client_name:
            st.session_state.current_undefined_client = client_name

    with col2:
        st.write("")  # Spacing
        st.write("")
        if st.session_state.current_undefined_client:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Move to Active", type="primary", key="move_active"):
                    # Agent handles the move
                    move_msg = f"Move client {st.session_state.current_undefined_client} to Active folder."
                    st.session_state.messages_undefined.append({
                        "role": "user",
                        "content": move_msg
                    })
                    with st.spinner("Moving client..."):
                        response = invoke_agent(
                            discovery_agent,
                            st.session_state.messages_undefined
                        )
                        st.session_state.messages_undefined.append({
                            "role": "assistant",
                            "content": response
                        })
                    st.rerun()

            with col_b:
                if st.button("Move to Inactive", key="move_inactive"):
                    move_msg = f"Move client {st.session_state.current_undefined_client} to Inactive folder."
                    st.session_state.messages_undefined.append({
                        "role": "user",
                        "content": move_msg
                    })
                    with st.spinner("Moving client..."):
                        response = invoke_agent(
                            discovery_agent,
                            st.session_state.messages_undefined
                        )
                        st.session_state.messages_undefined.append({
                            "role": "assistant",
                            "content": response
                        })
                    st.rerun()

    st.divider()

    # Chat messages display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages_undefined:
            with st.chat_message(message["role"]):
                display_text = strip_context_tags(message["content"]) if message["role"] == "user" else message["content"]
                st.markdown(display_text)

    # Download button for discovery prep
    if st.session_state.discovery_prep_content:
        st.download_button(
            label="📥 Download Discovery Prep",
            data=st.session_state.discovery_prep_content,
            file_name=f"discovery_prep_{st.session_state.current_undefined_client}.txt",
            mime="text/plain"
        )

    # Chat input with file upload
    uploaded_file = st.file_uploader(
        "Upload survey answers or documents",
        type=['txt', 'docx', 'pdf'],
        key="undefined_file_upload"
    )

    if uploaded_file:
        file_content = read_uploaded_file(uploaded_file)
        if file_content:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            # Add file content to context
            file_message = f"[Uploaded file: {uploaded_file.name}]\n\n{file_content}"
            if not any(file_message in m.get("content", "") for m in st.session_state.messages_undefined):
                st.session_state.messages_undefined.append({
                    "role": "user",
                    "content": file_message
                })

    # Chat input
    if prompt := st.chat_input(
        "Type your notes or instructions for the discovery agent...",
        key="undefined_chat_input"
    ):
        if not st.session_state.current_undefined_client:
            st.warning("Please enter a client name first.")
        else:
            # Add context about current client
            full_prompt = f"[Client: {st.session_state.current_undefined_client}]\n{prompt}"
            st.session_state.messages_undefined.append({
                "role": "user",
                "content": full_prompt
            })

            with st.spinner("Discovery agent is thinking..."):
                response = invoke_agent(
                    discovery_agent,
                    st.session_state.messages_undefined
                )
                st.session_state.messages_undefined.append({
                    "role": "assistant",
                    "content": response
                })

                # Check if discovery prep was generated
                if "discovery_prep" in response.lower() or "saved" in response.lower():
                    prep_path = os.path.join(
                        UNDEFINED_PATH,
                        st.session_state.current_undefined_client,
                        "discovery_prep.txt"
                    )
                    if os.path.exists(prep_path):
                        with open(prep_path, 'r', encoding='utf-8') as f:
                            st.session_state.discovery_prep_content = f.read()

            st.rerun()


# ============================================================================
# TAB 3: ACTIVE CLIENTS
# ============================================================================

with tab3:
    st.header("Active Client Sessions")

    # Row 1: Client selector
    active_clients = get_active_clients()
    if active_clients:
        selected_client = st.selectbox(
            "Select Client",
            options=active_clients,
            key="active_client_select"
        )
    else:
        st.info("No active clients found. Add clients from the Undefined tab.")
        selected_client = None

    if selected_client:
        # Reset state if client changed
        if st.session_state.current_active_client != selected_client:
            st.session_state.current_active_client = selected_client
            st.session_state.loaded_transcription = None
            st.session_state.transcription_added_to_chat = False
            st.session_state.documents_added_to_chat = False
            st.session_state.current_session_folder = None
            st.session_state.is_new_session_mode = False
            # Auto-clear LLM context when client changes
            st.session_state.messages_active = []
            st.session_state.session_documents = {}

        # Row 2: Session selection (existing vs new)
        existing_sessions = get_client_sessions(selected_client)

        session_options = ["+ Create New Session"] + existing_sessions
        # Default to latest session (index 1) if sessions exist, otherwise "Create New" (index 0)
        default_index = 1 if existing_sessions else 0
        selected_option = st.selectbox(
            "Select Session",
            options=session_options,
            index=default_index,
            key="session_select"
        )

        if selected_option == "+ Create New Session":
            # Show date picker for new session (empty by default to force selection)
            session_date = st.date_input(
                "Session Date (select to enable upload)",
                value=None,
                key="session_date"
            )
            if session_date:
                session_folder = get_or_create_session_folder(selected_client, session_date)
                st.caption(f"New session: {session_folder}")
                date_selected = True
            else:
                session_folder = None
                st.info("Select a date to create a new session")
                date_selected = False
            is_new_session = True
        else:
            # Use existing session
            session_folder = selected_option
            st.caption(f"Selected session: {session_folder}")
            is_new_session = False
            date_selected = True  # Existing sessions already have a folder

        # Reset state if switching to new session mode (clears context from previous session)
        if is_new_session and not st.session_state.is_new_session_mode:
            st.session_state.loaded_transcription = None
            st.session_state.transcription_added_to_chat = False
            st.session_state.documents_added_to_chat = False
            st.session_state.messages_active = []
            st.session_state.session_documents = {}
        st.session_state.is_new_session_mode = is_new_session

        # Reset state if session changed
        if st.session_state.current_session_folder != session_folder:
            st.session_state.current_session_folder = session_folder
            st.session_state.loaded_transcription = None
            st.session_state.transcription_added_to_chat = False
            st.session_state.documents_added_to_chat = False
            # Auto-clear LLM context when session changes
            st.session_state.messages_active = []
            st.session_state.session_documents = {}

        # Auto-load transcription for existing sessions
        session_path = os.path.join(ACTIVE_PATH, selected_client, session_folder) if session_folder else None

        if not is_new_session and st.session_state.loaded_transcription is None:
            transcription = load_session_transcription(selected_client, session_folder)
            if transcription:
                st.session_state.loaded_transcription = transcription

        # Auto-load existing documents for existing sessions
        if not is_new_session and not st.session_state.session_documents:
            loaded_docs = load_session_documents(selected_client, session_folder)
            if loaded_docs:
                st.session_state.session_documents = loaded_docs

        # Show transcription status and preview
        if st.session_state.loaded_transcription:
            with st.expander("View Transcription", expanded=False):
                st.text_area(
                    "Transcription Content",
                    value=st.session_state.loaded_transcription,
                    height=200,
                    disabled=True,
                    key=f"transcription_preview_{st.session_state.current_session_folder}"
                )

            # Add transcription to chat context if not already added
            if not st.session_state.transcription_added_to_chat:
                transcription_message = f"[Session transcription loaded]\n\n{st.session_state.loaded_transcription}"
                if not any("[Session transcription" in m.get("content", "") for m in st.session_state.messages_active):
                    st.session_state.messages_active.append({
                        "role": "user",
                        "content": transcription_message
                    })
                st.session_state.transcription_added_to_chat = True
        elif not is_new_session:
            st.info("No transcription found for this session")

        # Add documents to chat context if not already added
        if st.session_state.session_documents:
            if not st.session_state.documents_added_to_chat:
                for doc_type, content in st.session_state.session_documents.items():
                    marker = f"[Session {doc_type} loaded]"
                    if not any(marker in m.get("content", "") for m in st.session_state.messages_active):
                        st.session_state.messages_active.append({
                            "role": "user",
                            "content": f"{marker}\n\n{content}"
                        })
                st.session_state.documents_added_to_chat = True

        # Upload transcription
        st.divider()
        upload_label = "Upload transcription" if is_new_session else "Or upload a different transcription"
        uploaded_file = st.file_uploader(
            upload_label,
            type=['txt', 'docx', 'pdf'],
            key="active_file_upload",
            disabled=not date_selected
        )

        if uploaded_file and date_selected and session_folder:
            file_content = read_uploaded_file(uploaded_file)
            if file_content:
                # Save transcription to client's session folder
                session_path = os.path.join(ACTIVE_PATH, selected_client, session_folder)
                os.makedirs(session_path, exist_ok=True)

                transcription_path = os.path.join(session_path, "transcription.txt")
                with open(transcription_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)

                st.success(f"Transcription saved to: {selected_client}/{session_folder}/transcription.txt")

                # Update loaded transcription
                st.session_state.loaded_transcription = file_content

                # Add to chat context
                file_message = f"[Session transcription: {uploaded_file.name}]\n\n{file_content}"
                if not any(file_message in m.get("content", "") for m in st.session_state.messages_active):
                    st.session_state.messages_active.append({
                        "role": "user",
                        "content": file_message
                    })
                st.session_state.transcription_added_to_chat = True

        st.divider()

        # Chat messages display
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages_active:
                with st.chat_message(message["role"]):
                    # Condense loaded document messages for display (full content still in LLM context)
                    content = message["content"]
                    if "[Session transcription" in content:
                        display_text = "📄 Session transcription loaded"
                    elif "[Session summary loaded]" in content:
                        display_text = "📄 Summary loaded"
                    elif "[Session homework loaded]" in content:
                        display_text = "📄 Homework loaded"
                    elif "[Session next_session loaded]" in content:
                        display_text = "📄 Next Session Draft loaded"
                    elif message["role"] == "user":
                        display_text = strip_context_tags(content)
                    else:
                        display_text = content
                    st.markdown(display_text)

        # Chat input (disabled until transcription is loaded)
        chat_disabled = st.session_state.loaded_transcription is None
        chat_placeholder = "Upload a transcription first..." if chat_disabled else "Type your session notes or instructions..."
        if prompt := st.chat_input(
            chat_placeholder,
            key="active_chat_input",
            disabled=chat_disabled
        ):
            session_path = os.path.join(ACTIVE_PATH, selected_client, session_folder)
            full_prompt = f"[Client: {selected_client}] [Session: {session_folder}] [Session Path: {session_path}]\n{prompt}"

            st.session_state.messages_active.append({
                "role": "user",
                "content": full_prompt
            })

            with st.spinner("Session agent is processing..."):
                response = invoke_agent(
                    session_agent,
                    st.session_state.messages_active
                )
                st.session_state.messages_active.append({
                    "role": "assistant",
                    "content": response
                })

                # Check for generated documents
                session_path = os.path.join(ACTIVE_PATH, selected_client, session_folder)
                for key, filename in [("summary", "summary.txt"),
                                       ("homework", "homework.txt"),
                                       ("next_prep", "next_session.txt")]:
                    doc_path = os.path.join(session_path, filename)
                    if os.path.exists(doc_path):
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            st.session_state.session_documents[key] = f.read()

            st.rerun()

        # Generated documents with preview and download (below chat input)
        if st.session_state.session_documents:
            st.subheader("Generated Documents")

            doc_mapping = {
                "summary": ("📄 Summary", "summary.txt"),
                "homework": ("📝 Homework", "homework.txt"),
                "next_session": ("🔮 Next Session", "next_session.txt")
            }

            for key, (label, filename) in doc_mapping.items():
                if key in st.session_state.session_documents:
                    with st.expander(f"{label} - Preview & Download"):
                        st.text(st.session_state.session_documents[key])
                        st.download_button(
                            label=f"Download {label}",
                            data=st.session_state.session_documents[key],
                            file_name=filename,
                            mime="text/plain",
                            key=f"download_{key}"
                        )


# ============================================================================
# TAB 4: CHAT ASSISTANT
# ============================================================================

with tab4:
    st.header("Chat Assistant")
    st.caption("Ask about any client's progress, logs, or history (read-only)")

    st.divider()

    # Chat messages display
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages_chat:
            with st.chat_message(message["role"]):
                display_text = strip_context_tags(message["content"]) if message["role"] == "user" else message["content"]
                st.markdown(display_text)

    # Simple chat input (no file upload)
    if prompt := st.chat_input(
        "Ask about clients, progress, or coaching history...",
        key="chat_input"
    ):
        st.session_state.messages_chat.append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("Chat agent is thinking..."):
            response = invoke_agent(
                chat_agent,
                st.session_state.messages_chat
            )
            st.session_state.messages_chat.append({
                "role": "assistant",
                "content": response
            })

        st.rerun()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("🌿 Life Coach AI")
    st.markdown("<h3 style='font-family: serif; font-style: italic; color: #F4F1EA;'>Your Path, Your Power</h3>", unsafe_allow_html=True)
    st.divider()

    st.subheader("Quick Stats")
    st.metric("Active Clients", len(get_active_clients()))

    undefined_count = len([d for d in os.listdir(UNDEFINED_PATH)
                           if os.path.isdir(os.path.join(UNDEFINED_PATH, d))]) if os.path.exists(UNDEFINED_PATH) else 0
    st.metric("Pending Discovery", undefined_count)

