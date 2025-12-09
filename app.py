"""
Life Coach AI Assistant - Streamlit Application
A multi-tab interface for managing coaching clients and sessions.
"""

#run by writing:
#   streamlit run /Users/luiszg/Desktop/GitHub/LifeCoach_AI_System/app.py


import streamlit as st
import os
from datetime import datetime
from io import BytesIO

# File parsing imports
from docx import Document as DocxDocument
import PyPDF2
# Import agents (to be created by user)
from agents import session_agent

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Life Coach AI Assistant",
    page_icon="🧘",
    layout="wide"
)

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


def get_active_clients() -> list:
    """Get list of active client folders"""
    if not os.path.exists(ACTIVE_PATH):
        return []
    return [d for d in os.listdir(ACTIVE_PATH)
            if os.path.isdir(os.path.join(ACTIVE_PATH, d))]


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
    from langchain_core.messages import AIMessage

    response = agent.invoke({
        "messages": messages
    })

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

# Tab 1: Undefined Clients
if "messages_undefined" not in st.session_state:
    st.session_state.messages_undefined = []
if "current_undefined_client" not in st.session_state:
    st.session_state.current_undefined_client = ""
if "discovery_prep_content" not in st.session_state:
    st.session_state.discovery_prep_content = None

# Tab 2: Active Clients
if "messages_active" not in st.session_state:
    st.session_state.messages_active = []
if "session_documents" not in st.session_state:
    st.session_state.session_documents = {}

# Tab 3: Chat Assistant
if "messages_chat" not in st.session_state:
    st.session_state.messages_chat = []


# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("🧘 Life Coach AI Assistant")

tab1, tab2, tab3 = st.tabs([
    "📋 Undefined Clients",
    "✅ Active Clients",
    "💬 Chat Assistant"
])


# ============================================================================
# TAB 1: UNDEFINED CLIENTS
# ============================================================================

with tab1:
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
# TAB 2: ACTIVE CLIENTS
# ============================================================================

with tab2:
    st.header("Active Client Sessions")

    col1, col2 = st.columns([2, 1])

    with col1:
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

    with col2:
        session_date = st.date_input(
            "Session Date",
            value=datetime.today(),
            key="session_date"
        )

    if selected_client:
        session_num = get_next_session_number(selected_client)
        st.caption(f"Next session: Session {session_num}")

        st.divider()

        # Chat messages display
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages_active:
                with st.chat_message(message["role"]):
                    display_text = strip_context_tags(message["content"]) if message["role"] == "user" else message["content"]
                    st.markdown(display_text)

        # Download buttons for generated documents
        if st.session_state.session_documents:
            st.subheader("Generated Documents")
            doc_cols = st.columns(3)

            doc_mapping = {
                "summary": ("📄 Summary", "summary.txt"),
                "homework": ("📝 Homework", "homework.txt"),
                "next_prep": ("🔮 Next Session Prep", "next_session_prep.txt")
            }

            for idx, (key, (label, filename)) in enumerate(doc_mapping.items()):
                if key in st.session_state.session_documents:
                    with doc_cols[idx]:
                        st.download_button(
                            label=label,
                            data=st.session_state.session_documents[key],
                            file_name=filename,
                            mime="text/plain",
                            key=f"download_{key}"
                        )

        # File upload for transcription
        uploaded_file = st.file_uploader(
            "Upload session transcription",
            type=['txt', 'docx', 'pdf'],
            key="active_file_upload"
        )

        if uploaded_file:
            file_content = read_uploaded_file(uploaded_file)
            if file_content:
                st.success(f"Transcription '{uploaded_file.name}' uploaded successfully!")
                file_message = f"[Session transcription: {uploaded_file.name}]\n\n{file_content}"
                if not any(file_message in m.get("content", "") for m in st.session_state.messages_active):
                    st.session_state.messages_active.append({
                        "role": "user",
                        "content": file_message
                    })

        # Chat input
        if prompt := st.chat_input(
            "Type your session notes or instructions...",
            key="active_chat_input"
        ):
            session_folder = f"Session_{session_num}_{session_date.strftime('%Y%m%d')}"
            full_prompt = f"[Client: {selected_client}] [Session: {session_folder}]\n{prompt}"

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
                                       ("next_prep", "next_session_prep.txt")]:
                    doc_path = os.path.join(session_path, filename)
                    if os.path.exists(doc_path):
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            st.session_state.session_documents[key] = f.read()

            st.rerun()


# ============================================================================
# TAB 3: CHAT ASSISTANT
# ============================================================================

with tab3:
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
    st.header("🧘 Life Coach AI")
    st.divider()

    st.subheader("Quick Stats")
    st.metric("Active Clients", len(get_active_clients()))

    undefined_count = len([d for d in os.listdir(UNDEFINED_PATH)
                           if os.path.isdir(os.path.join(UNDEFINED_PATH, d))]) if os.path.exists(UNDEFINED_PATH) else 0
    st.metric("Pending Discovery", undefined_count)

    st.divider()

    st.subheader("Clear Conversations")
    if st.button("Clear Undefined Chat", key="clear_undefined"):
        st.session_state.messages_undefined = []
        st.session_state.discovery_prep_content = None
        st.rerun()

    if st.button("Clear Active Chat", key="clear_active"):
        st.session_state.messages_active = []
        st.session_state.session_documents = {}
        st.rerun()

    if st.button("Clear Assistant Chat", key="clear_chat"):
        st.session_state.messages_chat = []
        st.rerun()
