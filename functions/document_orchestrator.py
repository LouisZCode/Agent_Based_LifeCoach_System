"""
Document Orchestrator - Python-side verification loop for document creation.

This module moves the verify-edit loop from inside LangGraph to Python control,
reducing token usage by ~80% (only first call includes transcription, edits use draft only).
"""

from .agent_tools import verify_document_draft, save_summary, save_homework, save_session_draft
from .logger import log_orchestrator, log_separator

DRAFT_START = "---DRAFT_START---"
DRAFT_END = "---DRAFT_END---"
MAX_ATTEMPTS = 4

TEMPLATE_MAP = {
    "summary": "summary_template_llmready.docx",
    "homework": "homework_template_llmready.docx",
    "draft": "next_session_template_llmready.docx"
}


def extract_draft(response: str) -> str | None:
    """
    Extract draft content from agent response using markers.

    Args:
        response: Agent response text

    Returns:
        Draft content or None if markers not found
    """
    if DRAFT_START in response and DRAFT_END in response:
        start = response.index(DRAFT_START) + len(DRAFT_START)
        end = response.index(DRAFT_END)
        return response[start:end].strip()
    return None


def detect_document_type(prompt: str) -> str | None:
    """
    Detect which document type is requested from user prompt.

    Args:
        prompt: User's message

    Returns:
        "summary", "homework", "draft", or None
    """
    prompt_lower = prompt.lower()

    if "summary" in prompt_lower:
        return "summary"
    elif "homework" in prompt_lower:
        return "homework"
    elif "draft" in prompt_lower or "next session" in prompt_lower:
        return "draft"

    return None


def create_document(invoke_fn, transcription: str, doc_type: str, session_path: str, client_name: str = "", session_folder: str = "") -> tuple[str | None, str]:
    """
    Orchestrate document creation with Python-side verification loop.

    Flow:
    1. First invoke with transcription → get draft (~12k tokens)
    2. Python verifies draft (FREE - no LLM)
    3. If fails: new invoke with draft only (~2.5k tokens)
    4. Repeat until pass or max attempts
    5. Save via Python (FREE)

    Args:
        invoke_fn: Function to call agent (takes message list, returns response)
        transcription: Session transcription text
        doc_type: "summary", "homework", or "draft"
        session_path: Path to save document
        client_name: Optional client name for context
        session_folder: Optional session folder for context

    Returns:
        (draft, status_message) tuple
    """
    if doc_type not in TEMPLATE_MAP:
        return None, f"Unknown document type: {doc_type}"

    template_name = TEMPLATE_MAP[doc_type]

    # Log orchestrator activation
    log_separator(f"Python Orchestrator: {doc_type}")
    log_orchestrator("STARTED", {
        "doc_type": doc_type,
        "transcription_chars": len(transcription),
        "transcription_tokens_est": len(transcription) // 4
    })

    # Step 1: Initial draft request (WITH transcription - only time we send it)
    initial_prompt = f"""[DRAFT_MODE] Create a {doc_type} for this session.

Read the template at LifeCoach_Data/Templates/{template_name}
Follow the template structure and character limits exactly.

Return your draft wrapped in these markers:
{DRAFT_START}
(your draft here)
{DRAFT_END}

Do NOT call verify_document_draft or save tools - just return the draft.

[Client: {client_name}] [Session: {session_folder}] [Session Path: {session_path}]

[Session Transcription]
{transcription}"""

    log_orchestrator("INVOKE_1 (with transcription)", {
        "prompt_chars": len(initial_prompt),
        "prompt_tokens_est": len(initial_prompt) // 4
    })

    response = invoke_fn([{"role": "user", "content": initial_prompt}])
    draft = extract_draft(response)

    if not draft:
        # Fallback: try to use the whole response as draft (agent didn't use markers)
        draft = response.strip()
        if len(draft) < 500:  # Too short, probably an error message
            log_orchestrator("FAILED", {"reason": "Could not extract draft"})
            return None, f"Failed to extract draft from response: {response[:200]}..."
        log_orchestrator("WARNING", {"reason": "Draft markers not found, using full response"})

    log_orchestrator("DRAFT_RECEIVED", {
        "draft_chars": len(draft),
        "draft_tokens_est": len(draft) // 4
    })

    # Step 2: Python-side verification loop
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # FREE verification call (no LLM - just Python)
        log_orchestrator(f"VERIFY attempt={attempt}", {"draft_chars": len(draft)})

        result = verify_document_draft.invoke({
            "content": draft,
            "document_type": doc_type,
            "attempt": attempt
        })

        if "ALL SECTIONS PASS" in result:
            log_orchestrator("VERIFY_PASSED", {"attempt": attempt})
            break

        if attempt >= MAX_ATTEMPTS:
            log_orchestrator("VERIFY_LIMIT_REACHED", {"attempt": attempt})
            break

        # Step 3: Edit request (NO transcription - just draft + instructions ~2.5k tokens)
        edit_prompt = f"""[EDIT_MODE] Edit this draft according to the instructions below.

Do NOT regenerate from scratch or reference the original transcription.
Just edit what's provided to fix the character count issues.

Return the edited draft wrapped in these markers:
{DRAFT_START}
(your edited draft here)
{DRAFT_END}

{result}"""

        log_orchestrator(f"INVOKE_EDIT attempt={attempt}", {
            "prompt_chars": len(edit_prompt),
            "prompt_tokens_est": len(edit_prompt) // 4,
            "note": "NO transcription"
        })

        response = invoke_fn([{"role": "user", "content": edit_prompt}])
        new_draft = extract_draft(response)

        if new_draft:
            draft = new_draft
            log_orchestrator("EDIT_RECEIVED", {"new_draft_chars": len(draft)})
        else:
            # Fallback: use whole response
            if len(response.strip()) > 500:
                draft = response.strip()
                log_orchestrator("EDIT_RECEIVED (fallback)", {"new_draft_chars": len(draft)})

    # Step 4: Save directly via Python (no LLM call)
    try:
        log_orchestrator("SAVE (Python, no LLM)", {"doc_type": doc_type, "path": session_path})

        if doc_type == "summary":
            save_result = save_summary.invoke({"path": session_path, "content": draft})
        elif doc_type == "homework":
            save_result = save_homework.invoke({"path": session_path, "content": draft})
        else:
            save_result = save_session_draft.invoke({"path": session_path, "content": draft})

        log_orchestrator("COMPLETED", {"result": save_result[:100] if len(save_result) > 100 else save_result})
        return draft, save_result
    except Exception as e:
        log_orchestrator("SAVE_FAILED", {"error": str(e)})
        return draft, f"Draft created but save failed: {str(e)}"
