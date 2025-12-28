"""
Simple logging utility for tracking agent tool calls and workflow.
Logs are saved to the logs/ folder with daily rotation.
"""

import os
from datetime import datetime
from pathlib import Path

LOGS_PATH = Path(__file__).parent.parent / "logs"

def get_log_file():
    """Get today's log file path."""
    today = datetime.now().strftime("%Y-%m-%d")
    return LOGS_PATH / f"agent_log_{today}.txt"

def log_tool_call(tool_name: str, inputs: dict, output: str = None, status: str = "called"):
    """
    Log a tool call with timestamp.

    Args:
        tool_name: Name of the tool being called
        inputs: Dictionary of input parameters (will be truncated if too long)
        output: Output from the tool (optional, will be truncated)
        status: Status of the call (called, success, error)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Truncate long inputs/outputs for readability
    inputs_str = str(inputs)
    if len(inputs_str) > 200:
        inputs_str = inputs_str[:200] + "..."

    output_str = ""
    if output:
        output_str = str(output)
        if len(output_str) > 300:
            output_str = output_str[:300] + "..."
        output_str = f" → {output_str}"

    log_entry = f"[{timestamp}] [{status.upper()}] {tool_name} | inputs: {inputs_str}{output_str}\n"

    # Ensure logs directory exists
    LOGS_PATH.mkdir(exist_ok=True)

    # Append to log file
    with open(get_log_file(), "a", encoding="utf-8") as f:
        f.write(log_entry)

def log_workflow_step(step: str, details: str = ""):
    """Log a workflow step (e.g., 'STEP 1: Reading template')."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [WORKFLOW] {step}"
    if details:
        log_entry += f" | {details}"
    log_entry += "\n"

    LOGS_PATH.mkdir(exist_ok=True)
    with open(get_log_file(), "a", encoding="utf-8") as f:
        f.write(log_entry)

def log_separator(label: str = ""):
    """Add a visual separator in the log for new sessions."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = f"\n{'='*60}\n[{timestamp}] === {label} ===\n{'='*60}\n"

    LOGS_PATH.mkdir(exist_ok=True)
    with open(get_log_file(), "a", encoding="utf-8") as f:
        f.write(separator)


def log_orchestrator(action: str, details: dict = None):
    """
    Log Python orchestrator actions (not LLM tool calls).
    DEPRECATED: Use log_doc_creation() for cleaner output.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    details_str = ""
    if details:
        parts = []
        for k, v in details.items():
            parts.append(f"{k}={v}")
        details_str = " | " + ", ".join(parts)

    log_entry = f"[{timestamp}] [ORCHESTRATOR] {action}{details_str}\n"

    LOGS_PATH.mkdir(exist_ok=True)
    with open(get_log_file(), "a", encoding="utf-8") as f:
        f.write(log_entry)


def log_doc_creation(event: str, doc_type: str = "", client: str = "", session: str = "", **kwargs):
    """
    Unified logging for document creation workflow.

    Events:
        HEADER - Start of document creation (creates separator)
        START - Initial info (transcription size)
        DRAFT - Sending draft request to LLM
        DRAFT_OK - Draft received successfully
        VERIFY - Verification result (PASS/FAIL/LIMIT)
        EDIT - Sending edit request to LLM
        TOOL - Tool call (indented)
        SAVED - Document saved
        ERROR - Error occurred
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    if event == "HEADER":
        header = f"{doc_type.upper()}: {client} / {session}"
        entry = f"\n{'='*60}\n[{timestamp}] === {header} ===\n{'='*60}\n"
    elif event == "TOOL":
        tool_name = kwargs.get("tool", "unknown")
        result = kwargs.get("result", "")
        entry = f"[{timestamp}]   └─ {tool_name} → {result}\n"
    elif event == "VERIFY":
        attempt = kwargs.get("attempt", 0)
        status = kwargs.get("status", "")
        details = kwargs.get("details", "")
        entry = f"[{timestamp}] VERIFY #{attempt} {status} | {details}\n"
    else:
        details_parts = []
        for k, v in kwargs.items():
            details_parts.append(f"{k}: {v}")
        details_str = " | ".join(details_parts) if details_parts else ""
        entry = f"[{timestamp}] {event}{' | ' + details_str if details_str else ''}\n"

    LOGS_PATH.mkdir(exist_ok=True)
    with open(get_log_file(), "a", encoding="utf-8") as f:
        f.write(entry)
