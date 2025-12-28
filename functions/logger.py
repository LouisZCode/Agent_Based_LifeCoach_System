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

    Args:
        action: What the orchestrator is doing
        details: Optional dict with details (chars, tokens estimate, etc.)
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
