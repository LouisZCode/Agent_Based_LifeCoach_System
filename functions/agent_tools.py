"""
Here you will find the tools given to the ACTIVE Clients Agent, designed to read and write in the Local Files,
and create text and or documents based on transcriptions form sessions.

You will find here: 
read_folder, read_template, verify_document_draft, save_summary, save_homework, save_next_session_draft
"""

from langchain.tools import tool
from pathlib import Path
from docx import Document
from .logger import log_tool_call


@tool(
    "read_folder",
    parse_docstring=True,
    description="reads the existing folder names with client names"
)
def read_folder(path : str) -> str:
    """
    Description:
        Read the folder as a list of the names on 1 level.

    Args:
        path (str): this is the path where the folders you are searching for are.

    returns:
        The exact and correct names of the folders/clients, or an error message if path doesn't exist.
    """
    # Auto-convert spaces to underscores (client names use underscores)
    path = path.replace(" ", "_")

    log_tool_call("read_folder", {"path": path})
    active_path = Path(path)

    if not active_path.exists():
        log_tool_call("read_folder", {"path": path}, output="Path not found", status="not_found")
        return f"ERROR: Path '{path}' does not exist. Check the path and try again. Note: client names use underscores (e.g., 'Luis_Zermeno' not 'Luis Zermeno')."

    try:
        folders = [folder.name for folder in active_path.iterdir() if folder.is_dir()]
        log_tool_call("read_folder", {"path": path}, output=str(folders), status="success")
        return folders
    except Exception as e:
        log_tool_call("read_folder", {"path": path}, output=str(e), status="error")
        return f"ERROR: Could not read folder '{path}': {str(e)}"

@tool(
    "read_template",
    parse_docstring=True,
    description="reads the desired Template with desired structure"
)
def read_template(path : str, template_name : str) -> str:
    """
    Description:
        Reads the desired template structure.

    Args:
        path (str): this is the path where the template documents are.
        template_name (str): the desired template to read.

    returns:
        The template content to understand the structure inside it

    raises:
        Error if the template is not readable or does not exist

    """
    log_tool_call("read_template", {"path": path, "template_name": template_name})
    final_path = f"{path}/{template_name}"

    file_path = Path(final_path)
    doc = Document(file_path)

    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    log_tool_call("read_template", {"template_name": template_name}, output=f"Loaded {len(text)} chars", status="success")

    return text

@tool(
    "read_document",
    parse_docstring=True,
    description="reads a .txt document from a specific clients session folder"
)
def read_document(path : str, document_name : str) -> str:
    """
    Description:
        Reads a .txt document from the clients session folder (summary.txt, homework.txt, next_session.txt)

    Args:
        path (str): this is the path where the document is.
        document_name (str): the desired document to read.

    returns:
        The document content, or an error message if the file does not exist.
    """
    # Auto-convert spaces to underscores (client names use underscores)
    path = path.replace(" ", "_")

    log_tool_call("read_document", {"path": path, "document_name": document_name})
    final_path = f"{path}/{document_name}"
    file_path = Path(final_path)

    if not file_path.exists():
        log_tool_call("read_document", {"document_name": document_name}, output="File not found", status="not_found")
        return f"ERROR: Document '{document_name}' does not exist at path '{path}'. This session may not have this document yet."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        log_tool_call("read_document", {"document_name": document_name}, output=f"Loaded {len(text)} chars", status="success")
        return text
    except Exception as e:
        log_tool_call("read_document", {"document_name": document_name}, output=str(e), status="error")
        return f"ERROR: Could not read document '{document_name}': {str(e)}"

@tool(
    "verify_document_draft",
    parse_docstring=True,
    description="MUST use before saving any document. Checks if draft meets character limits."
)
def verify_document_draft(content: str, document_type: str, attempt: int = 1) -> str:
    """
    Description:
        Verifies that a document draft meets the character limits before saving.
        You MUST call this tool before using any save tool.

    Args:
        content (str): The full document draft to verify
        document_type (str): Type of document - "summary", "homework", or "draft"
        attempt (int): Current attempt number (1, 2, 3, 4). After attempt 4, save regardless.

    Returns:
        Analysis showing character count per section, limits, and PASS/FAIL status.
        If any section fails, you must edit the draft and verify again before saving.
    """
    # Hard limit: after 4 attempts, force save
    MAX_ATTEMPTS = 4
    if attempt >= MAX_ATTEMPTS:
        log_tool_call("verify_document_draft", {"document_type": document_type, "attempt": attempt}, output="LIMIT REACHED - forcing save", status="limit")
        return f"=== ATTEMPT LIMIT REACHED ({attempt}/{MAX_ATTEMPTS}) ===\n\nYou have made {attempt} verification attempts. Save the document now to avoid excessive costs.\n\nCall the save tool immediately with your current draft."
    log_tool_call("verify_document_draft", {"document_type": document_type, "attempt": attempt, "content_length": len(content)})
    lines = content.split('\n')
    total_chars = len(content)

    if document_type == "summary":
        # (min, max) ranges - relaxed by ~10% for flexibility
        limits = {
            "total": (2000, 2600),           # sweet spot: 2300, relaxed from 2200-2500
            "title": (10, 40),               # sweet spot: 25
            "warm_opening": (50, 200),       # sweet spot: 125
            "main_takeaways": (600, 900),    # sweet spot: 750, 3 bullets only
            "tools": (120, 350),             # sweet spot: 235
            "achievements": (250, 550),      # sweet spot: 400
            "next_steps": (100, 250)         # sweet spot: 175
        }

        # Parse sections
        title = lines[0] if lines else ""
        warm_opening = lines[2] if len(lines) > 2 else ""

        sections = {}
        current_section = None
        current_content = []
        section_markers = {
            "Main Takeaways": "main_takeaways",
            "Core \"Why\"": "core_why",
            "Tools": "tools",
            "Most Recent Achievements!": "achievements",
            "Most Recent Achievements": "achievements",
            "Next Steps": "next_steps"
        }

        for line in lines[3:]:
            if line.strip() in section_markers:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = section_markers[line.strip()]
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        # Build result
        results = []
        all_pass = True

        def check_range(value, min_val, max_val):
            if value < min_val:
                return f"✗ TOO SHORT (need {min_val - value} more)"
            elif value > max_val:
                return f"✗ TOO LONG (remove {value - max_val})"
            else:
                return "✓ PASS"

        # Total check
        total_min, total_max = limits['total']
        total_status = check_range(total_chars, total_min, total_max)
        results.append(f"TOTAL: {total_chars} chars (target: {total_min}-{total_max}) {total_status}")
        if total_chars < total_min or total_chars > total_max:
            all_pass = False

        # Title check
        title_len = len(title)
        title_min, title_max = limits['title']
        title_status = check_range(title_len, title_min, title_max)
        results.append(f"Title: {title_len} chars (target: {title_min}-{title_max}) {title_status}")
        if title_len < title_min or title_len > title_max:
            all_pass = False

        # Warm opening check
        warm_len = len(warm_opening)
        warm_min, warm_max = limits['warm_opening']
        warm_status = check_range(warm_len, warm_min, warm_max)
        results.append(f"Warm Opening: {warm_len} chars (target: {warm_min}-{warm_max}) {warm_status}")
        if warm_len < warm_min or warm_len > warm_max:
            all_pass = False

        # Section checks
        for section_key in ["main_takeaways", "tools", "achievements", "next_steps"]:
            section_content = sections.get(section_key, "")
            section_len = len(section_content)
            min_val, max_val = limits[section_key]
            status = check_range(section_len, min_val, max_val)
            passed = min_val <= section_len <= max_val
            if not passed:
                all_pass = False
            section_name = section_key.replace("_", " ").title()
            results.append(f"{section_name}: {section_len} chars (target: {min_val}-{max_val}) {status}")

        results.append("")
        if all_pass:
            results.append("=== ALL SECTIONS PASS === You may now save the document.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output="ALL PASS", status="success")
            return '\n'.join(results)
        else:
            # Build specific edit instructions
            edit_instructions = []

            # Helper: add 10% overshoot to REMOVE instructions so LLM cuts more aggressively
            def overshoot(chars_to_remove):
                return int(chars_to_remove * 1.1)

            total_min, total_max = limits['total']
            if total_chars < total_min:
                edit_instructions.append(f"TOTAL: ADD {total_min - total_chars} chars overall")
            elif total_chars > total_max:
                edit_instructions.append(f"TOTAL: REMOVE {overshoot(total_chars - total_max)} chars overall")

            if warm_len < limits['warm_opening'][0]:
                edit_instructions.append(f"Warm Opening: ADD {limits['warm_opening'][0] - warm_len} chars")
            elif warm_len > limits['warm_opening'][1]:
                edit_instructions.append(f"Warm Opening: REMOVE {overshoot(warm_len - limits['warm_opening'][1])} chars")

            for section_key in ["main_takeaways", "tools", "achievements", "next_steps"]:
                section_content = sections.get(section_key, "")
                section_len = len(section_content)
                min_val, max_val = limits[section_key]
                section_name = section_key.replace("_", " ").title()
                if section_len < min_val:
                    edit_instructions.append(f"{section_name}: ADD {min_val - section_len} chars")
                elif section_len > max_val:
                    edit_instructions.append(f"{section_name}: REMOVE {overshoot(section_len - max_val)} chars")

            # Return draft + edit instructions so LLM can edit in place
            results.append("=== EDIT REQUIRED ===")
            results.append("")
            results.append("SPECIFIC EDITS NEEDED:")
            for instruction in edit_instructions:
                results.append(f"  • {instruction}")
            results.append("")
            results.append("Edit the draft below. Do NOT regenerate from transcription.")
            results.append("")
            results.append("---START DRAFT---")
            results.append(content)
            results.append("---END DRAFT---")
            results.append("")
            results.append("Make the edits above, then call verify_document_draft again.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output=f"FAILED - edits needed: {edit_instructions}", status="fail")
            return '\n'.join(results)

    elif document_type == "homework":
        # (min, max) ranges for homework
        limits = {
            "total": (1200, 1800),
            "title": (20, 60),
            "goal": (80, 200),
            "instructions": (150, 350),
            "before_you_begin": (200, 350),
            "prompt_questions": (300, 700),
        }

        # Parse sections
        title = lines[0] if lines else ""

        sections = {}
        current_section = None
        current_content = []
        section_markers = {
            "Goal:": "goal",
            "Instructions": "instructions",
            "Before you begin": "before_you_begin",
            "Prompt Questions": "prompt_questions",
        }

        for line in lines[1:]:
            line_stripped = line.strip()
            matched = False
            for marker, key in section_markers.items():
                if line_stripped.startswith(marker):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = key
                    current_content = []
                    # If content on same line as marker (e.g., "Goal: Some text")
                    remaining = line_stripped[len(marker):].strip()
                    if remaining:
                        current_content.append(remaining)
                    matched = True
                    break
            if not matched and current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        # Build result
        results = []
        all_pass = True

        def check_range(value, min_val, max_val):
            if value < min_val:
                return f"✗ TOO SHORT (need {min_val - value} more)"
            elif value > max_val:
                return f"✗ TOO LONG (remove {value - max_val})"
            else:
                return "✓ PASS"

        def overshoot(chars_to_remove):
            return int(chars_to_remove * 1.1)

        # Total check
        total_min, total_max = limits['total']
        total_status = check_range(total_chars, total_min, total_max)
        results.append(f"TOTAL: {total_chars} chars (target: {total_min}-{total_max}) {total_status}")
        if total_chars < total_min or total_chars > total_max:
            all_pass = False

        # Title check
        title_len = len(title)
        title_min, title_max = limits['title']
        title_status = check_range(title_len, title_min, title_max)
        results.append(f"Title: {title_len} chars (target: {title_min}-{title_max}) {title_status}")
        if title_len < title_min or title_len > title_max:
            all_pass = False

        # Section checks
        for section_key in ["goal", "instructions", "before_you_begin", "prompt_questions"]:
            section_content = sections.get(section_key, "")
            section_len = len(section_content)
            min_val, max_val = limits[section_key]
            status = check_range(section_len, min_val, max_val)
            passed = min_val <= section_len <= max_val
            if not passed:
                all_pass = False
            section_name = section_key.replace("_", " ").title()
            results.append(f"{section_name}: {section_len} chars (target: {min_val}-{max_val}) {status}")

        results.append("")
        if all_pass:
            results.append("=== ALL SECTIONS PASS === You may now save the document.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output="ALL PASS", status="success")
            return '\n'.join(results)
        else:
            # Build specific edit instructions
            edit_instructions = []

            total_min, total_max = limits['total']
            if total_chars < total_min:
                edit_instructions.append(f"TOTAL: ADD {total_min - total_chars} chars overall")
            elif total_chars > total_max:
                edit_instructions.append(f"TOTAL: REMOVE {overshoot(total_chars - total_max)} chars overall")

            if title_len < limits['title'][0]:
                edit_instructions.append(f"Title: ADD {limits['title'][0] - title_len} chars")
            elif title_len > limits['title'][1]:
                edit_instructions.append(f"Title: REMOVE {overshoot(title_len - limits['title'][1])} chars")

            for section_key in ["goal", "instructions", "before_you_begin", "prompt_questions"]:
                section_content = sections.get(section_key, "")
                section_len = len(section_content)
                min_val, max_val = limits[section_key]
                section_name = section_key.replace("_", " ").title()
                if section_len < min_val:
                    edit_instructions.append(f"{section_name}: ADD {min_val - section_len} chars")
                elif section_len > max_val:
                    edit_instructions.append(f"{section_name}: REMOVE {overshoot(section_len - max_val)} chars")

            # Return draft + edit instructions
            results.append("=== EDIT REQUIRED ===")
            results.append("")
            results.append("SPECIFIC EDITS NEEDED:")
            for instruction in edit_instructions:
                results.append(f"  • {instruction}")
            results.append("")
            results.append("Edit the draft below. Do NOT regenerate from transcription.")
            results.append("")
            results.append("---START DRAFT---")
            results.append(content)
            results.append("---END DRAFT---")
            results.append("")
            results.append("Make the edits above, then call verify_document_draft again.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output=f"FAILED - edits needed: {edit_instructions}", status="fail")
            return '\n'.join(results)

    elif document_type == "draft":
        # New template structure: static content (~2008 chars) + dynamic sections
        # Static content should not be modified by LLM
        # Only dynamic sections (marked with specific content after instructions) are variable

        EXPECTED_STATIC = 2008  # chars of coaching instructions
        STATIC_TOLERANCE = 150  # allow small variations

        # Dynamic section limits (min, max) - these are the LLM-filled parts
        dynamic_limits = {
            "previous_homework": (100, 200),
            "suggested_concept": (150, 300),
            "suggested_exercises": (150, 300),
            "emerging_topics": (100, 200),
            "next_steps": (150, 300),
            "suggested_homework": (100, 200),
        }

        # Calculate expected total range
        dynamic_min = sum(v[0] for v in dynamic_limits.values())  # ~750
        dynamic_max = sum(v[1] for v in dynamic_limits.values())  # ~1500
        total_min = EXPECTED_STATIC + dynamic_min - STATIC_TOLERANCE  # ~2608
        total_max = EXPECTED_STATIC + dynamic_max + STATIC_TOLERANCE  # ~3658

        # Section markers to find dynamic content
        # Dynamic content appears AFTER these markers (after the static instructions)
        section_markers = [
            ("3. Homework Assessment", "previous_homework", "4. Main Focus"),
            ("4.1 Clarifying a Concept", "suggested_concept", "4.2 Practical"),
            ("4.2 Practical Exercise", "suggested_exercises", "5. Emerging"),
            ("5. Emerging Topics", "emerging_topics", "6. Wrap-Up"),
            ("7. Next Steps", "next_steps", "8. Homework"),
            ("8. Homework", "suggested_homework", "9. Final"),
        ]

        # Parse dynamic sections
        dynamic_sections = {}
        for start_marker, key, end_marker in section_markers:
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker) if end_marker else len(content)
            if start_idx != -1 and end_idx != -1:
                section_content = content[start_idx:end_idx]
                # Find the last paragraph (the dynamic part) after static instructions
                paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
                if paragraphs:
                    # The dynamic content is typically the last non-empty paragraph before next section
                    dynamic_sections[key] = paragraphs[-1] if len(paragraphs) > 1 else ""

        # Build result
        results = []
        all_pass = True

        def check_range(value, min_val, max_val):
            if value < min_val:
                return f"✗ TOO SHORT (need {min_val - value} more)"
            elif value > max_val:
                return f"✗ TOO LONG (remove {value - max_val})"
            else:
                return "✓ PASS"

        def overshoot(chars_to_remove):
            return int(chars_to_remove * 1.1)

        # Total check
        total_status = check_range(total_chars, total_min, total_max)
        results.append(f"TOTAL: {total_chars} chars (target: {total_min}-{total_max}) {total_status}")
        if total_chars < total_min or total_chars > total_max:
            all_pass = False

        # Dynamic sections check
        section_names = {
            "previous_homework": "Previous Homework",
            "suggested_concept": "Suggested Concept",
            "suggested_exercises": "Suggested Exercises",
            "emerging_topics": "Emerging Topics",
            "next_steps": "Next Steps",
            "suggested_homework": "Suggested Homework",
        }

        total_dynamic = 0
        for key in dynamic_limits.keys():
            section_content = dynamic_sections.get(key, "")
            section_len = len(section_content)
            total_dynamic += section_len
            min_val, max_val = dynamic_limits[key]
            status = check_range(section_len, min_val, max_val)
            passed = min_val <= section_len <= max_val
            if not passed:
                all_pass = False
            results.append(f"{section_names[key]}: {section_len} chars (target: {min_val}-{max_val}) {status}")

        # Static content check (total - dynamic = static)
        static_chars = total_chars - total_dynamic
        static_status = check_range(static_chars, EXPECTED_STATIC - STATIC_TOLERANCE, EXPECTED_STATIC + STATIC_TOLERANCE)
        results.append(f"Static Content: {static_chars} chars (expected: ~{EXPECTED_STATIC}) {static_status}")
        if static_chars < EXPECTED_STATIC - STATIC_TOLERANCE or static_chars > EXPECTED_STATIC + STATIC_TOLERANCE:
            all_pass = False
            results.append("  ⚠ WARNING: Static coaching instructions may have been modified!")

        results.append("")
        if all_pass:
            results.append("=== ALL SECTIONS PASS === You may now save the document.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output="ALL PASS", status="success")
            return '\n'.join(results)
        else:
            # Build specific edit instructions
            edit_instructions = []

            if total_chars < total_min:
                edit_instructions.append(f"TOTAL: ADD {total_min - total_chars} chars overall")
            elif total_chars > total_max:
                edit_instructions.append(f"TOTAL: REMOVE {overshoot(total_chars - total_max)} chars overall")

            for key in dynamic_limits.keys():
                section_content = dynamic_sections.get(key, "")
                section_len = len(section_content)
                min_val, max_val = dynamic_limits[key]
                if section_len < min_val:
                    edit_instructions.append(f"{section_names[key]}: ADD {min_val - section_len} chars")
                elif section_len > max_val:
                    edit_instructions.append(f"{section_names[key]}: REMOVE {overshoot(section_len - max_val)} chars")

            # Return draft + edit instructions
            results.append("=== EDIT REQUIRED ===")
            results.append("")
            results.append("SPECIFIC EDITS NEEDED:")
            for instruction in edit_instructions:
                results.append(f"  • {instruction}")
            results.append("")
            results.append("Edit the draft below. Do NOT regenerate from transcription.")
            results.append("")
            results.append("---START DRAFT---")
            results.append(content)
            results.append("---END DRAFT---")
            results.append("")
            results.append("Make the edits above, then call verify_document_draft again.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output=f"FAILED - edits needed: {edit_instructions}", status="fail")
            return '\n'.join(results)

    else:
        log_tool_call("verify_document_draft", {"document_type": document_type}, output="Unknown type", status="error")
        return f"Unknown document type: {document_type}. Use 'summary', 'homework', or 'draft'."


@tool(
    "save_summary",
    parse_docstring=True,
    description="saves the summary of the transcription and user notes in the correct Session Path"
)
def save_summary(path: str, content : str) -> str:
    """
    Description:
        saves the created summary in the current Session Path.

    Args:
        path (str): The current Session Path given in the context
        content (str): The summary created following the summary_template guidelines

    Returns:
        Confirmation that the summary has been saved in the Session Path

    Raises:
        Error is the path is incorrect of was an issue while saving.
    """
    log_tool_call("save_summary", {"path": path, "content_length": len(content)})
    file_path = Path(f"{path}/summary.txt")
    file_path.write_text(content, encoding="utf-8")
    log_tool_call("save_summary", {"path": path}, output=str(file_path), status="saved")

    return f"new summary created in {file_path}"

@tool(
    "save_homework",
    parse_docstring=True,
    description="saves the homework aligned with the transcription and user notes in the correct Session Path"
)
def save_homework(path: str, content : str) -> str:
    """
    Description:
        saves the created homework in the current Session Path.

    Args:
        path (str): The current Session Path given in the context
        content (str): The homework created following the homework_template guidelines

    Returns:
        Confirmation that the homework has been saved in the Session Path

    Raises:
        Error is the path is incorrect of was an issue while saving.
    """
    log_tool_call("save_homework", {"path": path, "content_length": len(content)})
    file_path = Path(f"{path}/homework.txt")
    file_path.write_text(content, encoding="utf-8")
    log_tool_call("save_homework", {"path": path}, output=str(file_path), status="saved")

    return f"new homework created in {file_path}"

@tool(
    "save_next_session_draft",
    parse_docstring=True,
    description="saves the next session draft aligned with the transcription and user notes in the correct Session Path"
)
def save_session_draft(path: str, content : str) -> str:
    """
    Description:
        saves the created next session draft in the current Session Path.

    Args:
        path (str): The current Session Path given in the context
        content (str): The session draft created following the next_session_template guidelines

    Returns:
        Confirmation that the session draft has been saved in the Session Path

    Raises:
        Error is the path is incorrect of was an issue while saving.
    """
    log_tool_call("save_next_session_draft", {"path": path, "content_length": len(content)})
    file_path = Path(f"{path}/next_session.txt")
    file_path.write_text(content, encoding="utf-8")
    log_tool_call("save_next_session_draft", {"path": path}, output=str(file_path), status="saved")

    return f"new next session draft created in {file_path}"

@tool(
    "save_initial_persona",
    parse_docstring=True,
    description="creates a new client folder and saves the initial client persona document in the Evolution folder"
)
def save_initial_persona(client_name: str, content: str) -> str:
    """
    Description:
        Creates a new client folder in Undefined, creates the Evolution subfolder,
        and saves the initial client persona document with today's date.

    Args:
        client_name (str): The sanitized client name (e.g., "Pedro_Perez")
        content (str): The formatted persona content to save

    Returns:
        Confirmation message with the file path created

    Raises:
        Error if there was an issue while saving.
    """
    from datetime import datetime

    log_tool_call("save_initial_persona", {"client_name": client_name, "content_length": len(content)})

    # Create client folder and Evolution subfolder
    client_path = Path(f"LifeCoach_Data/Undefined/{client_name}")
    evolution_path = client_path / "Evolution"
    evolution_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with today's date
    date_str = datetime.now().strftime("%d-%m-%Y")
    filename = f"{date_str}_initial_client_persona.txt"
    file_path = evolution_path / filename

    file_path.write_text(content, encoding="utf-8")
    log_tool_call("save_initial_persona", {"client_name": client_name}, output=str(file_path), status="saved")

    return f"Initial client persona saved at {file_path}"