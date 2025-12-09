"""
Here you will find the tools given to the Agent, specially to read and write
in the Local Files.
"""


from langchain.tools import tool
from pathlib import Path
from docx import Document


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
        The exact and correct names of the clients

    raises:
        Error if the path is empty

    """
    active_path = Path(path)
    clients = [folder.name for folder in active_path.iterdir() if folder.is_dir()]

    return clients

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
    final_path = f"{path}/{template_name}"

    file_path = Path(final_path)
    doc = Document(file_path)

    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    return text

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
    file_path = Path(f"{path}/summary.txt")
    file_path.write_text(content)

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
    file_path = Path(f"{path}/homework.txt")
    file_path.write_text(content)

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
    file_path = Path(f"{path}/next_session.txt")
    file_path.write_text(content)

    return f"new next sesison draft created in {file_path}"