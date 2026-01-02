"""
Here you will find the retriever tool, so our Assistant Agent is able to read the vector store

You will find here:
retriever_tool
"""

from pathlib import Path
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, VECTOR_STORE_PATH
from .logger import log_tool_call
from .vector_store_creator import vector_store_creation

# Lazy loading - only initialize when needed
_retriever = None
_creation_error = None  # Store error message if creation failed


def _get_retriever():
    """Lazy load the retriever on first use. Auto-creates vector store if missing."""
    global _retriever, _creation_error

    if _retriever is not None:
        return _retriever

    if _creation_error is not None:
        return None  # Already tried and failed

    vector_store_path = Path(VECTOR_STORE_PATH)

    # If vector store doesn't exist, try to create it
    if not vector_store_path.exists():
        print("Vector store not found. Attempting to create...")
        success, message = vector_store_creation()
        if not success:
            _creation_error = message
            print(f"Vector store creation failed: {message}")
            return None
        print(message)

    # Load the vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    _retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    return _retriever


@tool(
    "retriever_tool",
    parse_docstring=True,
    description="Searches the coaching techniques database for strategies, methods and techniques"
)
def retriever_tool(query: str) -> str:
    """
    Description:
        Searches the vector store for life coaching strategies, methods and techniques.

    Args:
        query (str): The search query to find relevant coaching techniques.

    Returns:
        Relevant coaching techniques and strategies from the database.
    """
    log_tool_call("retriever_tool", {"query": query})

    retriever = _get_retriever()
    if retriever is None:
        error_msg = _creation_error or "Vector store not available."
        log_tool_call("retriever_tool", {"query": query[:50]},
                      output="Vector store not available", status="error")
        return f"ERROR: {error_msg}"

    docs = retriever.invoke(query)
    result = "\n\n---\n\n".join([doc.page_content for doc in docs])

    log_tool_call("retriever_tool", {"query": query[:50]},
                  output=f"Found {len(docs)} docs", status="success")
    return result