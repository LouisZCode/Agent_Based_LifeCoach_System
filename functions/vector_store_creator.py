"""
Here the evolving function to create a vector store.

Triggered IF there is no vector store folder in the database

You find here:
vector_store_creation
"""

# Run independently using: uv run python -m functions.vector_store_creator

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, VECTOR_STORE_PATH
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

COACHING_TRAINING_PATH = Path("LifeCoach_Data/Coaching_Training")


def vector_store_creation() -> tuple[bool, str]:
    """
    Create vector store from coaching training documents.

    Returns:
        tuple[bool, str]: (success, message)
    """
    # Check if source folder exists
    if not COACHING_TRAINING_PATH.exists():
        return False, f"Source folder '{COACHING_TRAINING_PATH}' does not exist. Please create it and add coaching documents (.pdf, .docx)."

    # Load documents
    docx_loader = DirectoryLoader(
        path=COACHING_TRAINING_PATH,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
    )
    pdf_loader = DirectoryLoader(
        path=COACHING_TRAINING_PATH,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    docx_docs = docx_loader.load()
    pdf_docs = pdf_loader.load()
    all_documents = docx_docs + pdf_docs

    # Check if any documents found
    if not all_documents:
        return False, f"No documents found in '{COACHING_TRAINING_PATH}'. Please add coaching documents (.pdf, .docx)."

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # Create and save vector store
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(VECTOR_STORE_PATH)

    return True, f"Vector store created from {len(all_documents)} documents ({len(chunks)} chunks)."


# Run directly for manual creation
if __name__ == "__main__":
    success, message = vector_store_creation()
    print(message)

