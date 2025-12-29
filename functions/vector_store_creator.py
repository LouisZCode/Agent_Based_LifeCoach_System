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


coaching_training_path = Path("LifeCoach_Data/Coaching_Training")

docx_loader = DirectoryLoader(
    path=coaching_training_path,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)

pdf_loader = DirectoryLoader(
    path=coaching_training_path,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)

docx_docs = docx_loader.load()
pdf_docs = pdf_loader.load()

all_documents = docx_docs + pdf_docs


def vector_store_creation():
    docx_loader = DirectoryLoader(
        path=coaching_training_path,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
    )

    pdf_loader = DirectoryLoader(
        path=coaching_training_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    docx_docs = docx_loader.load()
    pdf_docs = pdf_loader.load()
    all_documents = docx_docs + pdf_docs

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap= 100,
    separators=["\n\n", "\n", " ", ""]
)

    chunks = text_splitter.split_documents(all_documents)

    # IDEA: create metadata of chunks by TOOL?

    embedding = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("Vector Store successfully created")

#vector_store_creation()

