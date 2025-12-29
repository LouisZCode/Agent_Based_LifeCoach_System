"""
Here the evolving function to create a vector store.

Triggered IF there is no vector store folder in the database

You find here:
vector_store_creation
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from pathlib import Path


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

all_document = docx_docs + pdf_docs

for i, doc in enumerate(docx_docs):
    print(len(i))

for i, doc in enumerate(pdf_docs):
    print(len(i))


#folder with all the documents.
Path("LifeCoach_Data/Coaching_Training")

#Read the documents
    #if document is .docx
        #convert to PDF




def vector_store_creation():
    pass

vector_store_creation()

