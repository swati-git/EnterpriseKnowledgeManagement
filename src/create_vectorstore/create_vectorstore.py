import os
from langchain_community.document_loaders import  DirectoryLoader, PyPDFLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import  Chroma
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain.schema import Document

load_dotenv()

def load_documents() -> list[Document]:
    path= Path(__file__).parent.parent.parent / "enterprise_data"
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    document_loader=DirectoryLoader(str(path) ,glob="*.pdf",loader_cls=PyPDFLoader) # type: ignore
    return document_loader.load() 

def split_text(documents: list[Document]):
  """
  Split the text content of the given list of Document objects into smaller chunks.
  Args:
    documents (list[Document]): List of Document objects containing text content to split.
  Returns:
    list[Document]: List of Document objects representing the split text chunks.
  """
  # Initialize text splitter with specified parameters
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Size of each chunk in characters
    chunk_overlap=100, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
  )
  chunks = text_splitter.split_documents(documents)

  return chunks 

def save_to_chroma(documents: list[Document]):
   global vectorstore
   
   vectorstore_db_path= Path(__file__).parent.parent.parent / "vectorstore_db"
   embeddings = CohereEmbeddings( model="embed-english-v3.0")   
   
   vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(vectorstore_db_path)
    )
   vectorstore.persist()

def initialize_retriever():
   retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
   return retriever

def initialize_vectorstore():
    
    documents = load_documents()
    split_documents = split_text(documents)
    save_to_chroma(split_documents)

