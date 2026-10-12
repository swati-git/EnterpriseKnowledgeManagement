"""Build the vector index once, then open it for every query.

Build or rebuild the index from the PDFs in enterprise_data/:
    python -m src.create_vectorstore.create_vectorstore            # builds only if empty
    python -m src.create_vectorstore.create_vectorstore --rebuild  # re-embeds everything
"""

import argparse
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "enterprise_data"
DB_DIR = PROJECT_ROOT / "vectorstore_db"
COLLECTION_NAME = "enterprise_documents"

# Starting values, not tuned. Fund documents state fees and terms in tables
# and long sentences, so very small chunks split them apart. Measure before changing.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

_vectorstore: Optional[Chroma] = None


def _default_embeddings() -> Embeddings:
    from langchain_cohere import CohereEmbeddings

    return CohereEmbeddings(model="embed-english-v3.0")


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"No document folder at {data_dir}. Put the PDFs there first.")
    loader = DirectoryLoader(str(data_dir), glob="*.pdf", loader_cls=PyPDFLoader)  # type: ignore[arg-type]
    return loader.load()


def split_text(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def get_vectorstore(
    embeddings: Optional[Embeddings] = None, persist_directory: Optional[Path] = None
) -> Chroma:
    """Open the persisted index. Never embeds anything."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings or _default_embeddings(),
            persist_directory=str(persist_directory or DB_DIR),
        )
    return _vectorstore


def reset_vectorstore_cache() -> None:
    global _vectorstore
    _vectorstore = None


def index_size() -> int:
    return get_vectorstore()._collection.count()


def build_vectorstore(rebuild: bool = False, documents: Optional[List[Document]] = None) -> int:
    """Embed the documents into the index. Skips the work if the index already has data.

    The original code called Chroma.from_documents on every start, which appended a
    fresh copy of every chunk each time, so retrieval started returning duplicates.
    """
    store = get_vectorstore()
    existing = store._collection.count()
    if existing and not rebuild:
        print(f"Index already has {existing} chunks; skipping. Use --rebuild to re-embed.")
        return existing

    if existing:
        store.delete(ids=store.get()["ids"])

    chunks = split_text(documents if documents is not None else load_documents())
    if not chunks:
        raise ValueError("No text found in the documents; nothing to index.")
    store.add_documents(chunks)
    count = store._collection.count()
    print(f"Indexed {count} chunks into {DB_DIR}.")
    return count


def get_retriever(k: int = TOP_K) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete the index and re-embed every document")
    build_vectorstore(rebuild=parser.parse_args().rebuild)