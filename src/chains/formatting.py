from typing import Iterable

from langchain_core.documents import Document


def source_label(document: Document) -> str:
    """A short, human-readable label: 'fund_a.pdf (page 3)' or a URL."""
    meta = document.metadata or {}
    source = meta.get("source", "unknown source")
    page = meta.get("page")
    return f"{source} (page {page + 1})" if isinstance(page, int) else str(source)


def format_documents(documents: Iterable[Document]) -> str:
    """Plain text for a prompt: numbered passages with their source, no Python reprs."""
    parts = [
        f"[{i}] Source: {source_label(doc)}\n{doc.page_content.strip()}"
        for i, doc in enumerate(documents or [], start=1)
    ]
    return "\n\n".join(parts) if parts else "(no documents)"