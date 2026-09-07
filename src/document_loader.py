from pathlib import Path
import re

from langchain_core.documents import Document


def load_documents(data_dir: str | Path = "data") -> list[Document]:
    """Load non-empty knowledge-base files as LangChain documents."""
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge-base directory not found: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.glob("*.txt")):
        text = re.sub(r"[ \t]+", " ", path.read_text(encoding="utf-8")).strip()
        if text:
            documents.append(Document(page_content=text, metadata={"source": path.name}))

    if not documents:
        raise ValueError(f"No non-empty .txt knowledge-base files found in {directory}")
    return documents
