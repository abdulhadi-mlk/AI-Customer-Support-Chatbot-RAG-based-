from pathlib import Path

from langchain_chroma import Chroma

from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.text_splitter import split_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_vector_store(
    data_dir: str | Path | None = None, persist_dir: str | Path | None = None
) -> Chroma:
    data_dir = data_dir or PROJECT_ROOT / "data"
    persist_dir = persist_dir or PROJECT_ROOT / "chroma_db"
    chunks = split_documents(load_documents(data_dir))
    if not chunks:
        raise ValueError("The knowledge base produced no searchable chunks.")
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name="safex_support",
    )


def get_vector_store(persist_dir: str | Path | None = None) -> Chroma:
    persist_dir = persist_dir or PROJECT_ROOT / "chroma_db"
    directory = Path(persist_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"Vector database not found at {directory}. Run: python -m src.vector_store"
        )
    return Chroma(
        persist_directory=str(directory),
        embedding_function=get_embeddings(),
        collection_name="safex_support",
    )


if __name__ == "__main__":
    store = build_vector_store()
    print(f"Vector database created with {store._collection.count()} chunks.")
