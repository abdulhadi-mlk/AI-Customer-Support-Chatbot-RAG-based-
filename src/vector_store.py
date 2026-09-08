from pathlib import Path

from langchain_chroma import Chroma

from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.text_splitter import split_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_vector_store(
    data_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
) -> Chroma:
    """
    Load documents, split them into chunks, create embeddings,
    and store them in ChromaDB.
    """

    data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data"
    persist_dir = (
        Path(persist_dir) if persist_dir else PROJECT_ROOT / "chroma_db"
    )

    # Make sure the data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {data_dir}"
        )

    # Load documents
    documents = load_documents(data_dir)

    # Split documents into chunks
    chunks = split_documents(documents)

    # Check if documents produced chunks
    if not chunks:
        raise ValueError(
            "The knowledge base produced no searchable chunks."
        )

    print("Creating vector database...")

    # Get embedding model
    embeddings = get_embeddings()

    # Create ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="safex_support",
    )

    print(
        f"Vector database created successfully with "
        f"{vector_store._collection.count()} chunks."
    )

    return vector_store


def get_vector_store(
    persist_dir: str | Path | None = None,
) -> Chroma:
    """
    Load the existing vector database.

    If the vector database does not exist,
    automatically create it.
    """

    persist_dir = (
        Path(persist_dir) if persist_dir else PROJECT_ROOT / "chroma_db"
    )

    # If vector database does not exist,
    # automatically build it
    if not persist_dir.exists():
        print(
            "Vector database not found. "
            "Creating it automatically..."
        )

        return build_vector_store(
            persist_dir=persist_dir
        )

    print("Loading existing vector database...")

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
        collection_name="safex_support",
    )


if __name__ == "__main__":
    store = build_vector_store()

    print(
        f"Vector database created with "
        f"{store._collection.count()} chunks."
    )