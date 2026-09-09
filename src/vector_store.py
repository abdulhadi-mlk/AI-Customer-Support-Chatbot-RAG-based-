from pathlib import Path
import shutil

from chromadb.errors import ChromaError
from langchain_chroma import Chroma

from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.text_splitter import split_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECTION_NAME = "safex_support"


def _project_path(path: str | Path | None, default: Path) -> Path:
    """Resolve relative paths from the project root, not the launch directory."""
    candidate = Path(path) if path is not None else default
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _remove_vector_store(persist_dir: Path) -> None:
    """Remove only the configured Chroma directory before rebuilding it."""
    if persist_dir.is_dir():
        shutil.rmtree(persist_dir)
    elif persist_dir.exists():
        persist_dir.unlink()


def build_vector_store(
    data_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
) -> Chroma:
    """
    Load documents, split them into chunks, create embeddings,
    and store them in ChromaDB.
    """

    data_dir = _project_path(data_dir, PROJECT_ROOT / "data")
    persist_dir = _project_path(persist_dir, PROJECT_ROOT / "chroma_db")

    # Make sure the data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {data_dir}"
        )

    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError(
            f"No searchable chunks were produced from knowledge base files in "
            f"{data_dir}."
        )

    try:
        embeddings = get_embeddings()
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "Could not create embeddings for the knowledge base. "
            "Check the embedding dependencies and network access."
        ) from error

    try:
        persist_dir.parent.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
            collection_name=COLLECTION_NAME,
        )
        chunk_count = vector_store._collection.count()
    except (ChromaError, OSError, RuntimeError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"Could not initialize Chroma at {persist_dir}."
        ) from error

    if chunk_count == 0:
        raise RuntimeError(
            f"Chroma initialized at {persist_dir}, but collection "
            f"{COLLECTION_NAME!r} is empty."
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

    persist_dir = _project_path(persist_dir, PROJECT_ROOT / "chroma_db")

    if not persist_dir.exists():
        return build_vector_store(persist_dir=persist_dir)

    try:
        embeddings = get_embeddings()
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "Could not create embeddings for the existing knowledge base. "
            "Check the embedding dependencies and network access."
        ) from error

    try:
        vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        chunk_count = vector_store._collection.count()
    except (ChromaError, OSError, RuntimeError, ValueError, TypeError) as error:
        _remove_vector_store(persist_dir)
        try:
            return build_vector_store(persist_dir=persist_dir)
        except (
            ChromaError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as rebuild_error:
            raise RuntimeError(
                f"Could not load or rebuild the Chroma database at {persist_dir}."
            ) from rebuild_error

    if chunk_count == 0:
        _remove_vector_store(persist_dir)
        return build_vector_store(persist_dir=persist_dir)

    return vector_store


if __name__ == "__main__":
    store = get_vector_store()

    print(
        f"Vector database created with "
        f"{store._collection.count()} chunks."
    )