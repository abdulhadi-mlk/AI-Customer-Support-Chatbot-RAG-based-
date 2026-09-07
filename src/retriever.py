from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma


REFUSAL = "I could not find this information in the available knowledge base."


@dataclass
class RetrievalResult:
    documents: list[Any]
    best_score: float | None

    @property
    def has_context(self) -> bool:
        return bool(self.documents)


def retrieve(
    vector_store: Chroma,
    question: str,
    k: int = 5,
    max_distance: float = 1.2
) -> RetrievalResult:
    question = question.strip()
    if not question:
        return RetrievalResult([], None)

    matches = vector_store.similarity_search_with_score(question, k=k)
    accepted = [(document, score) for document, score in matches if score <= max_distance]
    return RetrievalResult(
        documents=[document for document, _ in accepted],
        best_score=accepted[0][1] if accepted else (matches[0][1] if matches else None),
    )
