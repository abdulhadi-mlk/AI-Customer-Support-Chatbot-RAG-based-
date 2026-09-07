from src.document_loader import load_documents
from src.rag_chain import extract_text_content
from src.retriever import REFUSAL, retrieve
from src.text_splitter import split_documents


class FakeStore:
    def __init__(self, matches):
        self.matches = matches

    def similarity_search_with_score(self, question, k=3):
        return self.matches[:k]


def test_knowledge_base_loads_and_splits():
    documents = load_documents("data")
    chunks = split_documents(documents)
    assert len(documents) == 3
    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_retriever_rejects_low_confidence_match():
    result = retrieve(FakeStore([("irrelevant", 0.99)]), "unrelated question")
    assert not result.has_context
    assert result.documents == []


def test_refusal_message_is_stable():
    assert REFUSAL == "I could not find this information in the available knowledge base."


def test_extract_text_content_accepts_plain_string():
    assert extract_text_content("Plain answer") == "Plain answer"


def test_extract_text_content_filters_and_concatenates_text_blocks():
    content = [
        {"type": "thinking", "thinking": "hidden reasoning"},
        {"type": "text", "text": "First ", "extras": {"signature": "ignored"}},
        {"type": "tool_use", "name": "ignored"},
        {"type": "text", "text": "second."},
    ]

    assert extract_text_content(content) == "First second."
