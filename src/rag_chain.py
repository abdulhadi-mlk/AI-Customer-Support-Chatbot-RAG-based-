from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_google_api_key
from src.retriever import REFUSAL, RetrievalResult, retrieve
from src.vector_store import get_vector_store


def extract_text_content(content: object) -> str:
    """Return visible text from a model response's content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a customer support assistant for SafeX.
Answer ONLY using the provided context. Do not use outside knowledge.
Do not make up information. If the answer is not present in the context,
reply exactly: "{refusal}"
Keep answers clear, helpful, and concise.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


@dataclass
class ChatResult:
    answer: str
    sources: list[str]
    score: float | None


def answer_question(
    question: str, vector_store=None, model_name: str = "gemini-3.6-flash"
) -> ChatResult:
    if not question.strip():
        return ChatResult("Please enter a question.", [], None)
    if len(question.strip()) < 3:
        return ChatResult("Please enter a more detailed question.", [], None)
    api_key = get_google_api_key()

    store = vector_store or get_vector_store()
    result: RetrievalResult = retrieve(store, question)
    if not result.has_context:
        return ChatResult(REFUSAL, [], result.best_score)

    context = "\n\n".join(document.page_content for document in result.documents)
    messages = PROMPT.format_messages(context=context, question=question, refusal=REFUSAL)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )
    response = llm.invoke(messages)
    answer = extract_text_content(response.content)
    return ChatResult(
        answer=answer.strip(),
        sources=sorted({document.metadata.get("source", "unknown") for document in result.documents}),
        score=result.best_score,
    )
