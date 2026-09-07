import csv
from pathlib import Path

from src.rag_chain import REFUSAL, answer_question


QUESTIONS = [
    ("How long does standard shipping take?", "in"),
    ("What payment methods do you accept?", "in"),
    ("How do I reset my password?", "in"),
    ("Does SafeX provide a warranty?", "in"),
    ("Does SafeX ship internationally?", "in"),
    ("How much does express shipping cost?", "in"),
    ("Can I return a product?", "in"),
    ("When will I receive my refund?", "in"),
    ("Can I shop without an account?", "in"),
    ("How do I contact SafeX support?", "in"),
    ("What is the capital of Pakistan?", "out"),
    ("What is the weather today?", "out"),
    ("Who is the president of the USA?", "out"),
    ("Tell me a joke about computers.", "out"),
    ("What stock should I buy today?", "out"),
    ("Can I return something?", "ambiguous"),
    ("How long will it take?", "ambiguous"),
]


def run_evaluation(output_path: str | Path = "evaluation_results.csv") -> dict[str, float]:
    rows = []
    for question, category in QUESTIONS:
        result = answer_question(question)
        refused = result.answer == REFUSAL
        rows.append(
            {
                "question": question,
                "category": category,
                "generated_answer": result.answer,
                "is_refusal": refused,
                "sources": ", ".join(result.sources),
                "similarity_distance": result.score,
            }
        )

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    in_domain = [row for row in rows if row["category"] == "in"]
    out_domain = [row for row in rows if row["category"] in {"out", "ambiguous"}]
    correctly_refused = sum(row["is_refusal"] for row in out_domain)
    return {
        "grounded_answer_rate": sum(bool(row["sources"]) for row in in_domain) / len(in_domain),
        "correct_refusal_rate": correctly_refused / len(out_domain),
        "hallucination_rate": (len(out_domain) - correctly_refused) / len(out_domain),
    }


if __name__ == "__main__":
    print(run_evaluation())
