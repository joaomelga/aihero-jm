"""
Generate a golden Q&A dataset from the indexed knowledge base.

Usage:
    python -m src.generate_questions
"""

import json

from pydantic import BaseModel
from pydantic_ai import Agent

from src.config import EVAL_MODEL, GOLDEN_DATASET_PATH, load_env
from src.search import get_collection

GENERATION_PROMPT = """
You are an expert at creating evaluation datasets for RAG (Retrieval-Augmented Generation) systems.

Given a set of documentation chunks from the Evidently library, generate exactly 20 diverse
question-answer pairs that test different aspects of the knowledge base.

Guidelines:
- Questions should be specific and answerable from the provided documentation
- Cover different topics across the documentation (installation, features, API, concepts, etc.)
- Include a mix of factual questions, how-to questions, and conceptual questions
- Expected answers should be concise but accurate based on the source material
- Each question should be standalone (not referencing other questions)
""".strip()


class QuestionAnswer(BaseModel):
    question: str
    expected_answer: str


class QuestionSet(BaseModel):
    questions: list[QuestionAnswer]


def generate_questions():
    """Generate golden Q&A dataset from ChromaDB content."""
    load_env()

    collection = get_collection()
    sample = collection.get(limit=50, include=["documents", "metadatas"])

    chunks_text = ""
    for i, doc in enumerate(sample["documents"]):
        filename = sample["metadatas"][i].get("filename", "unknown")
        chunks_text += f"\n--- Chunk from {filename} ---\n{doc}\n"

    question_agent = Agent(
        name="question_generator",
        model=EVAL_MODEL,
        instructions=GENERATION_PROMPT,
        output_type=QuestionSet,
    )

    import asyncio

    result = asyncio.run(
        question_agent.run(
            user_prompt=f"Generate 20 Q&A pairs from these documentation chunks:\n{chunks_text}"
        )
    )

    questions_data = [q.model_dump() for q in result.output.questions]

    GOLDEN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(questions_data, f, indent=2)

    print(f"Generated {len(questions_data)} questions")
    print(f"Saved to {GOLDEN_DATASET_PATH}")
    return questions_data


if __name__ == "__main__":
    generate_questions()
