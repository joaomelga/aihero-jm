"""
Evaluation pipeline: run RAG agent on golden questions, score with LLM-as-judge.

Usage:
    python -m src.evaluate
"""

import asyncio
import json
from datetime import datetime

from pydantic import BaseModel
from pydantic_ai import Agent

from src.agent import create_agent
from src.config import EVAL_MODEL, EVAL_RESULTS_PATH, GOLDEN_DATASET_PATH, load_env

EVALUATION_PROMPT = """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).

We also include the entire log (<LOG>) for analysis.

For each item, check if the condition is met.
Checklist:
- instructions_follow: The agent followed the user's instructions (in <INSTRUCTIONS>)
- instructions_avoid: The agent avoided doing things it was told not to do
- answer_relevant: The response directly addresses the user's question
- answer_clear: The answer is clear and correct
- answer_citations: The response includes proper citations or sources when required
- completeness: The response is complete and covers all key aspects of the request
- tool_call_search: Is the search tool invoked?

Output true/false for each check and provide a short explanation for your judgment.
""".strip()

USER_PROMPT_FORMAT = """
<INSTRUCTIONS>{instructions}</INSTRUCTIONS>
<QUESTION>{question}</QUESTION>
<ANSWER>{answer}</ANSWER>
<LOG>{log}</LOG>
""".strip()


class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]
    summary: str


def simplify_log_messages(messages: list[dict]) -> list[dict]:
    """Simplify log messages by removing unnecessary fields and redacting search results."""
    log_simplified = []

    for m in messages:
        parts = []

        for original_part in m["parts"]:
            part = original_part.copy()
            kind = part["part_kind"]

            if kind == "user-prompt":
                part.pop("timestamp", None)
            if kind == "tool-call":
                part.pop("tool_call_id", None)
            if kind == "tool-return":
                part.pop("tool_call_id", None)
                part.pop("metadata", None)
                part.pop("timestamp", None)
                part["content"] = "RETURN_RESULTS_REDACTED"
            if kind == "text":
                part.pop("id", None)

            parts.append(part)

        message = {"kind": m["kind"], "parts": parts}
        log_simplified.append(message)

    return log_simplified


async def _evaluate_single(
    rag_agent: Agent,
    eval_agent: Agent,
    question: str,
    expected_answer: str,
) -> dict:
    """Run RAG agent on a question and evaluate with LLM-as-judge."""
    result = await rag_agent.run(user_prompt=question)
    answer = result.output

    from pydantic_ai.messages import ModelMessagesTypeAdapter

    dict_messages = ModelMessagesTypeAdapter.dump_python(result.new_messages())
    log_simplified = simplify_log_messages(dict_messages)
    log = json.dumps(log_simplified)

    user_prompt = USER_PROMPT_FORMAT.format(
        instructions=rag_agent._instructions,
        question=question,
        answer=answer,
        log=log,
    )

    eval_result = await eval_agent.run(
        user_prompt, output_type=EvaluationChecklist
    )
    checklist = eval_result.output

    return {
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": answer,
        "checklist": [c.model_dump() for c in checklist.checklist],
        "summary": checklist.summary,
        "all_passed": all(c.check_pass for c in checklist.checklist),
    }


def run_evaluation():
    """Run full evaluation pipeline on the golden dataset."""
    load_env()

    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {GOLDEN_DATASET_PATH}. "
            "Run `python -m src.generate_questions` first."
        )

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {GOLDEN_DATASET_PATH}")

    rag_agent = create_agent()
    eval_agent = Agent(
        name="eval_agent",
        model=EVAL_MODEL,
        instructions=EVALUATION_PROMPT,
        output_type=EvaluationChecklist,
    )

    async def _run_all():
        results = []
        for i, q in enumerate(questions):
            print(f"  [{i + 1}/{len(questions)}] {q['question'][:60]}...")
            result = await _evaluate_single(
                rag_agent, eval_agent, q["question"], q["expected_answer"]
            )
            results.append(result)
        return results

    results = asyncio.run(_run_all())

    passed = sum(1 for r in results if r["all_passed"])
    total = len(results)

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "all_passed": passed,
        "pass_rate": f"{passed / total * 100:.1f}%",
        "results": results,
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults: {passed}/{total} questions passed all checks ({output['pass_rate']})")
    print(f"Saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
