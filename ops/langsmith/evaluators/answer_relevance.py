"""Answer relevance evaluator: does the response address the CSM query?"""

from typing import Any

from ._judge import input_data, run_data, run_score_judge

INSTRUCTIONS = """Score answer relevance from 0 to 1. Judge whether the response directly and sufficiently answers the user's question. Penalize unrelated, evasive, or materially incomplete answers. Do not score factual correctness here."""


def evaluate_answer_relevance(run: Any, example: Any) -> dict[str, Any]:
    outputs = run_data(run)
    result = run_score_judge(INSTRUCTIONS, {
        "query": input_data(run, example).get("query", ""),
        "answer": outputs.get("answer", ""),
    })
    return {"key": "answer_relevance", "score": float(result.score), "comment": result.rationale}