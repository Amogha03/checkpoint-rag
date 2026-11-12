"""Faithfulness evaluator: does the answer follow from retrieved context?"""

from typing import Any

from ._judge import input_data, reference_data, run_data, run_score_judge

INSTRUCTIONS = """Score factual faithfulness from 0 to 1. Compare every material claim in the answer with the retrieved contexts. Penalize unsupported claims; do not reward fluency. Return 1 only when the answer is fully supported."""


def evaluate_faithfulness(run: Any, example: Any) -> dict[str, Any]:
    outputs = run_data(run)
    result = run_score_judge(INSTRUCTIONS, {
        "query": input_data(run, example).get("query", ""),
        "answer": outputs.get("answer", ""),
        "contexts": outputs.get("contexts", []),
        "reference_answer": reference_data(example).get("ground_truth_answer", ""),
    })
    return {"key": "faithfulness", "score": float(result.score), "comment": result.rationale}