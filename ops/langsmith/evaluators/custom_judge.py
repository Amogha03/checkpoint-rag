"""LLM-as-a-Judge evaluator for internal CSM support answers."""

import json
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ._judge import input_data, reference_data, run_data


class CSMJudgeResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    comment: str


def csm_support_judge(run: Any, example: Any) -> dict[str, Any]:
    """Score an answer against its query and expected output on a 0-1 scale."""
    inputs = input_data(run, example)
    outputs = run_data(run)
    reference = reference_data(example)
    record = {
        "query": inputs.get("query", ""),
        "generated_answer": outputs.get("answer", ""),
        "expected_output": reference.get("ground_truth_answer") or reference.get("answer", ""),
        "retrieved_contexts": outputs.get("contexts", []),
        "expected_sources": reference.get("expected_sources", []),
        "answer_sources": outputs.get("sources", []),
    }

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CSMJudgeResult)
    result = judge.invoke(
        "You evaluate internal customer-success support answers. Score how accurately and completely "
        "the generated answer responds to the query and matches the expected output. Use only the expected "
        "output and retrieved contexts as evidence. Penalize unsupported or incorrect claims, missing key "
        "facts, and incorrect citations when source information is available. Return 0.0 for incorrect or "
        "unusable answers and 1.0 for fully accurate, grounded answers, with a concise reason.\n\n"
        f"Evaluation record:\n{json.dumps(record, ensure_ascii=True)}"
    )
    return {
        "key": "csm_support_accuracy",
        "score": float(result.score),
        "comment": result.comment,
    }