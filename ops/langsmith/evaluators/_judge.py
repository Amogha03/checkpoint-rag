"""Shared structured scoring helpers for LLM-as-judge evaluators."""

import json
import os
from typing import Any

from pydantic import BaseModel, Field


class ScoreResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def run_score_judge(instructions: str, payload: dict[str, Any]) -> ScoreResult:
    """Score a response with the configured judge model."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
    ).with_structured_output(ScoreResult)
    return model.invoke(
        f"{instructions}\n\nEvaluate this JSON record:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def run_data(run: Any) -> dict[str, Any]:
    return getattr(run, "outputs", None) or (run.get("outputs", {}) if isinstance(run, dict) else {})


def reference_data(example: Any) -> dict[str, Any]:
    return getattr(example, "outputs", None) or (example.get("outputs", {}) if isinstance(example, dict) else {})


def input_data(run: Any, example: Any) -> dict[str, Any]:
    inputs = getattr(run, "inputs", None) or (run.get("inputs", {}) if isinstance(run, dict) else {})
    if inputs:
        return inputs
    return getattr(example, "inputs", None) or (example.get("inputs", {}) if isinstance(example, dict) else {})