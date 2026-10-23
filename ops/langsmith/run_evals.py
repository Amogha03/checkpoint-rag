"""Run the RAG pipeline against the LangSmith dataset and save scores locally."""

import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ops.langsmith.evaluators.answer_relevance import evaluate_answer_relevance
from ops.langsmith.evaluators.custom_judge import csm_support_judge
from ops.langsmith.evaluators.faithfulness import evaluate_faithfulness
from ops.langsmith.setup import DEFAULT_DATASET_NAME, setup_dataset
from pipeline import run_pipeline


def run_target(inputs: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response, documents = run_pipeline(inputs["query"], top_k=4)
    contexts = [doc.page_content for doc in documents]
    doc_ids = [
        doc.metadata.get("doc_id") or doc.metadata.get("source") or doc.metadata.get("source_path", "unknown")
        for doc in documents
    ]
    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "contexts": contexts,
        "retrieved_doc_ids": doc_ids,
        "latency_sec": round(time.perf_counter() - started, 4),
    }


def run_evaluations(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str | Path = ROOT / "ops" / "langsmith" / "results",
    client: Client | None = None,
):
    load_dotenv(ROOT / ".env")
    client = client or Client()
    setup_dataset(client=client, dataset_name=dataset_name)
    scores: list[dict[str, Any]] = []

    def capture(evaluator):
        def wrapped(run, example):
            result = evaluator(run, example)
            query_id = (getattr(example, "metadata", None) or {}).get("query_id")
            results = result if isinstance(result, list) else [result]
            scores.extend({"query_id": query_id, **item} for item in results)
            return result
        return wrapped

    experiment = evaluate(
        run_target,
        data=dataset_name,
        evaluators=[
            capture(evaluate_faithfulness),
            capture(evaluate_answer_relevance),
            capture(csm_support_judge),
        ],
        client=client,
        experiment_prefix="checkpoint-rag-ops",
        description="Faithfulness, relevance, and CSM support accuracy evaluation.",
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "eval_scores.json"
    result_path.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")
    return experiment


if __name__ == "__main__":
    run_evaluations()