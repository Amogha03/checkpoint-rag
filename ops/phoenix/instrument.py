"""Initialize Phoenix tracing and capture a sample set of RAG executions."""

import json
import sys
import time
from pathlib import Path
from typing import Any

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_SESSION = None
_INSTRUMENTED = False


def initialize_phoenix():
    """Launch the local Phoenix app and instrument LangChain once per process."""
    global _SESSION, _INSTRUMENTED
    if _SESSION is None:
        _SESSION = px.launch_app()
    if not _INSTRUMENTED:
        LangChainInstrumentor().instrument()
        _INSTRUMENTED = True
    return _SESSION


def run_sample_queries(
    limit: int = 10,
    test_set_path: str | Path = ROOT / "evals" / "test_set.json",
    output_path: str | Path = ROOT / "ops" / "phoenix" / "results" / "sample_traces.json",
) -> list[dict[str, Any]]:
    """Run at least ten evaluation queries and persist latency and retrieval metadata."""
    if limit < 10:
        raise ValueError("limit must be at least 10 to satisfy the sample tracing requirement")
    initialize_phoenix()

    from langchain_community.callbacks import get_openai_callback

    import pipeline

    payload = json.loads(Path(test_set_path).read_text(encoding="utf-8"))
    queries = payload.get("queries", [])[:limit]
    if len(queries) < 10:
        raise ValueError(f"At least 10 test queries are required; found {len(queries)}")

    results = []
    for item in queries:
        started = time.perf_counter()
        with get_openai_callback() as callback:
            response, documents = pipeline.run_pipeline(item["query"], top_k=4)
        results.append({
            "query_id": item["id"],
            "query": item["query"],
            "latency_sec": round(time.perf_counter() - started, 4),
            "token_usage": {
                "prompt_tokens": callback.prompt_tokens,
                "completion_tokens": callback.completion_tokens,
                "total_tokens": callback.total_tokens,
            },
            "retrieved_doc_ids": [
                doc.metadata.get("doc_id") or doc.metadata.get("source") or doc.metadata.get("source_path", "unknown")
                for doc in documents
            ],
            "response_sources": response.sources,
            "confidence": response.confidence,
        })

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"query_count": len(results), "results": results}, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    records = run_sample_queries()
    print(f"Captured {len(records)} Phoenix-traced query runs.")