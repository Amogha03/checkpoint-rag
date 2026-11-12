"""Initialize Phoenix tracing and capture a sample set of RAG executions."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_SESSION = None
_INSTRUMENTED = False
_TRACER_PROVIDER = None
_TRACER = trace.get_tracer("checkpoint-rag.ops.phoenix")


def _launch_phoenix():
    import phoenix as px

    return px.launch_app(run_in_thread=True)


def _configure_phoenix_tracing():
    tracer_provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": "checkpoint-rag"}),
    )
    exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)
    return tracer_provider


def _instrument_langchain(tracer_provider) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


def initialize_phoenix(launch_server: bool = True):
    """Launch Phoenix and export custom and LangChain spans to its local collector."""
    global _SESSION, _INSTRUMENTED, _TRACER_PROVIDER, _TRACER
    if launch_server and _SESSION is None:
        _SESSION = _launch_phoenix()
        if _SESSION is None or not _SESSION.active:
            raise RuntimeError("Phoenix failed to start; check whether port 6006 is already in use.")
    if _TRACER_PROVIDER is None:
        _TRACER_PROVIDER = _configure_phoenix_tracing()
        _TRACER = _TRACER_PROVIDER.get_tracer("checkpoint-rag.ops.phoenix")
    if not _INSTRUMENTED:
        _instrument_langchain(_TRACER_PROVIDER)
        _INSTRUMENTED = True
    return _SESSION


def run_sample_queries(
    limit: int = 10,
    test_set_path: str | Path = ROOT / "evals" / "test_set.json",
    output_path: str | Path = ROOT / "ops" / "phoenix" / "results" / "sample_traces.json",
    launch_server: bool = True,
) -> list[dict[str, Any]]:
    """Run at least ten evaluation queries and persist latency and retrieval metadata."""
    if limit < 10:
        raise ValueError("limit must be at least 10 to satisfy the sample tracing requirement")
    initialize_phoenix(launch_server=launch_server)

    from langchain_community.callbacks import get_openai_callback

    import pipeline

    payload = json.loads(Path(test_set_path).read_text(encoding="utf-8"))
    queries = payload.get("queries", [])[:limit]
    if len(queries) < 10:
        raise ValueError(f"At least 10 test queries are required; found {len(queries)}")

    results = []
    for item in queries:
        started = time.perf_counter()
        with _TRACER.start_as_current_span("checkpoint_rag.query") as span:
            span.set_attribute("query.id", str(item["id"]))
            span.set_attribute("input.value", item["query"])
            try:
                with get_openai_callback() as callback:
                    response, documents = pipeline.run_pipeline(item["query"], top_k=4)

                latency_sec = round(time.perf_counter() - started, 4)
                token_usage = {
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens,
                    "total_tokens": callback.total_tokens,
                }
                document_ids = [
                    str(doc.metadata.get("doc_id") or doc.metadata.get("source_path") or "unknown")
                    for doc in documents
                ]
                source_paths = [
                    str(doc.metadata.get("source_path") or doc.metadata.get("doc_id") or "unknown")
                    for doc in documents
                ]

                span.set_attribute("rag.latency_ms", latency_sec * 1000)
                span.set_attribute("llm.token_count.prompt", token_usage["prompt_tokens"])
                span.set_attribute("llm.token_count.completion", token_usage["completion_tokens"])
                span.set_attribute("llm.token_count.total", token_usage["total_tokens"])
                span.set_attribute("retrieval.document_ids", json.dumps(document_ids))
                span.set_attribute("retrieval.source_paths", json.dumps(source_paths))
                span.set_attribute("retrieval.document_count", len(documents))
                span.set_attribute("output.sources", json.dumps(response.sources))

                results.append({
                    "query_id": item["id"],
                    "query": item["query"],
                    "latency_sec": latency_sec,
                    "token_usage": token_usage,
                    "retrieved_doc_ids": document_ids,
                    "retrieved_source_paths": source_paths,
                    "response_sources": response.sources,
                    "confidence": response.confidence,
                })
            except Exception as error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.set_attribute("rag.latency_ms", (time.perf_counter() - started) * 1000)
                raise

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"query_count": len(results), "results": results}, indent=2), encoding="utf-8")
    if _TRACER_PROVIDER is not None:
        _TRACER_PROVIDER.force_flush()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace sample RAG queries with Phoenix.")
    parser.add_argument(
        "--connect-existing",
        action="store_true",
        help="Export to an already running Phoenix server at localhost:6006 instead of launching one.",
    )
    args = parser.parse_args()
    records = run_sample_queries(launch_server=not args.connect_existing)
    print(f"Captured {len(records)} Phoenix-traced query runs.")
    if args.connect_existing:
        print("Traces exported to the existing Phoenix server at http://localhost:6006/.")
    else:
        print("Phoenix is available at http://localhost:6006/; press Ctrl+C to stop the server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            if _TRACER_PROVIDER is not None:
                _TRACER_PROVIDER.force_flush()
                _TRACER_PROVIDER.shutdown()
            print("Phoenix tracing stopped.")