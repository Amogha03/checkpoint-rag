"""Run Great Expectations checks against a local representative chunk batch."""

import json
from pathlib import Path
from typing import Any

from ops.data_quality.expectations import EMBEDDING_DIMENSIONS, validate_records

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "ops" / "data_quality" / "results"


def build_sample_records(corpus_dir: str | Path = ROOT / "corpus") -> list[dict[str, Any]]:
    """Build pre-upsert-shaped records from ingestion chunks using local vectors."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from ingest import chunk_document, create_chunk_vectors

    class ShapeOnlyEmbeddings:
        def embed_query(self, _text: str) -> list[float]:
            return [0.0] * EMBEDDING_DIMENSIONS

    corpus_path = Path(corpus_dir)
    records = []
    embeddings = ShapeOnlyEmbeddings()
    for path in sorted(corpus_path.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_document(text, "markdown")
        if not chunks:
            continue
        source = str(path.relative_to(corpus_path))
        records.extend(create_chunk_vectors(chunks, source, "markdown", embeddings))
        if len(records) >= 25:
            return records[:25]
    if not records:
        raise ValueError(f"No Markdown chunks found in {corpus_path}")
    return records


def run_validations(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = records if records is not None else build_sample_records()
    result = validate_records(records)
    payload = result.to_json_dict()
    payload["fixture_note"] = "Vector values are zero-valued 1536-dimensional shape fixtures, not model outputs."
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "validation_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    summary = [
        f"Great Expectations validation: {'PASS' if payload.get('success') else 'FAIL'}",
        f"Evaluated {payload.get('statistics', {}).get('evaluated_expectations', 0)} expectations ",
        f"against {len(records)} chunk records.",
        "",
        "| Expectation | Result | Unexpected |",
        "|---|---:|---:|",
    ]
    for item in payload.get("results", []):
        config = item.get("expectation_config", {})
        expectation = config.get("expectation_type", "unknown")
        column = config.get("kwargs", {}).get("column", "")
        unexpected = item.get("result", {}).get("unexpected_count", 0)
        summary.append(f"| `{expectation}` ({column}) | {'PASS' if item.get('success') else 'FAIL'} | {unexpected} |")
    (RESULTS_DIR / "validation_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    report = run_validations()
    print(f"Data quality suite success: {report['success']}")
    print(f"Reports saved to {RESULTS_DIR}")
    if not report["success"]:
        raise SystemExit(1)