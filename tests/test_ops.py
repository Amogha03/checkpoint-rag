import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ingest
from ops.data_quality import run_validations as validation_runner
from ops.data_quality.expectations import validate_records
from ops.langsmith import setup as langsmith_setup
from ops.langsmith.evaluators import answer_relevance, custom_judge, faithfulness
from ops.langsmith.evaluators._judge import ScoreResult
from ops.phoenix import instrument as phoenix_instrument


def test_langsmith_evaluators_return_numeric_scores(monkeypatch):
    monkeypatch.setattr(
        faithfulness,
        "run_score_judge",
        lambda *_: ScoreResult(score=0.8, rationale="Supported by context."),
    )
    monkeypatch.setattr(
        answer_relevance,
        "run_score_judge",
        lambda *_: ScoreResult(score=0.9, rationale="Directly answers the query."),
    )

    run = SimpleNamespace(outputs={"answer": "The limit is 1000 rpm.", "contexts": ["Limit: 1000 rpm."]})
    example = SimpleNamespace(
        inputs={"query": "What is the limit?"},
        outputs={"ground_truth_answer": "1000 rpm."},
    )

    results = [
        faithfulness.evaluate_faithfulness(run, example),
        answer_relevance.evaluate_answer_relevance(run, example),
    ]
    assert all(isinstance(result["score"], float) for result in results)
    assert [result["score"] for result in results] == [0.8, 0.9]


def test_custom_judge_returns_bounded_numeric_scores(monkeypatch):
    class DummyModel:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _prompt):
            return custom_judge.CSMJudgeResult(
                score=0.9,
                comment="The answer matches the reference and context.",
            )

    monkeypatch.setattr(custom_judge, "ChatOpenAI", lambda **_kwargs: DummyModel())
    result = custom_judge.csm_support_judge(
        SimpleNamespace(outputs={"answer": "Answer", "sources": ["plans.md"], "contexts": ["Plan facts"]}),
        SimpleNamespace(
            inputs={"query": "Question"},
            outputs={"ground_truth_answer": "Reference", "expected_sources": ["plans.md"]},
        ),
    )

    assert result["key"] == "csm_support_accuracy"
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0
    assert result["score"] == 0.9
    assert isinstance(result["comment"], str)


def test_langsmith_dataset_examples_have_stable_ids(tmp_path):
    test_set = tmp_path / "test_set.json"
    test_set.write_text(
        '{"queries":[{"id":"Q001","query":"Question?","ground_truth_answer":"Answer.",'
        '"expected_sources":["guide.md"],"category":"api","difficulty":"easy"}]}',
        encoding="utf-8",
    )

    first = langsmith_setup.load_examples(test_set)
    second = langsmith_setup.load_examples(test_set)

    assert first == second
    assert first[0]["inputs"]["query_id"] == "Q001"
    assert first[0]["outputs"]["expected_sources"] == ["guide.md"]


def test_phoenix_initialization_launches_and_instruments_once(monkeypatch):
    calls = []
    session = SimpleNamespace(active=True)
    monkeypatch.setattr(phoenix_instrument, "_SESSION", None)
    monkeypatch.setattr(phoenix_instrument, "_INSTRUMENTED", False)
    monkeypatch.setattr(phoenix_instrument, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(phoenix_instrument, "_launch_phoenix", lambda: calls.append("launch") or session)
    provider = SimpleNamespace(get_tracer=lambda _name: phoenix_instrument._TRACER)
    monkeypatch.setattr(phoenix_instrument, "_configure_phoenix_tracing", lambda: calls.append("configure") or provider)
    monkeypatch.setattr(phoenix_instrument, "_instrument_langchain", lambda _provider: calls.append("instrument"))

    assert phoenix_instrument.initialize_phoenix() is session
    assert phoenix_instrument.initialize_phoenix() is session
    assert calls == ["launch", "configure", "instrument"]


def test_great_expectations_suite_validates_chunk_records():
    chunk_text = "word " * 60
    result = validate_records([{
        "id": "vector-0",
        "values": [0.0] * 1536,
        "metadata": {
            "chunk_text": chunk_text,
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }])

    assert result.success
    assert result.statistics["evaluated_expectations"] == 9


def test_validation_runner_writes_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(validation_runner, "RESULTS_DIR", tmp_path)
    records = [{
        "id": "vector-0",
        "values": [0.0] * 1536,
        "metadata": {
            "chunk_text": "word " * 60,
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }]

    result = validation_runner.run_validations(records)
    saved = json.loads((tmp_path / "validation_results.json").read_text(encoding="utf-8"))
    report = (tmp_path / "validation_report.md").read_text(encoding="utf-8")

    assert result["success"] is True
    assert saved["statistics"]["evaluated_expectations"] == 9
    assert "Great Expectations validation: PASS" in report


def test_great_expectations_detects_invalid_vector_dimensions():
    result = validate_records([{
        "values": [0.0] * 3,
        "metadata": {
            "chunk_text": "word " * 60,
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }])

    assert not result.success


def test_great_expectations_detects_missing_embedding():
    result = validate_records([{
        "metadata": {
            "chunk_text": "word " * 60,
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }])

    assert not result.success


def test_great_expectations_detects_empty_chunk_text():
    result = validate_records([{
        "values": [0.0] * 1536,
        "metadata": {
            "chunk_text": "",
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }])

    assert not result.success


def test_great_expectations_detects_pii():
    result = validate_records([{
        "values": [0.0] * 1536,
        "metadata": {
            "chunk_text": "word " * 60 + " alex@example.com",
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }])

    assert not result.success


def test_great_expectations_allows_shared_role_email():
    result = validate_records([{
        "values": [0.0] * 1536,
        "metadata": {
            "chunk_text": "Contact security@helix.io for allowlist questions. " + "word " * 60,
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }])

    assert result.success


def test_great_expectations_detects_api_key_patterns():
    result = validate_records([{
        "values": [0.0] * 1536,
        "metadata": {
            "chunk_text": "token = sk-" + "A" * 32 + " " + "word " * 60,
            "source_path": "guide.md",
            "chunk_index": 0,
            "doc_type": "markdown",
        },
    }])

    assert not result.success


def test_ingestion_chunks_default_to_100_tokens_and_preserve_source_path():
    class DummyEmbeddings:
        def embed_query(self, _text):
            return [0.0] * 1536

    chunks = ingest.chunk_document("word " * 500, "pdf")
    vectors = ingest.create_chunk_vectors(chunks, "guide.pdf", "pdf", DummyEmbeddings())

    assert chunks
    assert max(ingest.estimate_tokens(chunk) for chunk in chunks) <= 100
    assert min(ingest.estimate_tokens(chunk) for chunk in chunks) >= 50
    assert vectors[0]["metadata"]["source_path"] == "guide.pdf"


def test_phoenix_sample_runner_writes_ten_query_spans(tmp_path, monkeypatch):
    class DummySpan:
        def __init__(self):
            self.attributes = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def set_attribute(self, name, value):
            self.attributes[name] = value

        def record_exception(self, _error):
            pass

        def set_status(self, _status):
            pass

    class DummyTracer:
        def __init__(self):
            self.spans = []

        def start_as_current_span(self, _name):
            span = DummySpan()
            self.spans.append(span)
            return span

    class DummyCallback:
        prompt_tokens = 10
        completion_tokens = 4
        total_tokens = 14

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    test_set = tmp_path / "eval.json"
    test_set.write_text(
        json.dumps({"queries": [{"id": f"Q{i:03}", "query": f"Question {i}?"} for i in range(10)]}),
        encoding="utf-8",
    )
    pipeline = type(sys)("pipeline")
    pipeline.run_pipeline = lambda _query, top_k=4: (
        SimpleNamespace(sources=["guide.md"], confidence="high"),
        [SimpleNamespace(page_content="context", metadata={"doc_id": "guide.md"})],
    )
    callbacks = type(sys)("langchain_community.callbacks")
    callbacks.get_openai_callback = DummyCallback
    tracer = DummyTracer()
    monkeypatch.setattr(phoenix_instrument, "_TRACER", tracer)
    monkeypatch.setattr(phoenix_instrument, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(phoenix_instrument, "initialize_phoenix", lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "langchain_community.callbacks", callbacks)
    output_path = tmp_path / "traces.json"

    results = phoenix_instrument.run_sample_queries(
        test_set_path=test_set,
        output_path=output_path,
        launch_server=False,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(results) == saved["query_count"] == len(tracer.spans) == 10
    assert all(span.attributes["llm.token_count.total"] == 14 for span in tracer.spans)
    assert all(
        json.loads(span.attributes["retrieval.document_ids"]) == ["guide.md"]
        for span in tracer.spans
    )
    assert all(record["retrieved_doc_ids"] == ["guide.md"] for record in results)