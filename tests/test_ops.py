import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ingest
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
    monkeypatch.setattr(phoenix_instrument, "_SESSION", None)
    monkeypatch.setattr(phoenix_instrument, "_INSTRUMENTED", False)
    monkeypatch.setattr(phoenix_instrument.px, "launch_app", lambda: calls.append("launch") or "session")

    class DummyInstrumentor:
        def instrument(self):
            calls.append("instrument")

    monkeypatch.setattr(phoenix_instrument, "LangChainInstrumentor", DummyInstrumentor)

    assert phoenix_instrument.initialize_phoenix() == "session"
    assert phoenix_instrument.initialize_phoenix() == "session"
    assert calls == ["launch", "instrument"]


def test_great_expectations_suite_validates_chunk_records():
    result = validate_records([{
        "chunk_text": "A valid chunk without personal data.",
        "source_path": "guide.md",
        "embedding": [0.0] * 1536,
    }])

    assert result.success
    assert result.statistics["evaluated_expectations"] == 5


def test_great_expectations_detects_invalid_vector_dimensions():
    result = validate_records([{
        "chunk_text": "A valid chunk without personal data.",
        "source_path": "guide.md",
        "embedding": [0.0] * 3,
    }])

    assert not result.success


def test_great_expectations_detects_missing_embedding():
    result = validate_records([{
        "chunk_text": "A valid chunk without personal data.",
        "source_path": "guide.md",
    }])

    assert not result.success


def test_great_expectations_detects_empty_chunk_text():
    result = validate_records([{
        "chunk_text": "",
        "source_path": "guide.md",
        "embedding": [0.0] * 1536,
    }])

    assert not result.success


def test_great_expectations_detects_pii():
    result = validate_records([{
        "chunk_text": "Contact alex@example.com for help.",
        "source_path": "guide.md",
        "embedding": [0.0] * 1536,
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
    assert vectors[0]["metadata"]["source_path"] == "guide.pdf"