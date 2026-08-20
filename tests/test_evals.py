import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.documents import Document


spec = importlib.util.spec_from_file_location("eval_harness", ROOT / "evals" / "harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class DummyResponse:
    def __init__(self, answer="A grounded answer.", sources=None, confidence="high"):
        self.answer = answer
        self.sources = sources or ["sample.md"]
        self.confidence = confidence


class DummyResults:
    def __init__(self, rows):
        self._rows = rows

    def to_pandas(self):
        class DummyDataFrame:
            def __init__(self, rows):
                self.iloc = rows

        return DummyDataFrame(self._rows)


class DummyEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


class DummyChatModel:
    def __init__(self, *args, **kwargs):
        pass


def test_eval_harness_executes_end_to_end(tmp_path, monkeypatch):
    small_test_set = tmp_path / "mini_test_set.json"
    small_test_set.write_text(
        json.dumps({
            "queries": [
                {
                    "id": "Q001",
                    "query": "What is the rate limit?",
                    "ground_truth_answer": "The rate limit is 1000 requests per minute.",
                    "expected_sources": ["sample.md"],
                    "difficulty": "easy",
                    "category": "api",
                    "notes": "",
                },
                {
                    "id": "Q002",
                    "query": "What is the timeout?",
                    "ground_truth_answer": "The timeout is 30 minutes.",
                    "expected_sources": ["sample.md"],
                    "difficulty": "easy",
                    "category": "api",
                    "notes": "",
                },
            ]
        }),
        encoding="utf-8",
    )

    queries = json.loads(small_test_set.read_text(encoding="utf-8"))["queries"]

    def fake_run_pipeline(query, documents=None, top_k=4):
        return (
            DummyResponse(answer=f"Answer for: {query}", sources=["sample.md"], confidence="high"),
            [Document(page_content="Relevant context for the query.", metadata={"source_path": "sample.md"})],
        )

    rows = [
        {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
        }
        for _ in queries
    ]

    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(harness, "Dataset", type("DummyDataset", (), {"from_list": staticmethod(lambda records: object())}))
    monkeypatch.setattr(harness, "evaluate", lambda **kwargs: DummyResults(rows))
    monkeypatch.setattr(harness, "ChatOpenAI", DummyChatModel)
    monkeypatch.setattr(harness, "OpenAIEmbeddings", DummyEmbeddings)

    output_dir = tmp_path / "results"
    harness.run_eval_harness(
        test_set_path=str(small_test_set),
        output_dir=str(output_dir),
        documents=None,
    )

    json_path = output_dir / "eval_results.json"
    md_path = output_dir / "eval_report.md"

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total_queries"] == len(queries)
    assert payload["overall_metrics"]["faithfulness"] >= 0
    assert "query_results" in payload
    assert len(payload["query_results"]) == len(queries)
