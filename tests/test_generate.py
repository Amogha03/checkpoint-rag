import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.documents import Document

import generate


class DummyChain:
    def invoke(self, payload):
        assert "context" in payload
        assert "query" in payload
        return generate.RAGResponse(
            answer="The answer is grounded in the provided context.",
            sources=["sample.md"],
            confidence="high",
        )


def test_generate_response_returns_valid_structured_output(monkeypatch):
    monkeypatch.setattr(
        generate,
        "retrieve",
        lambda query, top_k=4, documents=None: [
            Document(page_content="This is relevant context.", metadata={"doc_id": "sample.md"})
        ],
    )
    monkeypatch.setattr(generate, "build_rag_chain", lambda: DummyChain())

    response = generate.generate_response("What does the docs say?")

    assert isinstance(response, generate.RAGResponse)
    assert response.answer
    assert isinstance(response.sources, list) and len(response.sources) >= 1
    assert response.confidence in {"low", "medium", "high"}
    assert response.model_dump()
