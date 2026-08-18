import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.documents import Document

import retrieve


class DummyEmbeddings:
    def embed_query(self, text: str):
        return [0.1] * 1536


class DummyIndex:
    def query(self, vector, top_k, include_metadata):
        return {
            "matches": [
                {
                    "id": "match-1",
                    "metadata": {
                        "source_path": "sample.md",
                        "chunk_text": "This is a test retrieval result for the query.",
                    },
                }
            ]
        }


def test_retrieve_returns_at_least_one_result_for_each_eval_query(monkeypatch):
    with open(ROOT / "evals" / "test_set.json", "r", encoding="utf-8") as fh:
        queries = json.load(fh)["queries"]

    monkeypatch.setattr(
        retrieve,
        "init_rag_components",
        lambda: (DummyIndex(), DummyEmbeddings()),
    )

    for item in queries:
        docs = retrieve.retrieve(item["query"], top_k=4)
        assert len(docs) >= 1
        assert all(isinstance(doc, Document) for doc in docs)
        assert any(doc.page_content for doc in docs)
