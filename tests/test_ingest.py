import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ingest


class DummyEmbeddings:
    def embed_query(self, text: str):
        return [0.1] * 1536


class DummyIndex:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def upsert(self, vectors, async_req=False):
        return type("AsyncResult", (), {"get": lambda self: None})()


class DummyPinecone:
    def __init__(self, api_key):
        self.api_key = api_key

    def Index(self, name, pool_threads=20):
        return DummyIndex()


def test_ingest_corpus_runs_without_errors(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "sample.md").write_text("# Sample\n\nThis is a short doc for ingestion tests.", encoding="utf-8")

    monkeypatch.setattr(ingest, "init_rag_components", lambda: DummyEmbeddings())
    monkeypatch.setattr(
        ingest,
        "load_documents",
        lambda _: [("sample.md", "markdown", "# Sample\n\nThis is a short doc for ingestion tests.")],
    )
    monkeypatch.setattr(ingest, "Pinecone", DummyPinecone)

    stats = ingest.ingest_corpus(str(corpus_dir))

    assert stats["files_processed"] >= 1
    assert stats["total_chunks"] >= 1
    assert stats["vectors_upserted"] >= 1
    assert set(stats).issuperset({"files_processed", "total_chunks", "vectors_upserted"})
