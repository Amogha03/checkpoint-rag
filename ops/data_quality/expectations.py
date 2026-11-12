"""Great Expectations suite for RAG document chunks and embedding vectors."""

import re
from typing import Any

import pandas as pd
from great_expectations.core.batch import Batch
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations.expectation import ExpectationConfiguration
from great_expectations.validator.validator import Validator

MIN_CHUNK_TOKENS = 50
MAX_CHUNK_TOKENS = 300
EMBEDDING_DIMENSIONS = 1536
EMAIL_REGEX = re.compile(r"\b([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.IGNORECASE)
ROLE_EMAIL_NAMES = {
    "abuse", "admin", "contact", "help", "info", "legal", "postmaster",
    "privacy", "sales", "security", "support",
}
PII_REGEX = re.compile(
    r"(?is).*(?:\b\d{3}-\d{2}-\d{4}\b|"
    r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\bpcsk_[A-Za-z0-9_-]{20,}\b|"
    r"\blsv2_[A-Za-z0-9_-]{20,}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}).*",
    re.IGNORECASE,
)


def contains_pii(text: str) -> bool:
    """Detect personal email addresses, phone/SSN values, and common API keys."""
    if PII_REGEX.search(text):
        return True
    return any(
        local_part.lower() not in ROLE_EMAIL_NAMES
        for local_part, _domain in EMAIL_REGEX.findall(text)
    )


def build_expectation_suite() -> ExpectationSuite:
    suite = ExpectationSuite("checkpoint_rag_document_chunks")
    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "chunk_tokens", "min_value": MIN_CHUNK_TOKENS, "max_value": MAX_CHUNK_TOKENS},
        ),
        *[
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column},
            )
            for column in ("source_path", "chunk_index", "doc_type", "chunk_text")
        ],
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "chunk_index", "min_value": 0},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "has_embedding", "value_set": [True]},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "embedding_dimensions", "min_value": EMBEDDING_DIMENSIONS, "max_value": EMBEDDING_DIMENSIONS},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "contains_pii", "value_set": [False]},
        ),
    ]
    for expectation in expectations:
        suite.add_expectation(expectation, send_usage_event=False)
    return suite


def normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    normalized = []
    for record in records:
        metadata = record.get("metadata") or record
        raw_text = metadata.get("chunk_text")
        text = str(raw_text) if raw_text is not None else ""
        embedding = record.get("values")
        if embedding is None:
            embedding = record.get("embedding")
        normalized.append({
            "chunk_text": text,
            "contains_pii": contains_pii(text),
            "chunk_tokens": len(text) // 4,
            "source_path": metadata.get("source_path"),
            "chunk_index": metadata.get("chunk_index"),
            "doc_type": metadata.get("doc_type"),
            "has_embedding": embedding is not None,
            "embedding_dimensions": len(embedding) if embedding is not None else None,
        })
    return pd.DataFrame(normalized, columns=[
        "chunk_text", "contains_pii", "chunk_tokens", "source_path", "chunk_index", "doc_type",
        "has_embedding", "embedding_dimensions",
    ])


def validate_records(records: list[dict[str, Any]]):
    """Run the suite on pre-upsert Pinecone vector records and return GE results."""
    frame = normalize_records(records)
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[Batch(data=frame)],
        expectation_suite=build_expectation_suite(),
    )
    return validator.validate(result_format="SUMMARY")