"""Great Expectations suite for RAG document chunks and embedding vectors."""

import re
from typing import Any

import pandas as pd
from great_expectations.core.batch import Batch
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations.expectation import ExpectationConfiguration
from great_expectations.validator.validator import Validator

TARGET_CHUNK_TOKENS = 100
MAX_CHUNK_TOKENS = 125
EMBEDDING_DIMENSIONS = 1536
PII_REGEX = re.compile(
    r"(?s).*(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|"
    r"\b\d{3}-\d{2}-\d{4}\b|"
    r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b).*",
    re.IGNORECASE,
)


def build_expectation_suite() -> ExpectationSuite:
    suite = ExpectationSuite("checkpoint_rag_document_chunks")
    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "chunk_tokens", "min_value": 1, "max_value": MAX_CHUNK_TOKENS},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "source_path"},
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
        text = str(record.get("chunk_text", ""))
        embedding = record.get("embedding")
        normalized.append({
            "chunk_text": text,
            "contains_pii": bool(PII_REGEX.search(text)),
            "chunk_tokens": (len(text) + 3) // 4,
            "source_path": record.get("source_path"),
            "has_embedding": embedding is not None,
            "embedding_dimensions": len(embedding) if embedding is not None else None,
        })
    return pd.DataFrame(normalized, columns=[
        "chunk_text", "contains_pii", "chunk_tokens", "source_path", "has_embedding", "embedding_dimensions",
    ])


def validate_records(records: list[dict[str, Any]]):
    """Run the suite on chunk records and return the GE validation result."""
    frame = normalize_records(records)
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[Batch(data=frame)],
        expectation_suite=build_expectation_suite(),
    )
    return validator.validate(result_format="SUMMARY")