"""Create or refresh the LangSmith evaluation dataset from the local test set."""

import json
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_SET = ROOT / "evals" / "test_set.json"
DEFAULT_DATASET_NAME = "checkpoint-rag-csm-evals"


def load_examples(test_set_path: str | Path = DEFAULT_TEST_SET) -> list[dict[str, Any]]:
    """Map the local query set to LangSmith examples with stable IDs."""
    with Path(test_set_path).open(encoding="utf-8") as file:
        payload = json.load(file)

    examples = []
    for item in payload.get("queries", []):
        query_id = item["id"]
        examples.append({
            "example_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"checkpoint-rag:{query_id}")),
            "inputs": {"query": item["query"], "query_id": query_id},
            "outputs": {
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "expected_sources": item.get("expected_sources", []),
            },
            "metadata": {
                "query_id": query_id,
                "category": item.get("category", "general"),
                "difficulty": item.get("difficulty", "unknown"),
            },
        })
    return examples


def setup_dataset(
    client: Client | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    test_set_path: str | Path = DEFAULT_TEST_SET,
):
    """Create the named dataset if needed and upsert its stable examples."""
    load_dotenv(ROOT / ".env")
    client = client or Client()
    datasets = client.list_datasets(dataset_name=dataset_name, limit=1)
    dataset = next(iter(datasets), None)
    if dataset is None:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="CSM RAG questions with reference answers and expected source documents.",
        )

    existing_ids = {
        str(existing.id)
        for existing in client.list_examples(dataset_id=dataset.id)
    }
    for example in load_examples(test_set_path):
        example_id = example["example_id"]
        if example_id in existing_ids:
            client.update_example(
                example_id,
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
                dataset_id=dataset.id,
            )
        else:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
                dataset_id=dataset.id,
                example_id=example_id,
            )
    return dataset


if __name__ == "__main__":
    created_dataset = setup_dataset()
    print(f"Dataset ready: {created_dataset.name} ({created_dataset.id})")