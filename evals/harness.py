import os
import json
import time
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
import sys
import types

# Polyfill legacy VertexAI module path to prevent import crash
dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex

# Safe to import ragas now
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pipeline import run_pipeline

load_dotenv()


def run_eval_harness(
    test_set_path: str = "evals/test_set.json",
    output_dir: str = "results",
    documents: list = None
):
    """
    Executes the 50-query test set through pipeline.py, evaluates results
    using Ragas metrics, and outputs JSON and Markdown reports.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load test set
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    queries = test_data.get("queries", [])
    print(f"Loaded {len(queries)} evaluation queries from {test_set_path}...")

    # 2. Run queries through RAG pipeline
    eval_records = []
    pipeline_outputs = []

    for idx, item in enumerate(queries, 1):
        query_id = item["id"]
        query_text = item["query"]
        ground_truth = item.get("ground_truth_answer", "")
        expected_sources = item.get("expected_sources", [])

        print(f"[{idx}/{len(queries)}] Running pipeline for Query {query_id}...")
        start_time = time.time()

        try:
            rag_response, retrieved_docs = run_pipeline(
                query=query_text, documents=documents, top_k=4
            )
            latency = round(time.time() - start_time, 2)
            generated_answer = rag_response.answer
            response_sources = rag_response.sources
            confidence = rag_response.confidence
        except Exception as e:
            latency = round(time.time() - start_time, 2)
            generated_answer = f"ERROR: Pipeline execution failed: {str(e)}"
            response_sources = []
            confidence = "low"
            retrieved_docs = []

        # Extract text contexts and sources
        context_texts = [doc.page_content for doc in retrieved_docs]
        retrieved_sources = [
            doc.metadata.get("source_path") or doc.metadata.get("doc_id", "unknown")
            for doc in retrieved_docs
        ]

        # Dataset format required by Ragas
        eval_records.append({
            "question": query_text,
            "answer": generated_answer,
            "contexts": context_texts if context_texts else ["No context retrieved."],
            "ground_truth": ground_truth,
        })

        # Detailed execution log
        pipeline_outputs.append({
            "id": query_id,
            "query": query_text,
            "category": item.get("category", "general"),
            "difficulty": item.get("difficulty", "unknown"),
            "ground_truth": ground_truth,
            "expected_sources": expected_sources,
            "generated_answer": generated_answer,
            "retrieved_sources": retrieved_sources,
            "response_sources": response_sources,
            "confidence": confidence,
            "latency_sec": latency,
            "notes": item.get("notes", ""),
        })

    # 3. Create HuggingFace Dataset for Ragas
    ragas_dataset = Dataset.from_list(eval_records)

    # 4. Execute Ragas Evaluation
    print("\nRunning Ragas evaluation metrics...")
    
    eval_llm = ChatOpenAI(model=os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini"), temperature=0)
    eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    ragas_results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    df_results = ragas_results.to_pandas()

    # 5. Aggregate Results
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_precision = 0.0

    for i, output in enumerate(pipeline_outputs):
        row = df_results.iloc[i]
        
        f_score = float(row.get("faithfulness", 0.0))
        r_score = float(row.get("answer_relevancy", 0.0))
        p_score = float(row.get("context_precision", 0.0))

        output["metrics"] = {
            "faithfulness": round(f_score, 4),
            "answer_relevance": round(r_score, 4),
            "context_precision": round(p_score, 4),
        }

        total_faithfulness += f_score
        total_relevance += r_score
        total_precision += p_score

    n = max(len(pipeline_outputs), 1)
    avg_faithfulness = round(total_faithfulness / n, 4)
    avg_relevance = round(total_relevance / n, 4)
    avg_precision = round(total_precision / n, 4)

    summary_data = {
        "total_queries": len(pipeline_outputs),
        "overall_metrics": {
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_relevance,
            "context_precision": avg_precision,
        },
        "query_results": pipeline_outputs,
    }

    # 6. Save JSON Output
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved machine-readable JSON: {json_path}")

    # 7. Save Markdown Report
    md_path = os.path.join(output_dir, "eval_report.md")
    md_content = f"""# Helix RAG Evaluation Report (Ragas Engine)

**Total Queries Evaluated:** {len(pipeline_outputs)}  
**Evaluation LLM:** `{os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")}`

---

## 📊 Summary Metrics

| Metric | Score | Target | Status |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | `{avg_faithfulness * 100:.1f}%` | ≥ 90.0% | {'✅ Pass' if avg_faithfulness >= 0.9 else '⚠️ Review'} |
| **Answer Relevancy** | `{avg_relevance * 100:.1f}%` | ≥ 85.0% | {'✅ Pass' if avg_relevance >= 0.85 else '⚠️ Review'} |
| **Context Precision** | `{avg_precision * 100:.1f}%` | ≥ 80.0% | {'✅ Pass' if avg_precision >= 0.80 else '⚠️ Review'} |

---

## 📋 Query-Level Results

| Query ID | Category | Difficulty | Precision | Faithfulness | Relevancy | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in pipeline_outputs:
        m = r["metrics"]
        md_content += f"| `{r['id']}` | {r['category']} | {r['difficulty']} | {m['context_precision'] * 100:.0f}% | {m['faithfulness'] * 100:.0f}% | {m['answer_relevance'] * 100:.0f}% | {r['latency_sec']}s |\n"

    md_content += "\n---\n\n## 🔍 Sample Detailed Breakdowns\n\n"
    for r in pipeline_outputs[:5]:
        md_content += f"""### Query `{r['id']}`: {r['query']}
- **Ground Truth:** {r['ground_truth']}
- **Generated Answer:** {r['generated_answer']}
- **Expected Sources:** `{r['expected_sources']}`
- **Retrieved Sources:** `{r['retrieved_sources']}`
- **Confidence:** `{r['confidence']}`
- **Notes:** *{r['notes']}*

---
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved human-readable report: {md_path}")


if __name__ == "__main__":
    run_eval_harness()