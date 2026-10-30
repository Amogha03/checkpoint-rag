# RAG Operations Monitoring Specification

## Scope and Ownership

This specification covers the customer-support RAG request path, offline quality evaluations, chunk ingestion gates, and observability backends. The thresholds below are initial release/SRE targets, not claims of measured production SLOs. Establish service baselines and assign an on-call owner before production traffic is enabled. Quality alerts are release gates or investigation alerts; they should not page on a single LLM-judge score.

## Service-Level Objectives and Alerts

| Area | Signal and source | Target | Alert/action |
|---|---|---|---|
| Availability | Successful RAG requests / all requests, from application outcome logs | At least 99.5% monthly availability | Page if failures exceed 1% over 5 minutes with at least 20 requests; investigate dependency-specific errors immediately. |
| Latency | End-to-end request duration, Phoenix parent spans | p95 <= 5 s; p99 <= 10 s | Warn when p95 exceeds 5 s for 10 minutes; page when p95 exceeds 10 s for 5 minutes. Exclude batch evaluation traffic from user-facing SLO calculations. |
| Error rate | Failed requests / all requests, classified by retrieval, generation, timeout, and dependency | < 1% over 5 minutes | Page at >= 2% for 5 minutes; alert on any sustained authentication/configuration failure. |
| Faithfulness | LangSmith `faithfulness` score, rolling evaluation window | Mean >= 0.90 | Block release if a representative 50-query run is below 0.90; investigate a rolling mean below 0.90 across two runs. |
| Factual accuracy | LangSmith `csm_support_accuracy` score | Mean >= 0.90 | Block release below 0.90; manually review any high-impact answer scoring below 0.70 and calibrate judge results against human review. |
| Answer relevance | LangSmith `answer_relevance` score | Mean >= 0.85 | Block release below 0.85; investigate repeated category-specific declines. |
| Citation/source quality | Expected source hit rate and citation correctness, stratified by evaluation category | Establish a baseline; initial release target >= 0.90 expected-source hit rate | Hold release if source coverage falls more than 5 percentage points from the approved baseline or citation correctness is below 0.90. |
| Ingestion quality | Great Expectations suite result before Pinecone mutation/upsert | 100% expectations pass | Hard fail the ingestion batch on any failed expectation. Preserve the report and do not proceed to index mutation or upsert. |
| Token/cost | Prompt and completion tokens per query, plus estimated USD/query from current model price configuration | Establish per-query budget by workload and model | Warn when rolling p95 token use exceeds its approved baseline by 20% for 30 minutes; page/disable an unexpectedly unbounded run if cost exceeds the configured hard budget. |

For cost accounting, store prompt and completion tokens separately and calculate estimated cost using the deployed model’s current, centrally configured input/output token rates. Do not hard-code vendor prices into this specification; update the configuration when model or pricing changes. Track mean, p50, p95, and total cost by model, route, and evaluation/production traffic class.

## Quality Evaluation Cadence

- Run the complete, versioned evaluation set before a release and after changes to chunking, retrieval, prompts, model, or source corpus.
- Preserve the dataset version, experiment ID, model, prompt version, evaluator version, aggregate scores, and per-query results.
- Review failure slices by category and difficulty; report judge confidence/limitations and confirm a sample of failures with human reviewers.
- Treat the 50-query LangSmith result as a quality gate, not a statistically sufficient estimate of every production request.
- Compare retrieved `source_path`/normalized `doc_id` with `expected_sources`. Deduplicate repeated source/chunk IDs before deciding that top-k coverage is adequate.

## Ingestion Gate

`src/ingest.py` prepares Pinecone-shaped records with `id`, `values`, and `metadata`. The Great Expectations adapter checks approximate 50–300-token chunk bounds, non-null `source_path`, `chunk_index`, `doc_type`, and `chunk_text`, nonnegative chunk indexes, presence and 1536-dimensional length of embeddings, and PII/credential patterns. Validation runs before the index is recreated or vectors are upserted. The CLI validates a local sample using zero-valued vectors for shape only; production ingestion validates the actual embedding vectors.

The token estimate is the ingestion pipeline's approximate four-characters-per-token calculation. Regex screening catches common email, phone, SSN, and credential formats but is not a guarantee that all sensitive data is absent. Role-based mailbox names are treated as shared contacts; review that exception for the corpus and organization's privacy policy.

## Incident Triage

1. Capture UTC time window, query/run/trace ID, dataset and code revision, model, and error category.
2. For quality regressions, inspect the expected source, retrieved source IDs and chunk text, generated answer, citations, confidence, and evaluator rationale together.
3. For latency/cost regressions, compare retrieval and generation child-span duration, token counts, top-k size, retries, and provider errors; separate evaluation traffic from production.
4. For ingestion failures, inspect each failed expectation and source path. Stop the batch before Pinecone changes; fix or quarantine the offending document, then revalidate.
5. Record mitigation, root cause, customer impact, and a regression test or evaluation example.

## Commands and Artifacts

- Dataset refresh: `python -m ops.langsmith.setup`
- LangSmith evaluation: `python -m ops.langsmith.run_evals`
- Phoenix server: `phoenix serve`; sample batch: `python -m ops.phoenix.instrument --connect-existing`
- Data-quality validation: `python -m ops.data_quality.run_validations`
- Offline tests: `python -m pytest tests/`
- LangSmith scores: `ops/langsmith/results/eval_scores.json`
- Phoenix sample summary: `ops/phoenix/results/sample_traces.json`; live traces are stored in the Phoenix project `checkpoint-rag`.
- GE results: `ops/data_quality/results/validation_results.json` and `ops/data_quality/results/validation_report.md`.

## Security and Retention

Queries, prompts, retrieved documents, answers, source IDs, and spans may contain internal or personal information. Restrict access to LangSmith/Phoenix projects and local storage, use approved retention/deletion settings, and do not export production traces to unapproved workspaces. Keep credentials in a secret manager and never include them in datasets, evaluator comments, logs, or reports. The PII regex is only a screening control.