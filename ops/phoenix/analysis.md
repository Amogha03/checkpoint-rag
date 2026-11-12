# Phoenix Trace Analysis

## Setup and Run

Install project dependencies, configure `OPENAI_API_KEY` and `PINECONE_API_KEY`, and ensure the Pinecone index has been ingested. To keep Phoenix available after a query batch, start the standalone server in one terminal with `phoenix serve`, then run `python -m ops.phoenix.instrument --connect-existing` in another. This instruments LangChain, exports spans to the running local collector, and runs the first ten queries from `evals/test_set.json`. Alternatively, `python -m ops.phoenix.instrument` launches Phoenix itself and stays running until Ctrl+C. Phoenix is not started by normal application imports.

The local Phoenix session URL is printed by Phoenix at startup. A machine-readable run summary is written to `ops/phoenix/results/sample_traces.json`.

## Captured Signals

- Phoenix spans capture instrumented LangChain calls and a `checkpoint_rag.query` parent span for each test query.
- The parent span annotates `rag.latency_ms`, prompt/completion/total token counts, `retrieval.document_ids`, `retrieval.source_paths`, and retrieved-document count.
- The local run summary records end-to-end latency, LangChain callback token totals, retrieved document IDs, response sources, and confidence for each query.
- Token totals depend on supported LangChain model callbacks and may be zero for models that do not report usage through that callback.
- Raw Pinecone chunks store `source_path`; `retrieve()` exposes that value as `doc_id` on LangChain `Document` objects. The span records both labels, using the normalized document ID as a fallback when raw `source_path` is not present on the returned object.

Compare latency by query category and inspect long spans for retrieval/model bottlenecks. Compare retrieved IDs and response sources with `expected_sources` in the evaluation set when diagnosing retrieval misses. Tracing is diagnostic; it does not itself establish answer correctness.

## Observed Retrieval Weakness

In the saved ten-query Phoenix sample, Q001 ("What's the maximum number of steps allowed in a single workflow?") retrieved four chunks, but only `product-docs/workflows/creating-workflows.md` matched the expected source; the other three came from loops, a downgrade ticket, and email notifications. Q002 also returned `tickets/ticket-1234-workflow-timeout.html` twice. This is retrieval noise and duplicate-result behavior: irrelevant context consumes prompt tokens and may distract generation. Follow-up: inspect ensemble deduplication and ranking, then compare expected-source hit rate and duplicate count in a new Phoenix run. Q001's measured end-to-end latency was 5.411 seconds in that sample; inspect its retrieval and LLM child spans before attributing the delay to a specific stage.

## Data Handling

Traces may contain employee queries, prompts, retrieved internal documentation, and generated answers. Restrict access to the local Phoenix process and any configured trace storage, apply the organization's retention policy, and do not export sensitive production traces to an unapproved environment. PII checks in the data-quality suite are screening controls, not a substitute for access control or redaction.