# Exam Answers

## Q1 (3 pts)
MLOps is about model lifecycle, while LLMOps is about the full AI app lifecycle. They overlap on testing, deployment, and monitoring, but LLMOps also covers prompts, retreival, chunking, embeddings, and evals. In my build, the corpus + retrieval flow is clearly LLMOps, while LangSmith/Phoenix/GE are more ops/control layer stuff. The biggest difference is that LLM apps are systems, not just models.

## Q2 (3 pts)
The lifecycle starts with defining the problem, then getting the right data, then choosing the architecture, then evaluating, then hardening, then deploying. A senior engineer asks whether this is really a retrieval problem, a workflow problem, or a classification problem before jumping to a model. A junior engineer usually just says “let’s use GPT” and misses the real failure mode.

## Q3 (2 pts)
A healthcare triage app needs full MLOps maturity because bad outputs can cause real harm and need governance, audit logs, and rollback. An internal support RAG can be lighter because the blast radius is smaller and humans can review mistakes. The deciding factor is risk, sensitivity, and how much the system can impact users or business decisions.

## Q4 (3 pts)
Eval-driven development means building around quality metrics from day one, not just shipping code and hoping it works. Unlike TDD, LLM evals are noisy, costly, and sometimes non-deterministic, so you need a benchmark set, human review, and layered checks. You treat a single score as a signal, not as truth, and look at regression trends instead.

## Q5 (3 pts)
LLM-as-Judge is good for relevance, helpfulness, and groundedness when a rubric is clear and human review at scale is too expensive. It is bad for exact factual checking or high-risk decisions because the judge can be wrong or too forgiving. To trust it, I would build a human-labeled calibration set, compare ratings, fix the rubric, and only use it after the judge matches human judgement closely enough, or at least close enough for a real-world relevent check.

## Q6 (3 pts)
Algorithmic evaluators are best for objective checks like chunk size, PII detection, and embedding dimensions. Human evaluators are best for subjective quality like helpfulness or policy fit, but they are slow and pricey. LLM-as-Judge sits in the middle: useful for rubric scoring at scale, but it needs calibration and should not be the only source of truth.

## Q7 (3 pts)
Agent evals are harder because agents can call the wrong tool, forget earlier constraints, or loop forever. In Arize, I would trace the full run and inspect the tool calls, arguments, and state transitions across steps. That catches tool misuse, state drift, and repetitive loops that a single-shot RAG would never show.

## Q8 (3 pts)
Tracing shows the full request flow, logging shows a single event, and metrics show aggregate health over time. In my build, a Phoenix trace could show retrieved docs, latency, and token use, while a log might capture a failed tool call. Each answers a different question: why did this answer go wrong, what failed, and is the system getting worse overall.

## Q9 (4 pts)
Four real production issues you miss in a passing eval run are latency spikes, source duplication, prompt bloat, and upstream dependency failures. You catch them with observability: p95 latency crossing a threshold, duplicate-source rate rising, prompt tokens jumping, or Pinecone/OpenAI errors. The alert action is to inspect the traces, reduce retrieval noise, or roll back before users feel the impact.

## Q10 (3 pts)
Data drift is when the input data changes, concept drift is when the meaning of a topic changes, and prompt drift is when the system prompt changes behavior. In a CSM RAG, data drift could be more ticket data about a new feature, concept drift could be “SLA” meaning something different, and prompt drift could happen after a prompt tweak that hurts grounding. Monitoring signals are shifts in retrieval patterns, answer quality, or confidence.

## Q11 (4 pts)
The key data quality dimensions are completeness, accuracy, consistency, timeliness, validity, and uniqueness. The corpus cares most about completeness, validity, timeliness, and uniqueness; retrieval results care most about accuracy and consistency; generation outputs care most about accuracy and validity. Example GE checks are chunk size between 50 and 300, source path non-null, and embedding dimensions exactly 1536. These catch broken chunks, missing metadata, and invalid vectors before Pinecone gets them.

## Q12 (3 pts)
To catch PII, I would use the existing `contains_pii` check and fail the batch if a chunk matches emails, phone numbers, or tokens. To stop it reaching generation, I would filter those chunks before prompt assembly and quarantine them. To prevent re-entry, I would block the source path or file hash on future ingestions so the same bad doc never gets re-indexed again.

## Q13 (3 pts)
Every ingestion should run the safety gate: chunk size, null checks, vector dimensions, and PII detection. Weekly, I would look for drift in corpus coverage, duplicate docs, stale content, and retrieval quality. Before prod promotion, I would run the benchmark set and a canary deploy, and only ship if answer quality, source hit rate, and latency stay acceptable. That’s the difference between “the data is valid” and “the system actually works.”
