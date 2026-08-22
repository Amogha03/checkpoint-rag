# Section A: Foundations (6 points)
## Q1 (3 pts)
Explain the relationship between context window, tokens, and cost. Why does chunking strategy
materially impact both retrieval quality AND inference cost? Use specific numbers from a model you
have actually used.

## A1

The context window is the maximum number of tokens that a model can take in at a time. 
LLM APIs charge per token basis, larger inputs directly inflate inference costs.
Chunking strategy materially impacts both retrieval quality and cost because large chunks dilute specific semantic details, leading to poor embedding matches, whereas smaller chunks provide precise context that improves retrieval while keeping input prompt payloads small. For example, in my Checkpoint project I used gpt-4o-mini. I started out with 512-token chunks which gave a generalized retrieval and Context Precision below 60% and similarity scores was below 0.40. I changed my strategy to 100-token chunks with a 20% overlap which improved overall retrieval performance by +27%. Assuming the same top_k = 4 for inference, 512-token chunks send 4 * 512 = 2,048 context tokens and 100-token chunks send 4 * 100 = 400 context tokens. This is approximately 80% fewer context tokens to the LLM which reduces inference cost by 5 times.

---

## Q2 (3 pts)
Compare temperature, top_p, and top_k. When would you use each, and why might you set
temperature=0 in a production RAG system? When would temperature=0 be the wrong call?

## A2
Tempertaure is control on determinism of the model (randomness). It usually ranges from 0-1, with 0 being most deterministic. 
top_p is how diverse we want the vocabulary of the output be. It usually ranges from 0-1 and lower top_p value makes the model to have more focused, conventional vocabulary.
top_k restricts the model to choose from only the "k" most likely next words or tokens at each step of text generation making the model relwvant.
we set temperature=0 when we need deterministic output. Production RAG systems usually need a reproducible output, hence we set them to 0. If we want creative outputs, we set temperature closer to 1. For example, when converting a paragraph into a poem, we adjust the temperature higher or lower depending on how creative we want the answers to be.

---

## Q3 (4 pts)
You're building an internal HR document Q&A system for a 5,000-person healthcare company.
Compare OpenAI GPT-(recent), Claude Sonnet, and self-hosted Llama 3 across (a) cost at scale, (b)
data privacy and compliance, (c) latency, (d) accuracy on long-context retrieval. Pick one. Defend the
choice in 4–6 sentences.

## A3
OpenAI GPT-(recent) and Claude Sonnet typically have the strongest accuracy and easiest scaling, but their per-token costs and enterprise access with privacy, compliance, audit, and data-residency incurs more costs than standard API usage. Claude Sonnet is very strong for long-context document reasoning, while GPT offers broad tooling and low-latency API access. Self-hosted Llama 3 has higher upfront infrastructure and operational costs, but its marginal cost can be lower at high volume, latency can be optimized locally, and data remains within the company’s controlled environment. I would choose self-hosted Llama 3 because healthcare HR documents require strict privacy, compliance, auditability, and control over data. I would validate its long-context retrieval accuracy with an internal benchmark and use a managed model only as an isolated fallback for queries where Llama fails.

---

## Q4 (2 pts)
Give two specific scenarios where fine-tuning Llama 3 is the correct choice over prompt engineering
with GPT-(recent). What is the use case, what does the training data look like, and what does success
measurement look like?

## A4

1. A Bank Reviewing Loan Applications

A bank wants an AI tool to check loan applications against its private internal rules to flag risks before human approval.
In this case, we can't use ChatGPT because sending customer financial data over the internet breaks privacy laws and prompting GPT with huge rulebooks every time can get expensive. By taking Llama 3 and fine-tuning it on their own secure servers, the AI learns the bank's exact rulebook without any sensitive data ever leaving the building. We can use RAGA metrics like answer relevance and accuracy, to mesure success.

2. A Hospital Processing Patient Medical Records

As a patient's medical records are higly sensitive and confidential we want to use fne-tuning Llama 3 to extract personally identifiable information. Training data would be thousands of unstructured medical notes, lab test results etc. context precision and accuracy can be used for success measurements.

---

## Q5 (3 pts)
You have 50,000 PDFs of insurance policies, each 30–100 pages, with tables, headers, footers, and
inline citations. Walk through your chunking strategy: chunk size, overlap, how you handle tables, what
metadata you attach. Justify each decision.

## A5
Step by step chunking strategy:
-Clean the data: Strip out repetitive headers, page numbers, and legal footers. Convert tables into clean Markdown or HTML structures so rows and columns stay linked.
-Chunk Size (512–1,024 Tokens): Keep chunks at approx 400 to 800 words. This size is big enough to capture full legal rules without cutting off mid-sentence, but small enough to fit nicely inside modern embedding models.
-Overlap (20%): A 20% overlap ensures clauses remain fully intact without losing surrounding context across boundaries.
-Metadata: Insert metadata like policy number, policy type, effective date, page number, citations etc to every chunk. This lets you run quick filters (like filtering out expired 2022 policies) before running the vector search.

---

## Q6 (2 pts)
Explain hybrid search (BM25 + semantic). When does it meaningfully outperform pure vector search?
Give a concrete example with the kind of query that would fail under pure vector retrieval but succeed
under hybrid.

## A6
BM25 helps with exact keyword matches and vector search helps with semantic similarity. Hybrid search gives us best of both worlds. Hybrid search meaningfully outperforms pure vector search when queries contain exact terms such as error codes, technical jargon etc. 
Example: "What's the average price per line under T-mobile Magenta Plan?"
Here a pure vector search might miss the relevant documentation if it uses different terminology, such as “rate per connection.” Hybrid search combines semantic similarity, which connects those related concepts, with BM25 keyword matching for exact terms such as “T-Mobile” and “Magenta Plan,” making it more likely to retrieve the correct pricing document

---

## Q7 (2 pts)
What is reranking, and why do production RAG systems often add a reranker even when retrieval looks
'good enough' on top-5 metrics? Name a specific reranker you would use and explain when you would
reach for it.

## A7
Reranking in RAG is basically a second-pass filter that improves your vector search results. If a vector search first grabs 20–100 candidate chunks, and then a cross-encoder model evaluates the query and each individual text chunk together side-by-side, scoring them on precise context and actual relevance and putting most accurate answers at the top 3-5.

production RAG systems often add a reranker because:

-LLMs pay attention primarily to the start and end of prompt contexts and reranking guarantees the most relevant chunk lands at position #1.
-Allows filtering 50 broad candidates down to 2–3 high precision chunks, saving token costs and reducing LLM latency.

I would reach for Cohere Rerank when building high stakes Enterprise RAG applications (like policy or financial doc search) where fine print precision and structured metadata awareness are mandatory.

---

## Q8 (3 pts)
Compare vector RAG vs Graph RAG. Describe a use case where Graph RAG meaningfully
outperforms vector RAG and explain the underlying reason. Not just 'it has relationships' but specifically
what kind of query benefits from a traversal that embeddings cannot answer.

## A8
Vectore RAG use semantic similarity to get relevant context. They captures semantic meaning but struggle to capture themes and relationships between entities in the document corpus. Graphs are great at representing and storing diverse and interconnected information in a structured manner. we have nodes and relationships between those nodes.

Usecase: When a long rulebook is cut into separate chunks, Vector RAG fails because it can only look for single chunks that match user query, missing rules that rely on each other across different pages.
query : "Can I get the free iPhone trade-in deal on the Magenta Plan?"
Vector search misses this because Page 1 says "Magenta includes Tier-3," while Page 20 says "Tier-3 includes the iPhone deal" and since neither chunk contains all the keywords, vector search fails. Graph RAG simply follows the connection Magenta -> Tier-3 -> iPhone deal across both pages to give the right answer.

---

## Q9 (3 pts)
Define faithfulness, answer relevance, and context precision in RAG evaluation. For each, give one
concrete failure mode you have seen (or could see) in your build from Part 1. How would you collect a
ground-truth set if you didn't have one provided?

## A9
Faithfulness checks if the model stuck to the provided notes without hallucinating, answer relevance checks if it actually answered the user's question instead of dodging it, and context precision checks whether the retriever retrieved useful context and ranked them at the top.

In my evaluation test set, I saw all three failure modes:

Faithfulness failure (Q046): The model hallucinated extra details beyond the retrieved context, dropping its faithfulness score to 0.5.

Answer relevance failure (Q042): The model's answer was only partially correct because it failed to address the multi-step diagnosis explicitly outlined in the ground truth.

Context precision failure (Q017): The retriever fetched and used oauth-flow.md as context instead of the correct document, api/errors.md.

To make a ground-truth set from scratch, I would use LLM pipeline from ragas to generate question and answer pairs from the source (corpus provided) by trying to extract key points from the document chuncks. I would manually spot check 15% of the generated set.

---

## Q10 (3 pts)
List 3 production failure modes for a deployed RAG system and how you would detect each in
monitoring. Be specific about the signal, not 'monitor accuracy' but what metric, what threshold, what
alert.

## A10
Retrieval Drift:
-Failure Mode:The vector database embeddings are out-of-sync with source data updates, meaning the system retrieves outdated or non-existent chunks.
-Signal: Max cosine similarity scores per query.
-Alert: Setup alerts to check max similarity scores which fall below 0.65 in 1 hour window

Hallucination:
-Failure Mode: The LLM generates responses containing facts, numbers, or rules that do not exist within the chunks provided by the retriever.
-Signal: check faithfullness scores by a smaller LLM-as-a-judge model on a 5-10% sample of production logs.
-Alert: Setup alerts to check if the rolling 1-hour average faithfulness score drops below 0.85.

Context Bloat & Latency Degradation:
-Failure Mode: Retrieval pulls excessively large chunks, pushing the prompt token payload beyond optimal limits, which causes generation timeouts, high latency, or truncated instructions.
-Signal: Total Prompt Token Volume + p99 Time-to-First-Token (TTFT).
-Alert: Setup alerts to check if prompt tokens exceed certain threshold tokens per request or if p99 TTFT exceeds a 5-minute window.

---

## Q11 (2 pts)
Your RAG chatbot is hallucinating on questions where the answer genuinely is not in the corpus. Walk
through your debugging process step by step. What do you check first, second, third? What is the fix
when the issue is the model 'helping' instead of admitting uncertainty?

## A11
I would first check the retrieved chunks and their similarity scores to see if the vector database is pulling correct documents, second I will review the system prompt to check for explicit refusal rules, and third I will verify if the model's temperature is set to zero.
When the issue is the model trying to "help" using its outside knowledge, the fix is to enforce a strict system prompt that instructs the model to answer ONLY using the provided context or state "Sorry I don't know". Implement a strict similarity score threshold in the vector search and also include negative few-shot examples showing the model how to correctly decline to answer.

---
