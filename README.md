# Checkpoint RAG

This project demonstrates a retrieval-augmented generation (RAG) workflow using LangChain, OpenAI, and Pinecone.


## Prerequisites

- Python 3.10 or newer
- An OpenAI API key
- A Pinecone API key
- A Pinecone index available in your account

## 1) Clone and enter the repository

```bash
git clone <your-repository-url>
cd checkpoint-rag
```

## 2) Create a virtual environment

```bash
# If needed, install Python 3.10 first
# brew install python@3.10

python3.10 -m venv .venv
source .venv/bin/activate
```

If `python3.10` is not available, install it first and then rerun the commands above.

## 3) Install project dependencies

```bash
python -V
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name checkpoint-rag
```

This installs the project dependencies declared in [pyproject.toml](pyproject.toml), including `python-dotenv` and `langchain-pinecone`.

## 4) Configure environment variables

Copy the sample environment file and fill in your real credentials:

```bash
cp .env.example .env
```

Then update `.env` with your values:

```bash
OPENAI_API_KEY="your_openai_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
```

## Chunking Strategy

This project uses semantic chunking with 20% token overlap to balance retrieval quality and cost efficiency.

### Chunk Size: 512 Tokens (~2 KB)

- **Why 512?** OpenAI recommends 256–512 tokens for search/retrieval tasks. This size provides:
  - ✅ Enough context for semantic meaning (vs. small chunks that lose coherence)
  - ✅ Focused scope (vs. large chunks that mix multiple topics)
  - ✅ Cost-efficient embeddings (~150 vectors for 100 documents vs. 400+ with smaller chunks)

### Overlap: 20% (~102 Tokens)

- **Why 20%?** 
  - ✅ Captures cross-boundary context (queries spanning chunk boundaries still retrieve relevant chunks)
  - ✅ Minimal redundancy cost (~1–2% additional embeddings vs. no overlap)
  - ✅ Standard for RAG systems requiring multi-hop reasoning
  - For a 512-token chunk, the last 102 tokens of chunk N = first 102 tokens of chunk N+1

### Document-Type Handling

| Document Type | Loader | Splitting Strategy | Why |
|---|---|---|---|
| **Markdown** (60 files) | `UnstructuredMarkdownLoader` | Split by headers (`#`, `##`, `###`), then recursively by `\n\n` if sections exceed 512 tokens | Markdown headers are natural semantic boundaries; preserves document hierarchy |
| **PDF (text-native)** (20 files) | `PyPDFLoader` | `RecursiveCharacterTextSplitter` with separators: `["\n\n", "\n", ". ", " "]` | Respects paragraph breaks and sentence structure; text-native PDFs have clean structure |
| **PDF (scanned)** (5 files) | `PyPDFLoader(extract_images=True)` | Same as text-native (OCR extracts text first) | `extract_images=True` triggers local OCR engine to convert scanned images to text |
| **HTML** (15 files) | `UnstructuredHTMLLoader` | `RecursiveCharacterTextSplitter` with separators: `["</p>", "</div>", "\n\n", "\n", ". ", " "]` | Respects HTML block structure; prioritizes `<p>` and `<div>` boundaries |

### Vector Metadata

Each chunk is stored with minimal metadata for efficiency:
```json
{
  "source_path": "product-docs/workflows/creating-workflows.md",
  "doc_type": "markdown",
  "chunk_index": 0,
  "chunk_text": "The embedded text content..."
}
```

### Ingestion Pipeline

Run the ingestion pipeline to load, chunk, embed, and upsert all documents:

```python
from src.ingest import ingest_corpus

stats = ingest_corpus("../corpus/")
# Output: files_processed, total_chunks, vectors_upserted
```

Expected output for 100 documents:
- ~120–180 total chunks (depending on document size variance)
- All vectors stored in Pinecone with 1536-dimensional embeddings
- Fits comfortably in Pinecone free tier (100K vector limit)