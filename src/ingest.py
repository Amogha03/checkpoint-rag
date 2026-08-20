import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import itertools
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Load environment variables from .env file
load_dotenv()

# Read configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "checkpoint-rag-index"  # global index name
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def init_pinecone_index():
    """Creates the Pinecone index required by the RAG app."""
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY in environment.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    active_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME in active_indexes:
        pc.delete_index(INDEX_NAME)

    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )


def init_rag_components():
    """Initializes and returns the embeddings, vector store, and LLM instances."""
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment.")

    # Ensure the Pinecone index exists before connecting
    init_pinecone_index()

    # 1. Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    return embeddings

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    # Convert the iterable into an iterator
    it = iter(iterable)
    # Slice the iterator into chunks of size batch_size
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        # Yield the chunk
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


# ============================================================================
# DOCUMENT LOADING & CHUNKING PIPELINE
# ============================================================================


def load_documents(corpus_dir: str) -> List[Tuple[str, str, str]]:
    """
    Recursively loads documents from corpus directory.
    Returns list of (source_path, doc_type, text) tuples.
    Supports: .md, .pdf, .html files
    """
    corpus_path = Path(corpus_dir)
    documents = []

    if not corpus_path.exists():
        raise ValueError(f"Corpus directory not found: {corpus_dir}")

    # Load Markdown files
    for md_file in corpus_path.rglob("*.md"):
        try:
            loader = UnstructuredMarkdownLoader(str(md_file))
            docs = loader.load()
            if docs:
                relative_path = str(md_file.relative_to(corpus_path))
                text = docs[0].page_content
                documents.append((relative_path, "markdown", text))
                print(f"✓ Loaded: {relative_path}")
        except Exception as e:
            print(f"✗ Error loading {md_file}: {e}", file=sys.stderr)

    # Load PDF files (with OCR for scanned documents)
    for pdf_file in corpus_path.rglob("*.pdf"):
        try:
            # Enable OCR for all PDFs to handle both text-native and scanned
            loader = PyPDFLoader(str(pdf_file), extract_images=True)
            docs = loader.load()
            if docs:
                relative_path = str(pdf_file.relative_to(corpus_path))
                # Combine all pages into single text
                text = "\n".join([doc.page_content for doc in docs])
                documents.append((relative_path, "pdf", text))
                print(f"✓ Loaded: {relative_path} ({len(docs)} pages)")
        except Exception as e:
            print(f"✗ Error loading {pdf_file}: {e}", file=sys.stderr)

    # Load HTML files
    for html_file in corpus_path.rglob("*.html"):
        try:
            loader = UnstructuredHTMLLoader(str(html_file))
            docs = loader.load()
            if docs:
                relative_path = str(html_file.relative_to(corpus_path))
                text = docs[0].page_content
                documents.append((relative_path, "html", text))
                print(f"✓ Loaded: {relative_path}")
        except Exception as e:
            print(f"✗ Error loading {html_file}: {e}", file=sys.stderr)

    print(f"\n📚 Loaded {len(documents)} documents")
    return documents


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of tokens in text (1 token ≈ 4 chars for English).
    For text-embedding-3-small, this is a reasonable approximation.
    """
    return len(text) // 4


def chunk_document(
    text: str,
    doc_type: str,
    chunk_size: int = 512,
    overlap_percent: float = 0.2,
) -> List[str]:
    """
    Splits document into semantic chunks with token-based overlap.
    
    Args:
        text: Document text to chunk
        doc_type: Type of document ('markdown', 'pdf', 'html')
        chunk_size: Target chunk size in tokens (~4 chars per token)
        overlap_percent: Overlap as percentage of chunk size (e.g., 0.2 = 20%)
    
    Returns:
        List of chunk texts
    """
    # Convert token size to character size (rough: 1 token ≈ 4 chars)
    char_chunk_size = chunk_size * 4
    char_overlap = int(char_chunk_size * overlap_percent)

    if doc_type == "markdown":
        # Split by markdown headers, then recursively by separators
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False,
        )
        md_header_splits = markdown_splitter.split_text(text)

        # Further split if chunks are too large
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = []
        for doc in md_header_splits:
            splits = recursive_splitter.split_text(doc.page_content)
            chunks.extend(splits)
        return chunks

    elif doc_type == "pdf":
        # Split by recursive separators (PDF text structure)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    elif doc_type == "html":
        # Split by recursive separators (HTML structure)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_overlap,
            separators=["</p>", "</div>", "\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    else:
        # Fallback: generic recursive split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)


def create_chunk_vectors(
    chunks_list: List[str], source_path: str, doc_type: str, embeddings
) -> List[Dict]:
    """
    Creates Pinecone vectors from chunks.
    
    Returns:
        List of {id, values (embedding), metadata} dicts ready for upsert
    """
    vectors = []
    for idx, chunk_text in enumerate(chunks_list):
        # Create deterministic ID using hash of source_path + chunk index
        id_input = f"{source_path}_{idx}"
        vector_id = hashlib.md5(id_input.encode()).hexdigest()

        # Embed the chunk text
        embedding = embeddings.embed_query(chunk_text)

        # Minimal metadata: source_path, doc_type, chunk_index
        metadata = {
            "source_path": source_path,
            "doc_type": doc_type,
            "chunk_index": idx,
            "chunk_text": chunk_text
        }

        vectors.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            }
        )

    return vectors


def ingest_corpus(corpus_dir: str) -> Dict[str, int]:
    """
    Complete ingestion pipeline: load -> chunk -> embed -> upsert to Pinecone.
    
    Args:
        corpus_dir: Path to corpus directory
    
    Returns:
        Stats dict with counts: files, chunks, vectors upserted
    """
    # Initialize RAG components
    embeddings = init_rag_components()

    # Load documents
    print("\n" + "=" * 70)
    print("STEP 1: Loading Documents")
    print("=" * 70)
    documents = load_documents(corpus_dir)

    # Chunk documents
    print("\n" + "=" * 70)
    print("STEP 2: Chunking Documents (100 tokens, 20% overlap)")
    print("=" * 70)
    all_vectors = []
    chunk_count = 0

    for source_path, doc_type, text in documents:
        chunks_list = chunk_document(text, doc_type, chunk_size=100, overlap_percent=0.2)
        print(f"  {source_path}: {len(chunks_list)} chunks")
        chunk_count += len(chunks_list)

        # Create vectors for these chunks
        vectors = create_chunk_vectors(chunks_list, source_path, doc_type, embeddings)
        all_vectors.extend(vectors)

    print(f"\n✓ Total chunks created: {chunk_count}")

    # Upsert to Pinecone in batches
    print("\n" + "=" * 70)
    print(f"STEP 3: Preparing vectors for upsert")
    print("=" * 70)
    
    # Verify vector dimensionality
    ids = []
    embeds = []
    metadata = []
    
    for v in all_vectors:
        vector_dim = len(v["values"])
        if vector_dim != 1536:
            raise ValueError(
                f"Vector {v['id']} has dimensionality {vector_dim}, expected 1536"
            )
        ids.append(v["id"])
        embeds.append(v["values"])
        metadata.append(v["metadata"])
    
    print(f"All {len(ids)} vectors verified (1536 dimensions each)")
    
    # Upsert all vectors asynchronously in batches
    print(f"STEP 4: Upserting {len(ids)} vectors to Pinecone (batches of 200, 20 simultaneous requests)")
    print("=" * 70)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create tuples for upsert: (id, values, metadata)
    vector_tuples = list(zip(ids, embeds, metadata))
    
    # Upsert in batches of 200 with 20 simultaneous requests
    with pc.Index(INDEX_NAME, pool_threads=20) as index:
        async_results = [
            index.upsert(vectors=batch, async_req=True) 
            for batch in chunks(vector_tuples, batch_size=200)
        ]
        # Wait for all async requests to complete
        for i, async_result in enumerate(async_results):
            async_result.get()
            print(f"  Batch {i + 1}/{len(async_results)} upserted")
    
    print(f"✓ Upserted {len(ids)} vectors")

    print("\n" + "=" * 70)
    print("✓ INGESTION COMPLETE")
    print("=" * 70)
    print(f"Files processed: {len(documents)}")
    print(f"Total chunks: {chunk_count}")
    print(f"Vectors upserted: {len(ids)}")
    print()

    return {
        "files_processed": len(documents),
        "total_chunks": chunk_count,
        "vectors_upserted": len(ids),
    }


if __name__ == "__main__":
    stats = ingest_corpus("./corpus/")
    print("Ingestion stats:", stats)