import os
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Read configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "checkpoint-rag-index"  # global index name
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def init_pinecone_index():
    """Initializes and returns a connection to the Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    return index


def init_rag_components():
    """Initializes and returns the embeddings, vector store, and LLM instances."""
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment.")

    # Ensure the Pinecone index exists before connecting
    index = init_pinecone_index()

    # 1. Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    return index, embeddings


def get_hybrid_retriever(documents: list[Document], k: int = 5):
    """Combines Pinecone dense retrieval and BM25 sparse retrieval."""
    if not documents:
        raise ValueError("documents must contain at least one chunk.")
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment.")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )
    pinecone_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[pinecone_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )


def answer_query_hybrid(query: str, documents: list[Document]):
    """Retrieves relevant chunks using hybrid search and asks the LLM to answer."""
    retriever = get_hybrid_retriever(documents)
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    prompt = f"Answer using the context:\n\nContext:\n{context}\n\nQuestion:\n{query}"
    response = llm.invoke(prompt)
    return response.content, docs


def retrieve(query: str, top_k: int = 5, documents: list[Document] = None) -> list[Document]:
    if documents is not None:
        retriever = get_hybrid_retriever(documents, k=top_k)
        return retriever.invoke(query)

    # Vector-only fallback: Pinecone mapping handled HERE
    index, embeddings = init_rag_components()
    query_vector = embeddings.embed_query(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    docs = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        
        # Standardize metadata directly inside retrieve()
        doc_id = meta.get("source_path", "")
        text_content = meta.get("chunk_text", "")

        docs.append(
            Document(
                page_content=text_content,
                metadata={
                    "doc_id": doc_id,
                }
            )
        )
    return docs
# result = retrieve("How do I rotate an API key safely?", top_k=5)
# print("Retrieved documents:", result)