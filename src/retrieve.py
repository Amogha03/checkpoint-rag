import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain_pinecone import PineconeVectorStore

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


def retrieve(query, top_k):
    """Retrieves relevant documents from the Pinecone index based on the query."""
    index, embeddings = init_rag_components()
    embedding = embeddings.embed_query(query)
    
    # Query the index using the embedding
    docs = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return docs
