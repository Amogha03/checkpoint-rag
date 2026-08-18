from langchain_core.documents import Document
from retrieve import retrieve
from generate import format_docs, build_rag_chain, RAGResponse


def run_pipeline(
    query: str,
    documents: list[Document] = None,
    top_k: int = 5
) -> tuple[RAGResponse, list[Document]]:
    """
    End-to-end RAG pipeline entry point.
    - If documents is passed: Executes Hybrid Search (Pinecone + BM25).
    - If documents=None: Fallback to Vector-Only search (Pinecone).
    
    Returns:
        tuple[RAGResponse, list[Document]]: The Pydantic structured output and raw retrieved docs.
    """
    # 1. Retrieve standardized Document objects
    retrieved_docs = retrieve(query=query, top_k=top_k, documents=documents)

    # 2. Format documents into prompt context string
    formatted_context = format_docs(retrieved_docs)

    # 3. Invoke LCEL generation chain
    chain = build_rag_chain()
    response: RAGResponse = chain.invoke({"context": formatted_context, "query": query})

    return response, retrieved_docs


if __name__ == "__main__":
    test_query = "Why am I getting 429 errors when I'm under the rate limit?"

    # Execute vector-only Pinecone fallback mode
    response, docs = run_pipeline(query=test_query, top_k=5)

    print("=== Final Answer ===")
    print(response.answer)
    print("\n=== Sources ===")
    print(response.sources)
    print("\n=== Confidence ===")
    print(response.confidence)
    print(f"\n=== Retrieved Chunks Count: {len(docs)} ===")