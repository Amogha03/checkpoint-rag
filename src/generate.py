import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from retrieve import retrieve

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# 1. Pydantic Output Schema
class RAGResponse(BaseModel):
    answer: str = Field(
        description="The detailed, context-grounded answer to the user query."
    )
    sources: list[str] = Field(
        description="List of source filenames used to formulate the answer."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Self-assessed confidence level based on how thoroughly the context covers the question."
    )


# 2. Document Context Formatter
def format_docs(docs: list[Document]) -> str:
    """Formats document chunks into prompt context using source_path for inline citations."""
    formatted = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id")
        formatted.append(f"--- Document ID: {doc_id} ---\nContent:\n{doc.page_content}\n")
    return "\n".join(formatted)

# 3. Prompt Template
SYSTEM_PROMPT = """You are an internal AI Support Assistant for the Customer Success team. Your primary task is to answer employee queries accurately using ONLY the provided internal documentation, runbooks, and post-mortems.

Instructions:
1. Grounding: Base your answer strictly on the facts present in the context. Do NOT extrapolate or bring in outside knowledge.
2. Sources: List the document IDs of the sources used to formulate the answer.
3. Unanswered Queries: If the context lacks sufficient information, explicitly state that you cannot answer from the knowledge base and set confidence to 'low'.

### Internal Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CSM Query: {query}"),
])


# 4. LCEL Generation Chain
def build_rag_chain():
    """Constructs the prompt and structured output generation chain."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(RAGResponse)
    
    return prompt | structured_llm


# 5. Main Execution Wrapper
def generate_response(query: str, documents: list[Document] = None, top_k: int = 4) -> RAGResponse:
    """Supports both Hybrid search (if documents provided) and Vector-only (if documents=None)."""
    
    # 1. Retrieve docs (hybrid or vector-only fallback)
    retrieved_docs = retrieve(query=query, top_k=top_k, documents=documents)
    
    # 2. Format context for prompt
    formatted_context = format_docs(retrieved_docs)
    
    # 3. Build & invoke chain
    chain = build_rag_chain()
    return chain.invoke({"context": formatted_context, "query": query})


if __name__ == "__main__":
    # Quick execution test using Pinecone fallback mode
    test_query = "Why am I getting 429 errors when I'm under the rate limit?"
    
    response: RAGResponse = generate_response(test_query)
    
    print("=== Answer ===")
    print(response.answer)
    print("\n=== Sources ===")
    print(response.sources)
    print("\n=== Confidence ===")
    print(response.confidence)