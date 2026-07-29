from typing import Any

from veritas.graph.state import GraphState, RetrievedChunk
from veritas.ingestion.indexer import (
    get_chroma_collection,
    get_embedding_model,
)
from veritas.llm.factory import get_llm_provider
from veritas.retrieval.grader import (
    embed_query,
    grade_relevance,
)


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """
    Retrieve the top-k most relevant chunks from ChromaDB.
    """

    active_query = (
        state["rewritten_query"]
        if state["rewritten_query"] is not None
        else state["query"]
    )

    embedding_model = get_embedding_model()
    collection = get_chroma_collection()

    query_embedding = embed_query(
        active_query,
        embedding_model,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    retrieved_chunks: list[RetrievedChunk] = []

    for doc_id, metadata, document in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
    ):
        retrieved_chunks.append(
            {
                "id": doc_id,
                "source": metadata["source"],
                "heading": metadata["heading"],
                "content": document,
            }
        )

    return {
        "retrieved_chunks": retrieved_chunks,
    }


def grade_node(state: GraphState) -> dict[str, Any]:
    """
    Grade retrieved chunks for relevance using the configured LLM.
    """

    active_query = (
        state["rewritten_query"]
        if state["rewritten_query"] is not None
        else state["query"]
    )

    llm = get_llm_provider("ollama")

    relevant_chunks: list[RetrievedChunk] = []

    for chunk in state["retrieved_chunks"]:
        is_relevant = grade_relevance(
            query=active_query,
            chunk_content=chunk["content"],
            llm=llm,
        )

        if is_relevant:
            relevant_chunks.append(chunk)

    return {
        "relevant_chunks": relevant_chunks,
    }


if __name__ == "__main__":

    state: GraphState = {
        "query": "What is multi-head attention?",
        "rewritten_query": None,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "web_results": [],
        "answer": None,
        "grounded": None,
        "retry_count": 0,
    }

    print("=" * 60)
    print("Retrieve Node")
    print("=" * 60)

    retrieve_update = retrieve_node(state)

    state["retrieved_chunks"] = retrieve_update["retrieved_chunks"]

    for chunk in state["retrieved_chunks"]:
        print(f"- {chunk['heading']}")

    print("\n" + "=" * 60)
    print("Grade Node")
    print("=" * 60)

    grade_update = grade_node(state)

    state["relevant_chunks"] = grade_update["relevant_chunks"]

    print(f"\nRelevant Chunks: {len(state['relevant_chunks'])}\n")

    for chunk in state["relevant_chunks"]:
        print(f"- {chunk['heading']}")