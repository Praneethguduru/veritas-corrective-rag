from typing import Any

from veritas.graph.state import GraphState, RetrievedChunk
from veritas.ingestion.indexer import (
    get_chroma_collection,
    get_embedding_model,
)
from veritas.retrieval.grader import embed_query


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """
    Retrieve the top-k most relevant chunks from ChromaDB.
    """

    # Use rewritten query if available.
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

    update = retrieve_node(state)

    print("\nRetrieved Chunks\n")

    for chunk in update["retrieved_chunks"]:
        print(f"- {chunk['heading']}")