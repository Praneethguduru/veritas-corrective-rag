from typing import Any
from veritas.retrieval.groundedness import check_groundedness
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
from veritas.retrieval.rewriter import rewrite_query
from veritas.retrieval.web_search import web_search


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

    query_embedding = embed_query(active_query, embedding_model)

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

    return {"retrieved_chunks": retrieved_chunks}


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
        if grade_relevance(
            query=active_query,
            chunk_content=chunk["content"],
            llm=llm,
        ):
            relevant_chunks.append(chunk)

    return {"relevant_chunks": relevant_chunks}


def rewrite_node(state: GraphState) -> dict[str, Any]:
    """
    Rewrite the ORIGINAL user query and increment retry count.
    """

    llm = get_llm_provider("ollama")

    rewritten_query = rewrite_query(
        query=state["query"],
        llm=llm,
    )

    return {
        "rewritten_query": rewritten_query,
        "retry_count": state["retry_count"] + 1,
    }


def web_search_node(state: GraphState) -> dict[str, Any]:
    """
    Search the web using DuckDuckGo when local retrieval fails.
    """

    active_query = (
        state["rewritten_query"]
        if state["rewritten_query"] is not None
        else state["query"]
    )

    web_results = web_search(active_query)

    return {"web_results": web_results}


def generate_node(state: GraphState) -> dict[str, Any]:
    """
    Generate a final answer using local chunks and/or web results.
    """

    context_parts: list[str] = []

    for chunk in state["relevant_chunks"]:
        context_parts.append(
            f"[Local: {chunk['heading']}]\n{chunk['content']}"
        )

    for result in state["web_results"]:
        context_parts.append(
            f"[Web: {result['title']}]\n{result['snippet']}"
        )

    context = "\n\n".join(context_parts)

    llm = get_llm_provider("ollama")

    prompt = f"""You are a helpful assistant answering questions using ONLY the provided context.

Requirements:
- Answer using ONLY the context.
- Do not invent facts.
- If the context is insufficient, explicitly say so.
- Be concise.

Context:
{context}

Question:
{state["query"]}

Answer:"""

    answer = llm.generate(prompt)

    return {"answer": answer}

def groundedness_node(state: GraphState) -> dict[str, Any]:
    """
    Verify whether the generated answer is grounded in the
    available context (local chunks + web results).
    """
    context_parts: list[str] = []

    for chunk in state["relevant_chunks"]:
        context_parts.append(f"[Local: {chunk['heading']}]\n{chunk['content']}")

    for result in state["web_results"]:
        context_parts.append(f"[Web: {result['title']}]\n{result['snippet']}")

    context = "\n\n".join(context_parts)

    llm = get_llm_provider("ollama")

    is_grounded = check_groundedness(
        answer=state["answer"],
        context=context,
        llm=llm,
    )

    return {"grounded": is_grounded}







if __name__ == "__main__":

    state: GraphState = {
        "query": "What is the capital of France?",
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

    update = retrieve_node(state)
    state["retrieved_chunks"] = update["retrieved_chunks"]

    for chunk in state["retrieved_chunks"]:
        print(f"- {chunk['heading']}")

    print("\n" + "=" * 60)
    print("Grade Node")
    print("=" * 60)

    update = grade_node(state)
    state["relevant_chunks"] = update["relevant_chunks"]

    print(f"\nRelevant Chunks: {len(state['relevant_chunks'])}\n")

    for chunk in state["relevant_chunks"]:
        print(f"- {chunk['heading']}")

    if len(state["relevant_chunks"]) == 0:

        print("\n" + "=" * 60)
        print("Rewrite Node")
        print("=" * 60)

        update = rewrite_node(state)
        state["rewritten_query"] = update["rewritten_query"]
        state["retry_count"] = update["retry_count"]

        print(f"\nOriginal Query : {state['query']}")
        print(f"Rewritten Query: {state['rewritten_query']}")
        print(f"Retry Count    : {state['retry_count']}")

        print("\n" + "=" * 60)
        print("Web Search Node")
        print("=" * 60)

        update = web_search_node(state)
        state["web_results"] = update["web_results"]

        print(f"\nResults Returned: {len(state['web_results'])}\n")

        for i, result in enumerate(state["web_results"], start=1):
            print(f"Result {i}")
            print(f"Title   : {result['title']}")
            print(f"URL     : {result['url']}")
            print(f"Snippet : {result['snippet']}")
            print("-" * 80)

    print("\n" + "=" * 60)
    print("Generate Node")
    print("=" * 60)

    update = generate_node(state)
    state["answer"] = update["answer"]

    print("\nFinal Answer:\n")
    print(state["answer"])
    print("\n" + "=" * 60)
    print("Groundedness Node")
    print("=" * 60)

    update = groundedness_node(state)
    state["grounded"] = update["grounded"]

    print(f"\nGrounded: {state['grounded']}")