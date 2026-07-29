from langgraph.graph import StateGraph, END

from veritas.graph.state import GraphState
from veritas.graph.nodes import (
    retrieve_node,
    grade_node,
    rewrite_node,
    web_search_node,
    generate_node,
    groundedness_node,
)


def route_after_grading(state: GraphState) -> str:
    """
    Decide what to do after grading retrieved chunks.
    """

    if state["relevant_chunks"]:
        return "generate"

    if state["retry_count"] < 1:
        return "rewrite"

    return "web_search"


def route_after_groundedness(state: GraphState) -> str:
    """
    Decide what to do after checking groundedness.
    """

    if state["grounded"]:
        return "end"

    if not state["web_results"]:
        return "web_search"

    return "end"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("groundedness", groundedness_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search",
        },
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "groundedness")

    graph.add_conditional_edges(
        "groundedness",
        route_after_groundedness,
        {
            "web_search": "web_search",
            "end": END,
        },
    )

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    tests = [
        "What is multi-head attention?",
        "What is the capital of France?",
    ]

    for query in tests:
        print("\n" + "=" * 70)
        print("Query:", query)
        print("=" * 70)

        initial_state: GraphState = {
            "query": query,
            "rewritten_query": None,
            "retrieved_chunks": [],
            "relevant_chunks": [],
            "web_results": [],
            "answer": None,
            "grounded": None,
            "retry_count": 0,
        }

        final_state = app.invoke(initial_state)

        print("\nAnswer:")
        print(final_state["answer"])

        print("\nGrounded:", final_state["grounded"])
        print("Retry Count:", final_state["retry_count"])
        print("Relevant Chunks:", len(final_state["relevant_chunks"]))
        print("Web Results:", len(final_state["web_results"]))