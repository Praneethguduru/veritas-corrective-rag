from typing import TypedDict

from veritas.retrieval.web_search import WebResult


class RetrievedChunk(TypedDict):
    id: str
    source: str
    heading: str
    content: str


class GraphState(TypedDict):
    # Original user query
    query: str

    # Rewritten query (None until rewritten)
    rewritten_query: str | None

    # Raw retrieved chunks
    retrieved_chunks: list[RetrievedChunk]

    # Chunks that passed relevance grading
    relevant_chunks: list[RetrievedChunk]

    # Results returned by web search
    web_results: list[WebResult]

    # Final generated answer
    answer: str | None

    # None = not checked yet
    grounded: bool | None

    # Prevent infinite loops
    retry_count: int