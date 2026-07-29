from fastapi import FastAPI
from pydantic import BaseModel

from veritas.graph.build import build_graph

app = FastAPI(title="Veritas CRAG API")

_graph = build_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    retry_count: int
    used_web_search: bool


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    initial_state = {
        "query": request.query,
        "rewritten_query": None,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "web_results": [],
        "answer": None,
        "grounded": None,
        "retry_count": 0,
    }

    final_state = _graph.invoke(initial_state)

    return QueryResponse(
        answer=final_state["answer"],
        grounded=final_state["grounded"],
        retry_count=final_state["retry_count"],
        used_web_search=len(final_state["web_results"]) > 0,
    )
