from veritas.ingestion.chunking import chunk_document
from veritas.llm.base import LLMProvider
from veritas.retrieval.utils import parse_yes_no

...

response = llm.generate(prompt)
return parse_yes_no(response)
def grade_relevance(
    query: str,
    chunk_content: str,
    llm: LLMProvider,
) -> bool:
    """
    Use an LLM to determine whether a retrieved chunk
    is relevant to the user's query.
    """

    prompt = f"""You are a retrieval relevance judge.

    Determine whether the retrieved document contains information
    that helps answer the user's query.

    Respond with ONLY one word:

    yes

    or

    no

    Do not explain your answer.

    Query:
    {query}

    Retrieved Document:
    {chunk_content}
"""

    response = llm.generate(prompt)
    return parse_yes_no(response)
    answer = response.strip().lower()

    if answer.startswith("yes"):
        return True

    if answer.startswith("no"):
        return False

    # Fallback for slightly verbose models like:
    # "I think yes, this seems relevant."
    return "yes" in answer


def embed_query(query: str, model) -> list[float]:
    """
    Embed a user query using the same BGE model used for indexing,
    applying the recommended query prefix for bge-small-en-v1.5.
    """

    prefix = "Represent this sentence for searching relevant passages: "

    return model.encode(
        prefix + query,
        convert_to_numpy=True,
    ).tolist()


if __name__ == "__main__":
    from veritas.ingestion.indexer import (
        get_chroma_collection,
        get_embedding_model,
    )
    from veritas.llm.factory import get_llm_provider

    llm = get_llm_provider("ollama")  # or "groq"

    embed_model = get_embedding_model()
    collection = get_chroma_collection()

    query = "What is multi-head attention?"

    # ----------------------------------------------------------
    # Isolated smoke test
    # ----------------------------------------------------------

    relevant_chunk = """
Instead of performing a single attention function with d_model-dimensional
keys, values and queries, we found it beneficial to linearly project the
queries, keys and values h times with different, learned linear projections.
"""

    irrelevant_chunk = """
We introduce a new language representation model called BERT, which stands
for Bidirectional Encoder Representations from Transformers. Unlike recent
language representation models, BERT is designed to pre-train deep
bidirectional representations.
"""

    print("=" * 60)
    print("Isolated smoke test")
    print("=" * 60)

    print(
        "Relevant chunk graded as:",
        grade_relevance(query, relevant_chunk, llm),
    )

    print(
        "Irrelevant chunk graded as:",
        grade_relevance(query, irrelevant_chunk, llm),
    )

    # ----------------------------------------------------------
    # Integration test
    # Query -> Embed -> Chroma -> Grade
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("Integration test (real retrieval)")
    print("=" * 60)

    query_embedding = embed_query(query, embed_model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
    )

    for doc_id, metadata, document in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
    ):
        is_relevant = grade_relevance(
            query=query,
            chunk_content=document,
            llm=llm,
        )

        print(f"\nDocument ID : {doc_id}")
        print(f"Heading     : {metadata['heading']}")
        print(f"Source      : {metadata['source']}")
        print(f"Relevant    : {is_relevant}")
        print("-" * 60)