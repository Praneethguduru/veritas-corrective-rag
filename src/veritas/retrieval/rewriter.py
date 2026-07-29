from veritas.llm.base import LLMProvider


def rewrite_query(
    query: str,
    llm: LLMProvider,
) -> str:
    """
    Rewrite a user query to improve document retrieval.

    The rewritten query should be more specific, use precise
    technical terminology where appropriate, and preserve the
    user's original intent.
    """

    prompt = f"""You are a search query rewriting assistant.

        Rewrite the user's query to improve document retrieval.

        Requirements:
        - Preserve the original intent.
        - Make the query more specific only if necessary.
        - Use precise technical terminology only when it improves retrieval.
        - Keep the rewritten query concise.
        - Use no more than 15 words.
        - Prefer a short search phrase over a full sentence.
        - Do NOT answer the query.
        - Respond with ONLY the rewritten query.
        - Do not include explanations.
        - Do not include quotation marks.
        - Do not include prefixes such as "Rewritten query:".

        Original query:
        {query}
        """
    response = llm.generate(prompt)

    rewritten = response.strip()

    # Remove surrounding quotes if the model adds them anyway.
    rewritten = rewritten.strip("\"'")
    
    return rewritten


if __name__ == "__main__":
    from veritas.llm.factory import get_llm_provider

    # Change to "ollama" if you want to test the local model.
    llm = get_llm_provider("groq")

    original_query = "how do transformers work"

    rewritten_query = rewrite_query(
        original_query,
        llm,
    )

    print("=" * 60)
    print("Query Rewriting Test")
    print("=" * 60)
    print(f"Original : {original_query}")
    print(f"Rewritten: {rewritten_query}")