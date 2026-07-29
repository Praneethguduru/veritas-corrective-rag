from veritas.llm.base import LLMProvider
from veritas.retrieval.utils import parse_yes_no


def check_groundedness(
    answer: str,
    context: str,
    llm: LLMProvider,
) -> bool:
    """
    Determine whether an answer is fully supported
    by the provided context.
    """

    prompt = f"""You are a factual verification assistant.

Determine whether every factual claim in the answer
is supported by the provided context.

Requirements:
- Respond with ONLY one word.
- Answer "yes" if every claim is supported.
- Answer "no" if any claim is unsupported,
  contradicted, or cannot be verified.
- Do not explain your answer.

Context:
{context}

Answer:
{answer}
"""

    response = llm.generate(prompt)

    return parse_yes_no(response)


if __name__ == "__main__":
    from veritas.llm.factory import get_llm_provider

    llm = get_llm_provider("groq")  # or "ollama"

    context = """
The Transformer architecture was introduced in
'Attention Is All You Need'. It replaces recurrent
networks with self-attention and multi-head attention.
"""

    grounded_answer = (
        "The Transformer architecture uses self-attention "
        "and multi-head attention instead of recurrent networks."
    )

    hallucinated_answer = (
        "The Transformer architecture was introduced by Google in 2019 "
        "and replaces recurrent networks with convolutional layers."
    )

    print("=" * 60)
    print("Groundedness Test")
    print("=" * 60)

    print(
        "Grounded answer:",
        check_groundedness(
            grounded_answer,
            context,
            llm,
        ),
    )

    print(
        "Hallucinated answer:",
        check_groundedness(
            hallucinated_answer,
            context,
            llm,
        ),
    )