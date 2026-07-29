from functools import lru_cache

from veritas.llm.base import LLMProvider
from veritas.llm.groq_provider import GroqProvider
from veritas.llm.ollama_provider import OllamaProvider


@lru_cache(maxsize=None)
def get_llm_provider(backend: str) -> LLMProvider:
    backend = backend.lower()

    if backend == "ollama":
        return OllamaProvider()

    if backend == "groq":
        return GroqProvider()

    raise ValueError(f"Unknown LLM backend: {backend}")


if __name__ == "__main__":

    prompt = "What is Retrieval-Augmented Generation in one sentence?"

    print("=" * 60)
    print("Testing Ollama")
    print("=" * 60)

    ollama = get_llm_provider("ollama")
    print(ollama.generate(prompt))

    print("\n" + "=" * 60)
    print("Testing Groq")
    print("=" * 60)

    groq = get_llm_provider("groq")
    print(groq.generate(prompt))