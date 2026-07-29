from ollama import Client

from veritas.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """
    LLM provider backed by Ollama.
    """

    def __init__(
        self,
        model: str = "llama3.1:latest",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.client = Client(host=host)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a response using Ollama.
        """

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response = self.client.chat(
            model=self.model,
            messages=messages,
        )

        return response["message"]["content"]
    
if __name__ == "__main__":

    llm = OllamaProvider()

    response = llm.generate(
        prompt="What is Retrieval-Augmented Generation in one sentence?"
    )

    print(response)