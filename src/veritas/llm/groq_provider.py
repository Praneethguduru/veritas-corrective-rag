import os

from groq import Groq
import os
from veritas.llm.base import LLMProvider
from dotenv import load_dotenv



class GroqProvider(LLMProvider):
    """
    LLM provider backed by Groq.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        load_dotenv()
        self.client = Groq(
            api_key=api_key or os.getenv("GROQ_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a response using Groq.
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content


if __name__ == "__main__":

    llm = GroqProvider()

    response = llm.generate(
        "What is Retrieval-Augmented Generation in one sentence?"
    )

    print(response)