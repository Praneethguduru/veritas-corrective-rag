from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.

        Returns:
            Generated text response.
        """
        pass