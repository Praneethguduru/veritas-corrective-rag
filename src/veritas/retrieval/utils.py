def parse_yes_no(response: str) -> bool:
    """
    Parse an LLM yes/no response.

    Returns:
        True if the response indicates "yes",
        otherwise False.
    """

    answer = response.strip().lower()

    if answer.startswith("yes"):
        return True

    if answer.startswith("no"):
        return False

    return "yes" in answer