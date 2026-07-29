from typing import TypedDict

from ddgs import DDGS


class WebResult(TypedDict):
    title: str
    url: str
    snippet: str


def web_search(
    query: str,
    max_results: int = 3,
) -> list[WebResult]:
    """
    Search the web using DuckDuckGo.

    Returns:
        A list of web search results. If the search fails,
        an empty list is returned.
    """

    try:
        results: list[WebResult] = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=max_results,
            )

            for result in search_results:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                    }
                )

        return results

    except Exception as e:
        print(f"Warning: Web search failed: {e}")
        return []


if __name__ == "__main__":
    query = "What is multi-head attention?"

    results = web_search(query)

    print("=" * 80)
    print(f"Web Search Results for: {query}")
    print("=" * 80)

    if not results:
        print("No web search results found.")
    else:
        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Title   : {result['title']}")
            print(f"URL     : {result['url']}")
            print(f"Snippet : {result['snippet']}")
            print("-" * 80)