import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# -----------------------------
# Configuration
# -----------------------------
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_TOKENS = 5

# -----------------------------
# Tokenizer
# -----------------------------
encoding = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(encoding.encode(text))


# -----------------------------
# Remove OCR picture text
# -----------------------------
def strip_picture_text(markdown_text: str) -> str:
    """
    Remove OCR'd picture text blocks inserted by pymupdf4llm.

    These blocks usually contain noisy text extracted from figures
    and aren't useful for RAG retrieval.
    """
    pattern = r"<!-- Start of picture text -->.*?<!-- End of picture text -->"
    return re.sub(pattern, "", markdown_text, flags=re.DOTALL)


# -----------------------------
# Validate headings
# -----------------------------
def looks_like_real_heading(text: str) -> bool:
    """
    Ignore fake headings such as equations.

    Example rejected:
        # x = y + z

    Example accepted:
        # Introduction
        # 3 Experiments
        # A. Appendix
    """
    if "=" in text:
        return False

    return bool(re.search(r"[A-Za-z]", text))


# -----------------------------
# Split by markdown headings
# -----------------------------
def split_by_headers(markdown_text: str) -> list[dict]:
    """
    Returns:
    [
        {
            "heading": "...",
            "content": "..."
        },
        ...
    ]
    """

    pattern = r"^(#{1,6})\s+(.*)$"

    sections = []

    current_heading = "front_matter"
    current_content = []

    def flush():
        joined = "\n".join(current_content).strip()

        if joined:
            sections.append(
                {
                    "heading": current_heading,
                    "content": joined,
                }
            )

    for line in markdown_text.splitlines():

        match = re.match(pattern, line)

        if match and looks_like_real_heading(match.group(2)):
            flush()

            current_content.clear()
            current_heading = match.group(2).strip()

        else:
            current_content.append(line)

    flush()

    return sections


# -----------------------------
# Split oversized sections
# -----------------------------
def sub_split_if_needed(
    section: dict,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:

    if token_length(section["content"]) <= max_tokens:
        return [
            {
                **section,
                "chunk_index": 0,
            }
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=token_length,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    pieces = splitter.split_text(section["content"])

    return [
        {
            "heading": section["heading"],
            "content": piece,
            "chunk_index": i,
        }
        for i, piece in enumerate(pieces)
    ]


# -----------------------------
# Main chunking pipeline
# -----------------------------
def chunk_document(
    markdown_text: str,
    source_file: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Complete chunking pipeline.

    Returns:
    [
        {
            "id": 0,
            "source": "...",
            "heading": "...",
            "chunk_index": 0,
            "content": "...",
            "token_count": 421,
        },
        ...
    ]
    """

    markdown_text = strip_picture_text(markdown_text)

    sections = split_by_headers(markdown_text)

    chunks = []
    chunk_id = 0

    for section in sections:

        split_chunks = sub_split_if_needed(
            section,
            max_tokens=max_tokens,
            overlap=overlap,
        )

        for chunk in split_chunks:

            tokens = token_length(chunk["content"])

            # Skip tiny fragments
            if tokens < MIN_CHUNK_TOKENS:
                continue

            chunks.append(
                {
                    "id": chunk_id,
                    "source": source_file,
                    "heading": chunk["heading"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "token_count": tokens,
                }
            )

            chunk_id += 1

    return chunks


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    import pymupdf4llm

    pdfs = [
        "attention_is_all_you_need.pdf",
        "BERT.pdf",
        "Language_model_for_few_shot_learners.pdf",
    ]

    for pdf in pdfs:

        markdown = pymupdf4llm.to_markdown(f"data/raw/{pdf}")

        chunks = chunk_document(markdown, pdf)

        print(f"\n{'=' * 90}")
        print(f"{pdf}")
        print(f"Total chunks: {len(chunks)}")
        print(f"{'=' * 90}")

        for chunk in chunks:

            preview = chunk["content"][:100].replace("\n", " ")

            print(
                f"[{chunk['id']:03}] "
                f"{chunk['heading']} "
                f"(chunk {chunk['chunk_index']}, "
                f"{chunk['token_count']} tokens)"
            )
            print(f"Preview: {preview}...")
            print("-" * 90)