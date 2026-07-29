import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# -----------------------------
# Tokenizer
# -----------------------------
encoding = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    return len(encoding.encode(text))


# -----------------------------
# Preprocessing: strip OCR'd image/picture text
# -----------------------------
def strip_picture_text(markdown_text: str) -> str:
    """
    Removes <!-- Start of picture text --> ... <!-- End of picture text -->
    blocks. These are OCR'd captions/text pulled from embedded images
    (e.g. attention visualization figures) and are usually noisy,
    repetitive, and not useful for retrieval.
    """
    pattern = r"<!-- Start of picture text -->.*?<!-- End of picture text -->"
    return re.sub(pattern, "", markdown_text, flags=re.DOTALL)


# -----------------------------
# Heading validity check (Fix 4)
# -----------------------------
def looks_like_real_heading(text: str) -> bool:
    """
    Returns True only if the text looks like a real section title:
    - contains at least one real word (3+ consecutive letters)
    - does NOT contain an '=' sign (a strong signal it's a formula, not a heading)
    """
    if "=" in text:
        return False
    return bool(re.search(r"[A-Za-z]{3,}", text))


# -----------------------------
# Stage 1: Split by Markdown headers
# -----------------------------
def split_by_headers(markdown_text: str) -> list[dict]:
    """
    Splits markdown into sections based on headings.

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
    current_heading = "front_matter"  # Fix 1: no longer a fake "Introduction"
    current_content: list[str] = []

    def flush():
        """Append the current section if it has real (non-empty) content."""
        joined = "\n".join(current_content).strip()
        if joined:  # Fix 2: check actual text, not just list truthiness
            sections.append(
                {
                    "heading": current_heading,
                    "content": joined,
                }
            )

    for line in markdown_text.splitlines():
        match = re.match(pattern, line)

        if match and looks_like_real_heading(match.group(2)):
            # Fix 4: only treat as a new heading if it looks like a real title,
            # not a formula or stray symbol line.
            flush()
            current_heading = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    flush()  # flush whatever is left at end of document

    return sections


# -----------------------------
# Stage 2: Split long sections
# -----------------------------
def sub_split_if_needed(
    section: dict,
    max_tokens: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    If a section fits within max_tokens, returns it as-is (with chunk_index=0
    for consistency — Fix 3). Otherwise recursively splits it into overlapping
    sub-chunks, carrying the parent heading forward on each piece.
    """
    if token_length(section["content"]) <= max_tokens:
        return [{**section, "chunk_index": 0}]

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

    output = []
    for i, piece in enumerate(pieces):
        output.append(
            {
                "heading": section["heading"],
                "content": piece,
                "chunk_index": i,
            }
        )

    return output


# -----------------------------
# Stage 3: Orchestrator
# -----------------------------
def chunk_document(
    markdown_text: str,
    source_file: str,
    max_tokens: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Full pipeline: strip picture-text noise -> split by headers ->
    sub-split long sections -> attach full metadata.

    Returns a list of chunk dicts, each with:
        - id: int, unique within this document
        - source: str, source filename
        - heading: str, section heading this chunk belongs to
        - chunk_index: int, position within the section (0 if not split)
        - content: str, the chunk text
        - token_count: int
    """
    markdown_text = strip_picture_text(markdown_text)

    sections = split_by_headers(markdown_text)

    chunks = []
    chunk_id = 0

    for section in sections:
        split_chunks = sub_split_if_needed(section, max_tokens=max_tokens, overlap=overlap)

        for chunk in split_chunks:
            chunk["id"] = chunk_id
            chunk["source"] = source_file
            chunk["token_count"] = token_length(chunk["content"])

            chunks.append(chunk)
            chunk_id += 1

    return chunks


# -----------------------------
# Manual test run
# -----------------------------
if __name__ == "__main__":
    import pymupdf4llm

    for filename in [
        "attention_is_all_you_need.pdf",
        "BERT.pdf",
        "Language_model_for_few_shot_learners.pdf",
    ]:
        markdown = pymupdf4llm.to_markdown(f"data/raw/{filename}")
        chunks = chunk_document(markdown, filename)
        print(f"\n\n########## {filename}: {len(chunks)} chunks ##########\n")
        for chunk in chunks:
            preview = chunk["content"][:80].replace("\n", " ")
            print(f"[{chunk['id']}] {chunk['heading']} (idx {chunk['chunk_index']}, {chunk['token_count']} tok): {preview}...")