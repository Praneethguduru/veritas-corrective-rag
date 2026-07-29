from functools import lru_cache
from sentence_transformers import SentenceTransformer
import chromadb
from veritas.ingestion.chunking import chunk_document


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the BAAI BGE embedding model.

    Returns:
        SentenceTransformer: Cached embedding model.
    """
    return SentenceTransformer("BAAI/bge-small-en-v1.5")



def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[list[float]]:
    """
    Generate embeddings for document chunks.
    """

    texts = [
        f"Section: {chunk['heading']}\n\n{chunk['content']}"
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()


def get_chroma_collection(
    path: str = "vectorstore/",
    name: str = "veritas_papers",
) -> chromadb.Collection:
    """
    Create (or open) a persistent ChromaDB collection.
    """

    client = chromadb.PersistentClient(path=path)

    collection = client.get_or_create_collection(
        name=name
    )

    return collection

def index_chunks(
    chunks: list[dict],
    collection,
    model,
) -> None:
    """
    Generate embeddings for chunks and store them in ChromaDB.
    """

    # Generate embeddings
    embeddings = embed_chunks(chunks, model)

    # Build IDs
    ids = [
        f"{chunk['source']}_{chunk['id']}"
        for chunk in chunks
    ]

    # Original chunk text (not heading-prefixed)
    documents = [
        chunk["content"]
        for chunk in chunks
    ]

    # Metadata
    metadatas = [
        {
            "source": chunk["source"],
            "heading": chunk["heading"],
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk["token_count"],
        }
        for chunk in chunks
    ]

    # Store in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


if __name__ == "__main__":

    import pymupdf4llm

    # Load Chroma collection
    collection = get_chroma_collection()

    # Optional: clear existing data to avoid duplicate ID errors
    try:
        collection.delete(where={})
        print("Existing collection cleared.\n")
    except Exception:
        pass

    # Load embedding model once
    model = get_embedding_model()

    pdfs = [
        "attention_is_all_you_need.pdf",
        "BERT.pdf",
        "Language_model_for_few_shot_learners.pdf",
    ]

    total_chunks = 0

    # -----------------------------
    # Index all papers
    # -----------------------------
    for pdf in pdfs:

        print(f"Processing {pdf}...")

        markdown = pymupdf4llm.to_markdown(f"data/raw/{pdf}")

        chunks = chunk_document(markdown, pdf)

        index_chunks(chunks, collection, model)

        total_chunks += len(chunks)

        print(f"Indexed {len(chunks)} chunks.\n")

    print(f"Finished indexing {total_chunks} chunks.\n")

    # -----------------------------
    # Test retrieval
    # -----------------------------
    query = "query: What is multi-head attention?"

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    print("=" * 80)
    print("Top 3 Results")
    print("=" * 80)

    for i in range(len(results["ids"][0])):

        print(f"\nRank {i + 1}")
        print(f"ID      : {results['ids'][0][i]}")
        print(f"Source  : {results['metadatas'][0][i]['source']}")
        print(f"Heading : {results['metadatas'][0][i]['heading']}")
        print(f"Distance: {results['distances'][0][i]}")

        preview = results["documents"][0][i][:120].replace("\n", " ")
        print(f"Preview : {preview}...")