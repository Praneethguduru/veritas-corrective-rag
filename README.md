# Veritas — A Self-Correcting RAG System

Veritas is a question-answering system built on top of a small set of ML papers (Attention Is All You Need, BERT, and GPT-3). What makes it different from a basic "chat with your PDF" tool is that it **checks its own work**.

Most simple RAG systems just retrieve some text and hope it's relevant. Veritas doesn't trust its own retrieval by default. Instead, it:

1. Retrieves chunks from the papers
2. Grades whether those chunks are actually relevant to the question
3. If they're not relevant, it rewrites the question and tries again
4. If that still doesn't work, it searches the live web instead
5. After generating an answer, it double-checks whether the answer is actually backed by the source material
6. If the answer isn't backed by anything, it tries to fix that too, instead of just returning it

This pattern is called **Corrective RAG (CRAG)**. The goal is reliability — a system that notices when it doesn't know something, instead of confidently making things up.

---

## Why I built it this way

A basic RAG demo is easy to build but doesn't say much about engineering judgment — it just calls an embedding model and an LLM. I wanted to build something that shows how to make an AI system more trustworthy, since that's a real problem companies actually care about. Every extra step in this project (grading, rewriting, web fallback, groundedness checking) exists specifically to catch failure cases that a basic RAG system would silently get wrong.

---

## How it works (architecture)

The whole pipeline is built as a graph using **LangGraph**, where each step is a node, and the system decides which path to take based on what happened at the previous step.

```
User Question
      │
      ▼
  [ Retrieve ]  ← searches the local paper database (ChromaDB)
      │
      ▼
   [ Grade ]  ← is this content actually relevant?
      │
      ├── Relevant? ──────────────► [ Generate Answer ]
      │
      ├── Not relevant, first try ──► [ Rewrite Question ] ──► back to Retrieve
      │
      └── Not relevant, already retried ──► [ Web Search ] ──► [ Generate Answer ]
                                                                       │
                                                                       ▼
                                                            [ Check Groundedness ]
                                                                       │
                                              ┌────────────────────────┴───────────────┐
                                              │                                         │
                                     Answer is grounded                     Answer NOT grounded
                                              │                                         │
                                             Done                          Did we already try web search?
                                                                                        │
                                                                       ┌────────────────┴────────────────┐
                                                                       │                                  │
                                                                No, try it                          Yes, stop and
                                                                                                    return answer as-is
```

The system will retry at most once and web-search at most once, so it can never get stuck in an infinite loop.

---

## Tech stack

| Piece | What's used | Why |
|---|---|---|
| Language model | Ollama (local) or Groq (free API) | Free to run, switchable between local dev and a faster demo |
| Embeddings | BAAI/bge-small-en-v1.5 | Free, runs locally, good quality for its size |
| Vector database | ChromaDB | Simple, free, runs locally with no server setup |
| PDF parsing | PyMuPDF4LLM | Extracts clean text with headings preserved |
| Chunking | Custom two-stage splitter | Splits by section headers first, then by size, so chunks stay meaningful |
| Orchestration | LangGraph | Lets each step of the pipeline make its own decision about what to do next |
| Backend | FastAPI | Serves the pipeline as an API |
| Frontend | Streamlit | Simple demo UI |
| Web fallback | DuckDuckGo search | Free, no API key needed |

Everything in this project is free — no paid APIs, no paid cloud services.

---

## Running it yourself

**1. Install dependencies**
```bash
uv sync
```

**2. Pull a local model with Ollama** (or set up a free Groq API key instead)
```bash
ollama pull llama3.1:8b
```

**3. Index the papers** (only needs to be done once)
```bash
uv run src/veritas/ingestion/indexer.py
```

**4. Start the backend**
```bash
uv run uvicorn veritas.api.main:app --reload --app-dir src
```

**5. Start the demo UI** (in a second terminal)
```bash
uv run streamlit run src/veritas/ui/app.py
```

Then open the Streamlit link in your browser and ask a question.

---

## Example: watching it self-correct

**Question that the papers can answer directly:**
> "What is multi-head attention?"

Result: answered from the papers directly. No retry needed, no web search needed, answer is grounded.

**Question the papers can't answer at all:**
> "What is the capital of France?"

Result: local retrieval found nothing relevant, so the system rewrote the question, still found nothing, searched the web instead, and returned the correct answer ("Paris") — still checked for groundedness before returning it.

This is the actual point of the project: the system behaves differently depending on whether it actually knows the answer, instead of guessing either way.

---

## Evaluation

I built a small test set of 5 sample questions and checked the answers against three things:

- **Faithfulness** — is the answer actually backed by the source text?
- **Relevancy** — does the answer actually address the question?
- **Correctness** — does the answer match the known right answer?

Results:

| Metric | Score |
|---|---|
| Faithfulness | 4 / 5 |
| Relevancy | 5 / 5 |
| Correctness | 5 / 5 |

**Note on the one "failure":** the one answer marked as not "faithful" was actually correct — the answer was just a short one-word reply ("Paris."), and the judge model seemed to penalize short answers for not restating their sources, even though the answer itself was accurate. This is a known limitation of using an LLM to judge other LLM answers, not a real bug in the system.

---

## Known limitations

- The system can occasionally miss connections that require basic reasoning — for example, seeing "GPT-3 175" in the text and not immediately connecting that to "175 billion parameters."
- The LLM-as-judge approach for both relevance grading and evaluation is not perfectly consistent — it can be overly strict on short, correct answers.
- Currently only tested on 3 papers. Retrieval quality with a much larger document set hasn't been tested yet.

---

## Next steps

- Automated tests / CI pipeline
- A larger, more thorough evaluation set
- Docker packaging for easier deployment
- Deploying a live public demo (Hugging Face Spaces or Streamlit Community Cloud)

---

## License

MIT