import json

from veritas.graph.build import build_graph
from veritas.llm.factory import get_llm_provider
from veritas.retrieval.utils import parse_yes_no  # adjust import path if needed


def score_faithfulness(answer: str, context: str, llm) -> bool:
    """Is the answer fully supported by the context? (reuses groundedness-style judging)"""
    prompt = f"""You are a factual verification assistant.
Determine whether every factual claim in the answer is supported by the context.
Respond with ONLY one word: yes or no.

Context:
{context}

Answer:
{answer}
"""
    return parse_yes_no(llm.generate(prompt))


def score_relevancy(question: str, answer: str, llm) -> bool:
    """Does the answer actually address the question asked?"""
    prompt = f"""You are an answer relevancy judge.
Determine whether the answer directly addresses the question asked.
Respond with ONLY one word: yes or no.

Question:
{question}

Answer:
{answer}
"""
    return parse_yes_no(llm.generate(prompt))


def score_correctness(answer: str, ground_truth: str, llm) -> bool:
    """Does the answer match the known correct ground truth?"""
    prompt = f"""You are a correctness judge.
Determine whether the given answer is factually consistent with the ground truth below.
Respond with ONLY one word: yes or no.

Ground truth:
{ground_truth}

Answer:
{answer}
"""
    return parse_yes_no(llm.generate(prompt))


def main():
    with open("data/eval/golden_qa.json") as f:
        golden = json.load(f)

    app = build_graph()
    judge_llm = get_llm_provider("ollama")

    results = []

    for item in golden:
        initial_state = {
            "query": item["question"],
            "rewritten_query": None,
            "retrieved_chunks": [],
            "relevant_chunks": [],
            "web_results": [],
            "answer": None,
            "grounded": None,
            "retry_count": 0,
        }

        final_state = app.invoke(initial_state)

        context_parts = [c["content"] for c in final_state["relevant_chunks"]]
        context_parts += [w["snippet"] for w in final_state["web_results"]]
        context = "\n\n".join(context_parts)

        faithful = score_faithfulness(final_state["answer"], context, judge_llm)
        relevant = score_relevancy(item["question"], final_state["answer"], judge_llm)
        correct = score_correctness(final_state["answer"], item["ground_truth"], judge_llm)

        results.append({
            "question": item["question"],
            "answer": final_state["answer"],
            "faithful": faithful,
            "relevant": relevant,
            "correct": correct,
            "retries": final_state["retry_count"],
            "used_web_search": len(final_state["web_results"]) > 0,
        })

        print(f"Q: {item['question']}")
        print(f"A: {final_state['answer']}")
        print(f"Faithful: {faithful} | Relevant: {relevant} | Correct: {correct}")
        print("-" * 80)

    n = len(results)
    print("\n=== Summary ===")
    print(f"Faithfulness: {sum(r['faithful'] for r in results)}/{n}")
    print(f"Relevancy:    {sum(r['relevant'] for r in results)}/{n}")
    print(f"Correctness:  {sum(r['correct'] for r in results)}/{n}")


if __name__ == "__main__":
    main()