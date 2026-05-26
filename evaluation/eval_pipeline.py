"""RAGAS evaluation runner.

Loads test queries from `test_queries.json`, runs each through the LangGraph
pipeline, then scores the (question, answer, contexts, ground_truth) tuples
with RAGAS. Reuses the same HF Inference Provider stack as the rest of the
app — no second API key needed.

Metrics:
- `faithfulness` — does the answer factually align with the retrieved context?
- `answer_relevancy` — does the answer address the question?
- `context_precision` — are the retrieved chunks relevant? (needs ground_truth)
- `context_recall` — does context cover the ground truth? (needs ground_truth)

Without ground_truth, only the first two metrics run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from evaluation.llm_wrapper import HFInferenceChatLLM
from ingestion.embedder import HFInferenceEmbedder
from rag_pipeline.graph import get_graph
from rag_pipeline.llm import DEFAULT_LLM
from vectorstore import Domain

TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"
DEFAULT_TOP_K = 5
# HF free tier has rate limits; keep concurrency low to avoid 429s during eval.
DEFAULT_MAX_WORKERS = 2


def load_test_queries(domain: Optional[Domain] = None) -> dict:
    """Load test queries.

    If `domain` is set, returns the list for that domain; otherwise returns the
    full {domain: [queries]} dict. Empty dict if the file is missing.
    """
    if not TEST_QUERIES_PATH.exists():
        return {} if domain is None else []
    data = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    if domain is None:
        return data
    return data.get(domain.value, [])


def _run_pipeline(domain: Domain, question: str, model: str, top_k: int) -> tuple[str, list[str]]:
    """Returns (answer, list_of_context_texts) for one question."""
    result = get_graph().invoke(
        {
            "question": question,
            "query_mode": "single",
            "domain": domain.value,
            "llm_name": model,
            "top_k": top_k,
        }
    )
    answer = result.get("answer", "")
    contexts = [c["text"] for c in result.get("retrieved", [])]
    return answer, contexts


def evaluate_domain(
    domain: Domain,
    model: str = DEFAULT_LLM,
    judge_model: Optional[str] = None,
    queries: Optional[List[dict]] = None,
    top_k: int = DEFAULT_TOP_K,
    max_queries: Optional[int] = None,
) -> dict:
    """Run RAGAS evaluation for one domain. Returns a JSON-serializable dict.

    `judge_model` defaults to `model` — using the same model for generation and
    judging is biased but cheap; pass a stronger model here for a less biased
    evaluation (you'll pay for two model lineups in HF credits).

    `max_queries` caps how many ground-truth pairs to evaluate. RAGAS runs
    multiple LLM calls per metric per query, so 10 queries × 4 metrics can
    easily mean 80+ HF API calls. Default = all queries; set 2-3 for a fast
    smoke test.
    """
    if queries is None:
        queries = load_test_queries(domain)
    if not queries:
        return {"error": f"No test queries defined for domain '{domain.value}'."}

    if max_queries is not None:
        queries = queries[:max_queries]

    questions: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    ground_truths: list[str] = []
    has_all_ground_truths = True

    for q in queries:
        answer, contexts = _run_pipeline(domain, q["question"], model, top_k)
        questions.append(q["question"])
        answers.append(answer)
        contexts_list.append(contexts)
        gt = (q.get("ground_truth") or "").strip()
        if not gt:
            has_all_ground_truths = False
        ground_truths.append(gt)

    dataset_dict: dict = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts_list,
    }
    # RAGAS 0.2.x renamed the columns; keep legacy aliases for older versions.
    dataset_dict["question"] = questions
    dataset_dict["answer"] = answers
    dataset_dict["contexts"] = contexts_list

    metrics = [faithfulness, answer_relevancy]
    if has_all_ground_truths:
        dataset_dict["reference"] = ground_truths
        dataset_dict["ground_truth"] = ground_truths
        metrics.extend([context_precision, context_recall])

    dataset = Dataset.from_dict(dataset_dict)

    judge_llm = HFInferenceChatLLM(model=judge_model or model)
    embeddings = HFInferenceEmbedder()
    run_config = RunConfig(max_workers=DEFAULT_MAX_WORKERS, max_retries=3, max_wait=120)

    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,  # one bad query shouldn't kill the whole eval
    )

    df = eval_result.to_pandas()
    metric_names = [m.name for m in metrics]

    overall: dict[str, float] = {}
    for name in metric_names:
        if name not in df.columns:
            continue
        values = df[name].dropna()
        if len(values) == 0:
            overall[name] = float("nan")
        else:
            overall[name] = float(values.mean())

    per_query: list[dict] = []
    for i in range(len(questions)):
        row: dict = {
            "question": questions[i],
            "answer": answers[i],
            "context_count": len(contexts_list[i]),
            "ground_truth": ground_truths[i] if ground_truths[i] else None,
        }
        for name in metric_names:
            if name in df.columns:
                val = df[name].iloc[i]
                # Convert any numpy/pandas types to plain python for JSON.
                row[name] = float(val) if val == val else None  # NaN → None
        per_query.append(row)

    return {
        "domain": domain.value,
        "model": model,
        "judge_model": judge_model or model,
        "has_ground_truths": has_all_ground_truths,
        "metrics_run": metric_names,
        "overall": overall,
        "per_query": per_query,
    }
