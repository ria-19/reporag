"""
Evaluation harness for RepoRAG.

Metrics:
  Precision@5      — retrieval signal-to-noise
  Context Recall   — retrieval coverage
  Faithfulness     — generation groundedness (LLM judge)
  Answer Relevance — generation quality (cosine vs ground truth)

Two entry points:
  CLI:  python eval.py --dataset data/qa_pairs.jsonl
  HTTP: GET /eval → api.py calls run_eval(pipeline)

WHY two entry points same function?
  CLI: run during development, see results immediately.
  HTTP: run from frontend or CI without SSH access.
  Same logic, different callers. Don't duplicate.

Output:
  Console: per-query table + aggregate scores
  File:    data/eval_results.json (for RESULTS.md)
"""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.query.pipeline import QueryPipeline, QueryRequest
from src.embedding.embedder import NomicEmbedder
from src.config import settings
from src.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════
# METRICS
# Pure functions — no side effects, easy to unit test.
# ════════════════════════════════════════════════════════

def precision_at_k(
    retrieved_ids: list[str],
    expected_ids:  list[str],
    k: int = 5,
) -> float:
    """
    Fraction of top-k retrieved chunks that are relevant.
    Signal-to-noise ratio of the retrieval step.

    precision_at_5 = |retrieved[:5] ∩ expected| / 5

    Range: [0, 1]. 1.0 = all top-5 are relevant.
    Penalizes returning irrelevant chunks (noise) — quality over quantity.
    """
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(expected_ids)
    return len(top_k & relevant) / k


def context_recall(
    retrieved_ids: list[str],
    expected_ids:  list[str],
) -> float:
    """
    Fraction of expected chunks that were retrieved.
    Coverage of the retrieval step.

    recall = |retrieved ∩ expected| / |expected|

    Range: [0, 1]. 1.0 = all expected chunks were found.
    Penalizes missing relevant chunks (missing info) — coverage over precision.

    WHY both precision and recall?
    Recall alone: return everything → R=1.0 trivially. (context bloat and hallicuinates)
    Precision alone: return only the best → P=1.0 trivially. (not enough info)
    Together: they expose the actual retrieval quality tradeoff.
    """
    if not expected_ids:
        return 0.0
    retrieved = set(retrieved_ids)
    expected  = set(expected_ids)
    return len(retrieved & expected) / len(expected)


def answer_relevance_score(
    answer:       str,
    ground_truth: str,
    embedder:     NomicEmbedder,
) -> float:
    """
    Cosine similarity between embedded answer and ground truth.

    WHY cosine not BLEU/ROUGE?
    BLEU/ROUGE: n-gram overlap — penalizes valid paraphrases.
    Cosine: semantic similarity — paraphrases score high.
    A correct answer in different words = high cosine.
    An off-topic answer = low cosine regardless of word overlap.

    Range: [-1, 1], typically [0.3, 0.95] for good answers.
    Threshold: > 0.7 = semantically equivalent.
    """
    if not answer.strip() or not ground_truth.strip():
        return 0.0

    vecs_a, _ = embedder.embed_batch([answer])
    vecs_g, _ = embedder.embed_batch([ground_truth])

    a = np.array(vecs_a[0])
    g = np.array(vecs_g[0])
    # Both unit vectors → dot product = cosine similarity
    return float(np.dot(a, g))


def faithfulness_score(
    question:  str,
    answer:    str,
    context:   str,
    llm,
) -> float | None:
    """
    LLM judge: is the answer grounded in the retrieved context?

    WHY LLM not cosine?
    Cosine measures semantic similarity, not logical grounding.
    A hallucinated answer can be semantically close to ground truth
    while being entirely unsupported by the context.
    LLM judge reads both answer and context — detects fabrication.

    Returns None on parse failure — logged, not crashed.
    Caller treats None as missing data, not zero.
    """
    prompt = f"""\
You are a skeptical, high-precision Technical Auditor. Your task is to verify "Faithfulness": 
Can every individual claim in the Answer be derived EXCLUSIVELY from the Context?

Context:
{context[:2000]}

Question: {question}

Answer: {answer}

INSTRUCTIONS:
1. Extract every distinct factual claim made in the Answer.
2. For each claim, check if the Context provides direct evidence for it.
3. If a claim relies on outside knowledge (even if true in the real world), mark it as UNFAITHFUL.
4. Calculate the score as: (Number of Supported Claims) / (Total Number of Claims).

OUTPUT FORMAT:
Reasoning: <brief bullet points of support vs contradictions>
Score: <0.0 to 1.0>

Only output the Reasoning and the Score."""

    try:
        raw, _ = llm.generate(prompt)
        match = re.search(r"Score:\s*([\d\.]+)", raw)
        if match:
            return float(raw.strip())
        
        # Fallback: just try to find any float in the text
        match = re.search(r"([\d\.]+)", raw.split("Score:")[-1])
        return float(match.group(1))

    except (ValueError, TypeError):
        logger.warning(
            "Faithfulness judge returned non-numeric: %s", raw[:50]
        )
        return None


# ════════════════════════════════════════════════════════
# EVAL RUNNER
# ════════════════════════════════════════════════════════

def run_eval(
    pipeline:         QueryPipeline,
    dataset_path:     str  = "data/qa_pairs.jsonl",
    output_path:      str  = "data/eval_results.json",
    run_faithfulness: bool = True,
) -> dict[str, Any]:
    """
    Run full evaluation against golden dataset.

    Args:
        pipeline:         QueryPipeline instance (injected)
        dataset_path:     path to qa_pairs.jsonl
        output_path:      where to write results JSON
        run_faithfulness: False to skip LLM judge (faster)

    Returns:
        dict with aggregate scores + per-query breakdown
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {dataset_path}\n"
            f"Generate it first: python scripts/generate_golden_dataset.py"
        )

    # Load golden dataset
    pairs = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if not pairs:
        raise ValueError(f"Empty dataset: {dataset_path}")

    logger.info("Running eval on %d QA pairs", len(pairs))

    # Embedder for answer relevance — shared, loaded once
    # Pipeline shouldn't expose its embedder publicly.
    # WHY use pipeline's embedder directly?
    # Tradeoff: instantiate separately. Costs ~2s + 500MB.
    # Here: expose pipeline.embedder
    llm      = pipeline.llm   # Issue: Self-Referential Bias but cost + latency optimized, semantic alignment
    embedder = pipeline.embedder

    per_query_results = []
    start_total = time.perf_counter()

    def evaluate_single_pair(indexed_pair):
        i, pair = indexed_pair

        question         = pair["question"]
        expected_ids     = pair["expected_chunk_ids"]
        ground_truth     = pair["ground_truth_answer"]

        logger.info("[%d/%d] Evaluating: %s", i + 1, len(pairs), question[:60])

        # Run pipeline
        try:
            result = pipeline.query(QueryRequest(question=question))
        except Exception as e:
            logger.error("Pipeline failed on query %d: %s", i, e)
            per_query_results.append({
                "question":         question,
                "error":            str(e),
                "precision_at_5":   0.0,
                "context_recall":   0.0,
                "answer_relevance": 0.0,
                "faithfulness":     None,
            })
            continue

        # Extracted retrieved chunk IDs
        retrieved_ids = [rc.chunk.chunk_id for rc in result.sources]

        # ── Metric 1: Precision@5 ─────────────────────
        p_at_5 = precision_at_k(
            retrieved_ids=retrieved_ids,
            expected_ids=expected_ids,
            k=5,
        )

        # ── Metric 2: Context Recall ──────────────────
        c_recall = context_recall(
            retrieved_ids=retrieved_ids,
            expected_ids=expected_ids,
        )

        # ── Metric 3: Answer Relevance ────────────────
        a_relevance = answer_relevance_score(
            answer=result.answer,
            ground_truth=ground_truth,
            embedder=embedder,
        )

        # ── Metric 4: Faithfulness ────────────────────
        faith = None
        if run_faithfulness:
            # Build context string from sources for the judge
            context_for_judge = "\n\n---\n\n".join(
                rc.chunk.text for rc in result.sources
            )
            faith = faithfulness_score(
                question=question,
                answer=result.answer,
                context=context_for_judge,
                llm=llm,
            )

        result_dict = {
            "question":         question,
            "precision_at_5":   round(p_at_5,     3),
            "context_recall":   round(c_recall,    3),
            "answer_relevance": round(a_relevance, 3),
            "faithfulness":     round(faith, 3) if faith is not None else None,
            "route":            result.route.value,
            "latency_ms":       round(result.metrics.total_latency_ms, 1),
            "chunks_retrieved": result.metrics.chunks_retrieved,
        }

        # Live progress
        print(
            f"  P@5={p_at_5:.2f} "
            f"Recall={c_recall:.2f} "
            f"Relevance={a_relevance:.2f} "
            f"Faith={faith:.2f if faith else 'N/A'} "
            f"| {question[:50]}"
        )

        return result_dict

    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(evaluate_single_pair, enumerate(pairs)))
    
    per_query_results.extend(results)

    # ── Aggregate scores ──────────────────────────────
    def mean(values: list[float | None]) -> float:
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 3) if clean else 0.0

    aggregate = {
        "n_queries":        len(pairs),
        "precision_at_5":   mean([r["precision_at_5"]   for r in per_query_results]),
        "context_recall":   mean([r["context_recall"]    for r in per_query_results]),
        "answer_relevance": mean([r["answer_relevance"]  for r in per_query_results]),
        "faithfulness":     mean([r["faithfulness"]      for r in per_query_results]),
        "avg_latency_ms":   mean([r.get("latency_ms")    for r in per_query_results]),
        "elapsed_seconds":  round(time.perf_counter() - start_total, 1),
    }

    output = {
        "aggregate":  aggregate,
        "per_query":  per_query_results,
    }

    # Write results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print("\n" + "═" * 55)
    print("EVAL RESULTS")
    print("═" * 55)
    print(f"  Queries evaluated : {aggregate['n_queries']}")
    print(f"  Precision@5       : {aggregate['precision_at_5']}")
    print(f"  Context Recall    : {aggregate['context_recall']}")
    print(f"  Answer Relevance  : {aggregate['answer_relevance']}")
    print(f"  Faithfulness      : {aggregate['faithfulness']}")
    print(f"  Avg latency       : {aggregate['avg_latency_ms']}ms")
    print(f"  Total time        : {aggregate['elapsed_seconds']}s")
    print("═" * 55)
    print(f"  Full results → {output_path}")

    return output


# ════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RepoRAG pipeline")
    parser.add_argument(
        "--dataset",
        default="data/qa_pairs.jsonl",
        help="Path to golden dataset",
    )
    parser.add_argument(
        "--output",
        default="data/eval_results.json",
        help="Where to write results JSON",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Skip LLM faithfulness judge (faster)",
    )
    args = parser.parse_args()

    setup_logging()

    # Build pipeline for CLI usage
    # WHY rebuild here and not reuse api.py lifespan?
    # CLI has no FastAPI app. We build the minimum needed.
    from src.embedding.embedder import NomicEmbedder
    from src.storage.lance_store import LanceStore
    from src.storage.kuzu_store import KuzuStore
    from src.query.validator import CosineValidator, LLMJudge
    from src.llm import build_llm

    embedder    = NomicEmbedder(device=settings.embedding_device)
    lance       = LanceStore(db_path=settings.lance_db_path)
    kuzu        = KuzuStore(db_path=settings.kuzu_db_path)
    llm         = build_llm()
    cosine_val  = CosineValidator(embedder=embedder)
    llm_judge   = LLMJudge(llm=llm, sample_rate=1.0)
    # sample_rate=1.0 for eval — judge every query
    # WHY 1.0 here but 0.1 in production?
    # Eval is offline — we WANT to judge every query.
    # Production is online — we sample to save tokens.

    pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=lance,
        graph_store=kuzu,
        llm=llm,
        cosine_validator=cosine_val,
        llm_judge=llm_judge,
    )

    run_eval(
        pipeline=pipeline,
        dataset_path=args.dataset,
        output_path=args.output,
        run_faithfulness=not args.no_faithfulness,
    )


