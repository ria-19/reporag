"""
Bootstrap golden dataset from indexed repo.

Strategy: sample high-degree nodes from call graph,
generate QA pairs using a strong LLM, write to jsonl.

WHY high-degree nodes?
  Functions with many callees are architecturally important.
  Questions about them have richer expected context
  (the callee chunks are natural expected_chunk_ids).
  Low-degree nodes (leaf functions) have thin context —
  questions about them are less useful for eval.

Output: data/qa_pairs.jsonl
Each line: {question, expected_chunk_ids, expected_context, ground_truth_answer}

Usage:
  python scripts/generate_golden_dataset.py \
    --repo myrepo \
    --n 30 \
    --model gemini

Golden dataset generation bias:
  QA pairs are generated from high-degree nodes only.
  This means eval measures performance on orchestrator
  functions — not leaf functions, not setup code, not
  error handling paths.
  The model that generates questions may ask questions
  it can answer from the same context window — easier
  than questions a real developer would ask.
  
  Mitigation: manually review 5-10 pairs before running eval.
  Look for: questions that are too obvious, answers that
  are too long, questions that reference code not in context.
  
  Real fix: hire 2-3 developers to write 10 pairs each
  from their actual onboarding experience.
  Not doing this for v1 — YAGNI.

"""

from __future__ import annotations

import json
import argparse
import random
from pathlib import Path

from src.storage.lance_store import LanceStore
from src.storage.kuzu_store import KuzuStore
from src.llm import build_llm
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# Prompt for QA generation
# Constraints are tight — we want questions that:
# 1. A real developer would ask during onboarding
# 2. Are answerable from ONLY the provided context
# 3. Have a specific, verifiable answer

_GENERATION_PROMPT = """\
You are generating evaluation data for a code search system.

Here is a Python/JavaScript function and its direct callees:

{context}

Write ONE question a developer would ask while onboarding to this codebase.
The question must be:
- Answerable using ONLY the provided code above
- Specific (not "what does this do?" but "how does X handle Y?")
- About behavior, flow, or structure — not syntax

Then write the ground truth answer in 2-4 sentences.

Respond in this exact JSON format with no other text:
{{
  "question": "...",
  "ground_truth_answer": "..."
}}"""


def get_high_degree_chunks(
    kuzu: KuzuStore,
    lance: LanceStore,
    n: int,
    min_degree: int = 2,
) -> list[dict]:
    """
    Find chunks with high out-degree in call graph.
    These are the most interesting for QA generation.

    Returns list of {chunk_id, symbol_name, file_path, degree}
    sorted by degree descending.
    """

    # Query KuzuDB for out-degree of each node
    # WHY out-degree not in-degree?
    # Out-degree = how many things this function calls.
    # High out-degree = orchestrator function = rich context.
    # In-degree = how many things call this function.
    # High in-degree = utility function = thin context.
    result = kuzu._conn.execute("""
        MATCH (a:Chunk)-[:CALLS]->(b:Chunk)
        RETURN a.chunk_id    AS chunk_id,
               a.symbol_name AS symbol_name,
               a.file_path   AS file_path,
               count(b)      AS degree
        HAVING degree >= $min_degree
        ORDER BY degree DESC
        LIMIT $limit
    """, {"min_degree": min_degree, "limit": n * 3})

    # Fetch 3x because some will fail generation — oversample
    rows = result.fetchall()
    if not rows:
        logger.warning(
            "No chunks with degree >= %d found. "
            "Is the graph populated? Try --min-degree 1",
            min_degree,
        )
        return []

    return [
        {
            "chunk_id":    row[0],
            "symbol_name": row[1],
            "file_path":   row[2],
            "degree":      row[3],
        }
        for row in rows
    ]


def build_context_for_chunk(
    chunk_id: str,
    lance: LanceStore,
    kuzu: KuzuStore,
) -> tuple[str, list[str]]:
    """
    Fetch chunk text + direct callee text.
    Returns (context_string, all_chunk_ids_in_context).
    """
    
    # Fetch the anchor chunk
    anchor_chunks = lance.fetch_by_ids([chunk_id])
    if not anchor_chunks:
        return "", []

    anchor = anchor_chunks[0]

    # Fetch 1-hop neighbors
    neighbor_ids = kuzu.expand_neighbors(
        chunk_ids=[chunk_id],
        limit=5,   # cap at 5 neighbors for context size
    )
    neighbor_chunks = lance.fetch_by_ids(neighbor_ids) if neighbor_ids else []

    # Build context string
    parts = []
    all_ids = [chunk_id]

    # Anchor first — it's the most important
    parts.append(
        f"# {anchor.file_path} — {anchor.symbol_name}\n{anchor.text}"
    )

    for nc in neighbor_chunks:
        parts.append(
            f"# {nc.file_path} — {nc.symbol_name}\n{nc.text}"
        )
        all_ids.append(nc.chunk_id)

    context = "\n\n---\n\n".join(parts)
    return context, all_ids


def generate_qa_pair(
    context: str,
    all_ids: list[str],
    llm,
    anchor_chunk,
) -> dict | None:
    """
    Call LLM to generate one QA pair from context.
    Returns None if generation fails or JSON is malformed.
    """
    prompt = _GENERATION_PROMPT.format(context=context[:4000])

    try:
        raw, _ = llm.generate(prompt)
        # Strip markdown fences if model adds them
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean)

        return {
            "question":            parsed["question"],
            "expected_chunk_ids":  all_ids,
            "expected_context":    context[:2000],
            "ground_truth_answer": parsed["ground_truth_answer"],
            # Metadata — not used in eval, useful for debugging
            "_anchor_symbol": anchor_chunk.symbol_name,
            "_anchor_file":   anchor_chunk.file_path,
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("QA generation failed: %s | raw: %s", e, raw[:100])
        return None


def run(repo_name: str, n: int, output_path: str) -> None:
    lance = LanceStore(db_path=settings.lance_db_path)
    kuzu  = KuzuStore(db_path=settings.kuzu_db_path)
    llm   = build_llm()

    logger.info("Finding high-degree nodes...")
    candidates = get_high_degree_chunks(kuzu, lance, n=n)

    if not candidates:
        logger.error("No candidates found. Exiting.")
        return

    logger.info("Generating %d QA pairs from %d candidates...", n, len(candidates))

    pairs = []
    for candidate in candidates:
        if len(pairs) >= n:
            break

        context, all_ids = build_context_for_chunk(
            chunk_id=candidate["chunk_id"],
            lance=lance,
            kuzu=kuzu,
        )
        if not context:
            continue

        anchor_chunks = lance.fetch_by_ids([candidate["chunk_id"]])
        if not anchor_chunks:
            continue

        pair = generate_qa_pair(
            context=context,
            all_ids=all_ids,
            llm=llm,
            anchor_chunk=anchor_chunks[0],
        )
        if pair:
            pairs.append(pair)
            logger.info(
                "[%d/%d] Generated: %s",
                len(pairs), n, pair["question"][:60]
            )

    # Write to jsonl — one JSON object per line
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(
        "Wrote %d QA pairs to %s",
        len(pairs), output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",   required=True)
    parser.add_argument("--n",      type=int, default=30)
    parser.add_argument("--output", default="data/qa_pairs.jsonl")
    args = parser.parse_args()

    run(repo_name=args.repo, n=args.n, output_path=args.output)
