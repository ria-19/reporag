# RESULTS.md

Evaluation results from running `eval.py` against generated golden datasets.

---

## Baseline — fastapi (to be filled)

**Repo:** https://github.com/tiangolo/fastapi  
**Index date:** —  
**Chunks indexed:** —  
**Eval dataset:** `data/qa_pairs.jsonl` (n=20)  
**LLM:** llama3.2 (local, Ollama)  
**Embedding:** nomic-embed-text-v1.5, cpu  

| Metric | Score | Notes |
|---|---|---|
| Precision@5 | — | |
| Context Recall | — | |
| Answer Relevance | — | |
| Faithfulness | — | |
| Avg latency (ms) | — | |

```
# Run to generate:
uv run python scripts/generate_golden_dataset.py --repo fastapi --n 20
uv run python eval.py --dataset data/qa_pairs.jsonl
```

---

## Baseline — micrograd (to be filled)

**Repo:** https://github.com/karpathy/micrograd  
**Chunks indexed:** ~35  
**Note:** Small repo — uses flat search (IVF-PQ skipped, < 256 chunks)

| Metric | Score | Notes |
|---|---|---|
| Precision@5 | — | |
| Context Recall | — | |
| Answer Relevance | — | |
| Faithfulness | — | |
| Avg latency (ms) | — | |

---

## Retrieval Ablation (planned)

Comparing retrieval strategies on same golden dataset:

| Strategy | Precision@5 | Context Recall | Notes |
|---|---|---|---|
| BM25 only | — | — | |
| Vector (768d) only | — | — | |
| Hybrid (vector + BM25 + RRF) | — | — | Current system |
| Hybrid + graph expand | — | — | Current system with graph |
| Hybrid + graph + reranker | — | — | V2 — add if Faithfulness low |

---

## Reranker Decision Gate

Add `bge-reranker-v2-m3` reranker **only if**:

- Faithfulness score < 0.70 consistently across eval runs, **and**
- Baseline hybrid + graph already achieving Precision@5 > 0.50

Current status: **not yet evaluated.**

If added: measure +latency cost (expected +150-250ms on CPU) vs faithfulness gain.
Record here before merging.

---

## Known Performance Characteristics (qualitative, pre-eval)

**Strong cases (expected high scores):**
- "How does X work?" on well-named, documented Python functions
- Exact symbol queries: `--symbol calculate_loss` → direct fetch, skips ANN
- Small repos (< 500 chunks): flat search = 100% accurate retrieval

**Weak cases (expected lower scores):**
- Cross-file reasoning requiring 2+ hop graph traversal (single-hop only)
- Queries about JS/TS-heavy codebases (same parser but less symbol richness)
- Ambiguous routing: "why is this slow?" (could be debugging or conceptual)
- Any query needing IMPORTS edges (not implemented in v1)

---

## Latency Budget (target)

| Step | Target | Notes |
|---|---|---|
| Embed query | < 50ms | CPU, nomic 768d |
| Hybrid search (Stage 1 + 2 + BM25 + RRF) | < 200ms | LanceDB, flat < 256 chunks |
| Graph expand | < 30ms | KuzuDB, single-hop |
| LLM generate | < 800ms | llama3.2 local, depends on output length |
| **Total** | **< 1100ms** | p50 target |