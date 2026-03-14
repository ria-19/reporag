# RepoRAG

Local-first codebase intelligence. Ask questions about any GitHub repository in natural language and get answers grounded in the actual source code.

**Not a code editor. Not a copilot.** A system for understanding massive codebases — onboarding, open-source contribution, structural questions.

```
POST /query
{
  "question": "how does dependency injection work?",
  "repo_name": "fastapi"
}

→ {
  "answer": "FastAPI's DI system works by...",
  "sources": [{"file_path": "fastapi/dependencies/utils.py", ...}],
  "metrics": {"total_latency_ms": 840, "chunks_retrieved": 12, ...}
}
```

Every response includes per-step latency, token counts, and retrieval scores. No magic.

---

## What It Does

1. **Indexes a GitHub repo** — clones (shallow), parses with tree-sitter, embeds with nomic-embed-text-v1.5, writes to LanceDB (vector) + KuzuDB (graph)
2. **Answers questions** — hybrid retrieval (vector + BM25) → graph expansion → LLM generation → faithfulness validation
3. **Measures itself** — Precision@5, Context Recall, Answer Relevance, Faithfulness score on a golden dataset

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Parser | tree-sitter >= 0.22 | 40+ languages, single API, binary wheels |
| Embedding | nomic-embed-text-v1.5 (768d MRL) | Matryoshka dims — two vectors, one forward pass |
| Vector + BM25 | LanceDB | Embedded Rust, hybrid search, IVF-PQ, no server |
| Graph | KuzuDB | Embedded, columnar, SIMD, Cypher |
| Merge | RRF (Reciprocal Rank Fusion) | Rank-based, no score normalization needed |
| LLM | Ollama (local) or Gemini (cloud) | Config switch, no code change |
| API | FastAPI (sync handlers) | Sync for CPU-heavy paths, ThreadPool managed |

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com) installed and running (for local LLM)
- `git` on PATH

### Install

```bash
git clone <this-repo>
cd reporag

# Create virtualenv and install all dependencies
uv venv
source .venv/bin/activate
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env — defaults work for local dev with Ollama
```

Key settings:
```bash
LLM_PROVIDER=ollama       # or: gemini
LLM_MODEL=llama3.2        # any model pulled in Ollama
EMBEDDING_DEVICE=cpu      # or: cuda, mps
```

For Gemini:
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
```

### Pull Ollama model (if using local LLM)

```bash
ollama pull llama3.2
```

The API will auto-pull on first startup if the model is missing — but pulling manually avoids a long wait on first request.

---

## Running

```bash
# Start the API
uv run uvicorn api:app --reload --port 8000

# Verify it's alive
curl http://localhost:8000/health
```

---

## Usage

### Index a repository

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/tiangolo/fastapi",
    "repo_name": "fastapi",
    "force": false
  }'
```

Response:
```json
{
  "repo": "fastapi",
  "files_processed": 87,
  "files_skipped": 0,
  "chunks_indexed": 1243,
  "elapsed_seconds": 142.3
}
```

`force: true` clears the WAL and re-indexes from scratch.

### Check what's indexed

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "lance": {"chunks_indexed": 1243, "table_name": "chunks"},
  "kuzu":  {"nodes": 1243, "edges_calls": 847, "edges_imports": 0}
}
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "how does dependency injection work?",
    "repo_name": "fastapi"
  }'
```

With filters:
```bash
# Exact symbol lookup
curl -X POST http://localhost:8000/query \
  -d '{
    "question": "what does solve_shared do?",
    "repo_name": "fastapi",
    "symbol": "solve_shared"
  }'

# File filter
curl -X POST http://localhost:8000/query \
  -d '{
    "question": "how is routing set up?",
    "repo_name": "fastapi",
    "filename": "routing"
  }'

# Language filter
curl -X POST http://localhost:8000/query \
  -d '{
    "question": "how are requests validated?",
    "repo_name": "fastapi",
    "language": "python"
  }'
```

Response:
```json
{
  "question": "how does dependency injection work?",
  "answer": "FastAPI's DI system inspects function signatures...",
  "route": "conceptual",
  "sources": [
    {
      "file_path": "fastapi/dependencies/utils.py",
      "symbol_name": "solve_dependencies",
      "chunk_type": "function",
      "start_line": 142,
      "end_line": 198,
      "snippet": "async def solve_dependencies(...",
      "rrf_score": 0.043,
      "source": "hybrid"
    }
  ],
  "metrics": {
    "total_latency_ms": 840,
    "chunks_retrieved": 8,
    "tokens_used": 1240,
    "embed_query": {"latency_ms": 45, "output_count": 2},
    "hybrid_search": {"latency_ms": 120, "output_count": 5},
    "graph_expand": {"latency_ms": 18, "output_count": 12},
    "llm_generate": {"latency_ms": 620, "tokens_used": 1240}
  },
  "faithfulness": 0.82,
  "answer_relevance": null
}
```

`answer_relevance` is `null` when LLM judge was not sampled this request (10% sample rate in production).

### Run evaluation

```bash
# Generate golden dataset first
uv run python scripts/generate_golden_dataset.py \
  --repo fastapi \
  --n 20 \
  --output data/qa_pairs.jsonl

# Run eval
uv run python eval.py \
  --dataset data/qa_pairs.jsonl \
  --output data/eval_results.json
```

Or via HTTP:
```bash
curl http://localhost:8000/eval
```

---

## Architecture

### Write Path (Indexing)

```
GitHub URL
  → git clone --depth=1 (temp dir, auto-cleaned)
  → LocalRepoLoader.stream_files() → RawFile (one at a time, constant RAM)
  → WAL check: skip if already indexed (resume on crash)
  → CodeParser.parse(raw_file) → CodeChunk[], RawEdge[]
      Python: tree-sitter recursive walk, extracts functions/classes/imports
      JS/TS:  same, plus JSDoc extraction
      Other:  FallbackStrategy (line-based chunking, no symbol extraction)
  → Batch accumulate (batch_size=32 by default)
  → NomicEmbedder.embed_batch() → (vector_768[], vector_128[])
      One forward pass, two vectors via MRL truncation
  → LanceStore.add_chunks() — merge_insert on chunk_id (idempotent)
  → KuzuStore.add_nodes() — MERGE on chunk_id (idempotent)
  → WAL.record("file_indexed", repo=repo_name, path=...)
  
  [Pass 2 — after all files]
  → resolve_edges(repo_name, all_raw_edges, name_map)
      Matches call targets to known chunk_ids
      Unresolved = external library = safely dropped
  → KuzuStore.add_edges() — MERGE (idempotent, silent skip if node missing)
  → WAL.record("edges_written", repo=repo_name)
```

### Read Path (Query)

```
QueryRequest (question, repo_name, symbol?, filename?, language?, k=5)
  → Step 1: Parse filters (already validated by Pydantic)
  → Step 2: Exact symbol lookup (if --symbol provided)
      If > 10 results: too ambiguous, fall through to hybrid search
  → Step 3: Route query → prompt template
      BoW keyword matching: debugging > setup > code_search > conceptual
      Default on zero matches: conceptual
  → Step 4: Embed query (768d + 128d, QUERY_PREFIX)
  → Step 5: Hybrid search (if symbol lookup didn't find enough)
      Stage 1: vector_128 IVF-PQ ANN → top 1000 candidates (fast)
      Stage 2: vector_768 flat numpy cosine on 1000 candidates (precise)
      BM25: FTS on search_content column
      Merge: RRF(stage2_ranks, bm25_ranks) → top k
  → Step 6: Graph expand (single-hop, max 20 neighbors)
      MATCH (a)-[:CALLS|IMPORTS]->(b) WHERE a.chunk_id IN [seeds]
  → Step 7: Fetch neighbor text from LanceDB by chunk_ids
  → Step 8: Merge + deduplicate (seeds + neighbors)
  → Step 9: Assemble context
      Interleave: retrieved chunks at attention peaks (start/end)
      Graph neighbors in middle
      Cap at 32,000 chars
  → Step 10: Generate
      prompt_template.format(context=..., question=...)
      OllamaLLM or GeminiLLM
  → Step 11: Validate
      CosineValidator: always (cosine(embed(answer), embed(context)))
      LLMJudge: 10% sample (faithfulness + answer relevance)
  → QueryResult (answer, sources, metrics, faithfulness, answer_relevance)
```

### Multi-Tenant Isolation

All data is namespaced by `repo_name`:

| Layer | Isolation mechanism |
|---|---|
| chunk_id | `{repo}::{safe_path}::{class}.{func}::{hash[:8]}` |
| WAL keys | `op::repo::path=value` |
| LanceDB | `repo_name` column + `WHERE repo_name = $repo` on all searches |
| KuzuDB | `repo_name` property on every Chunk node + enforced in all MATCH patterns |

### MRL Cascade (Two-Stage Retrieval)

```
Index time:
  Store vector_768 (768d, precise) + vector_128 (128d, fast) per chunk
  One forward pass — truncate + re-normalize for 128d

Query time:
  Stage 1: IVF-PQ ANN on vector_128 → top 1000 candidates (fast)
  Stage 2: flat numpy cosine on vector_768 for candidates only (exact, ~microseconds)
  
Cost: quality close to full 768d, speed close to 128d
```

---

## Repo Structure

```
reporag/
├── src/
│   ├── core/
│   │   ├── models.py          Domain models: CodeChunk, GraphEdge, QueryResult...
│   │   └── ports.py           Write-path Protocols
│   ├── ingestion/
│   │   └── loaders.py         LocalRepoLoader, GitHubRepoLoader
│   ├── chunking/
│   │   ├── parser.py          CodeParser, PythonStrategy, JavaScriptStrategy, FallbackStrategy
│   │   └── graph.py           resolve_edges()
│   ├── embedding/
│   │   └── embedder.py        NomicEmbedder, MRL cascade
│   ├── storage/
│   │   ├── lance_store.py     LanceStore — hybrid search, upsert, FTS
│   │   └── kuzu_store.py      KuzuStore — graph nodes, edges, expansion
│   ├── query/
│   │   ├── pipeline.py        QueryPipeline — 10-step read path
│   │   ├── router.py          QueryRouter (BoW), ContextAssembler
│   │   └── validator.py       CosineValidator, LLMJudge
│   ├── observability/
│   │   └── metrics.py         Timer (context manager), StepMetrics
│   ├── llm.py                 OllamaLLM, GeminiLLM, build_llm()
│   ├── indexer.py             Indexer, WAL
│   ├── config.py              Settings (pydantic-settings, all env vars)
│   └── logger.py              setup_logging(), get_logger()
├── scripts/
│   └── generate_golden_dataset.py  Golden QA pair generation
├── data/
│   ├── qa_pairs.jsonl         Generated golden dataset
│   └── eval_results.json      Eval output
├── api.py                     FastAPI app, lifespan, 4 endpoints
├── eval.py                    Evaluation harness + CLI entry point
├── DECISIONS.md               All major architectural decisions
├── FAILURES.md                All bugs found and case studies
├── RESULTS.md                 Eval results (fill after running eval)
├── pyproject.toml
└── .env.example
```

---

## Eval Metrics

| Metric | Formula | What it catches |
|---|---|---|
| Precision@5 | `\|retrieved[:5] ∩ expected\| / 5` | Retrieval noise |
| Context Recall | `\|retrieved ∩ expected\| / \|expected\|` | Missing relevant chunks |
| Answer Relevance | `cosine(embed(answer), embed(ground_truth))` | Off-topic answers |
| Faithfulness | LLM judge: is answer grounded in context? | Hallucination |

Diagnosing failures:
- **Low Precision + Low Recall** → retrieval problem (embedding quality, BM25 config, chunking)
- **High Precision + Low Faithfulness** → generation problem (prompt, model, context assembly)
- **Low Answer Relevance** → routing problem (wrong prompt template selected)

---

## Limitations

- **IMPORTS graph not implemented** — only CALLS edges. File-level dependency traversal is V2.
- **Languages:** Python, JS, TypeScript only. Other languages chunked as plaintext (no symbol extraction).
- **Query router ~90% accuracy** — BoW fails on ambiguous queries. LLM-based routing would add +500ms.
- **Reranker not implemented** — graph neighbors ranked only by retrieval order, not query relevance.
- **WAL is single-process** — `threading.Lock()` protects threads; not safe for `uvicorn --workers N`.

See `FAILURES.md` for full details on each limitation and the V2 upgrade path.

---

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src/

# Tests (when written)
uv run pytest
```

---

## Design Philosophy

**Transparent over magical.** Every query shows exactly what was retrieved, why, and how long each step took.

**Measurable over assumed.** Nothing is added unless eval shows it helps. The reranker is not in v1 — it will be added only if faithfulness score proves it's needed.

**Embedded over networked.** LanceDB and KuzuDB run in-process. No servers to manage, no network latency, no auth.

**Local-first.** Works fully offline with Ollama. Cloud LLM is opt-in, not required.