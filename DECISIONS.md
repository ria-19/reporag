# DECISIONS.md

Architectural and engineering decisions made during RepoRAG's design and build.
Each entry: the decision, the reasoning, and the trade-off accepted.

---

## Technology Stack

### Parser: tree-sitter >= 0.22
**Decision:** Use tree-sitter with binary wheels (no compilation step).  
**Why:** 40+ language grammars, single consistent API, fast C parser, works on any platform via pre-built wheels. Evaluated regex-based parsing — rejected because it cannot handle nested structures, decorators, or method-class hierarchy correctly.  
**Trade-off:** tree-sitter queries give you nodes, not traversal context. We use explicit recursive walks for Python/JS (not tree-sitter Query API) so we can thread `parent_class` and `chunk_id` context down the tree. Query API used only for import extraction where patterns are uniform.

### Embedding: nomic-embed-text-v1.5 (768d, MRL)
**Decision:** Use Matryoshka Representation Learning model that supports valid truncation.  
**Why:** Enables the two-stage MRL cascade — store 768d for precision, use 128d for fast ANN candidate retrieval. Both vectors produced from one forward pass. Task prefixes (`search_document:` / `search_query:`) required and enforced.  
**Trade-off:** `trust_remote_code=True` required. Model card must be read to understand correct normalization sequence (encode → truncate → re-normalize; LayerNorm only needed if using raw HuggingFace API, not sentence-transformers).

### Vector Store: LanceDB (embedded)
**Decision:** LanceDB over FAISS + JSON metadata + separate BM25 index.  
**Why:** FAISS stores vectors only — no metadata, no BM25, no persistence API. LanceDB is Rust-native, embedded (no server), Apache Arrow columnar, supports hybrid search via tantivy FTS, and supports IVF-PQ indexing. One dependency replacing three systems.  
**Trade-off:** Schema is defined once — changing it requires re-indexing. Dynamic index threshold (min 256 vectors for IVF-PQ) added to avoid KMeans crash on small repos.

### Graph Store: KuzuDB (embedded)
**Decision:** KuzuDB over Neo4j, NetworkX, or SQLite adjacency table.  
**Why:** Neo4j requires a server process. NetworkX is in-memory only (no persistence). SQLite adjacency table means manual joins in SQL, no graph query language. KuzuDB is embedded, columnar-storage, SIMD-vectorized, and speaks Cypher. Free.  
**Trade-off:** KuzuDB path must be a directory, not a file (v0.8.0+). `QueryResult.fetchall()` not `.fetchall()` on raw result (API differs from PEP-249). DDL must be split on `;` and executed one statement at a time.

### Hybrid Search: RRF (Reciprocal Rank Fusion)
**Decision:** RRF with `rrf_k=60` over learned score normalization.  
**Why:** Rank-based fusion requires no score normalization across heterogeneous sources (vector cosine ≠ BM25 score). rrf_k=60 is the standard constant from the 2009 Cormack et al. paper, empirically robust across datasets. Adding learned weights would require a training set we don't have.  
**Trade-off:** Equal weight to vector and BM25 signals — no tuning. For code retrieval this is acceptable: BM25 dominates on exact symbol names, vector dominates on paraphrase queries.

### Reranker: Deferred
**Decision:** Ship without bge-reranker-v2-m3. Add only if eval shows need.  
**Why:** Hybrid search (vector + BM25) + RRF already strong for code. BM25 is unusually powerful for code because function names are exact keywords. Reranker's primary value is re-scoring graph-expanded neighbors that scored zero in retrieval — but faithfulness eval will tell us if that's actually hurting generation.  
**Trade-off:** Graph neighbors appended after retrieval hits with no quality ranking. If faithfulness score is low, add reranker then. "I evaluated whether the reranker was necessary, measured the gap, then added it" is stronger engineering than cargo-culting it.

### LLM: Ollama (local) / Gemini (cloud)
**Decision:** Config-switched via `LLM_PROVIDER` env var, Strategy Pattern.  
**Why:** Local-first is the product promise. Cloud option for users without GPU. Both wrapped behind `LLMPort` Protocol — `pipeline.py` never imports either directly.  
**Self-healing:** `_ensure_model_ready()` injected into `OllamaLLM.__init__()`. Verifies model weights exist on disk; if not, pulls them before returning. Blocks startup. Prevents silent 500 errors where server boots but model is missing. Uses `m.model` not `m['name']` (SDK schema changed post-training).

---

## Architecture

### Dependency Injection Throughout
**Decision:** Every component receives its dependencies via constructor. Nothing instantiates `LanceStore`, `KuzuDB`, or `NomicEmbedder` internally.  
**Why:** Open/Closed Principle. Swapping a backend means passing a different object — `indexer.py` and `pipeline.py` never change. Makes unit testing trivial: pass mock stores, no real DB needed.

### sync def Handlers in FastAPI
**Decision:** All four API handlers are `def`, not `async def`.  
**Why:** `async def` with blocking I/O (git clone, sentence-transformers, LanceDB, KuzuDB, Ollama) blocks the event loop — no other requests served while that call runs. FastAPI automatically moves `def` handlers to a ThreadPoolExecutor. The event loop stays free for health checks and lightweight requests.  
**Trade-off:** Default ThreadPool is 40 threads. 40+ concurrent `/index` requests would queue. Acceptable for a local system. Production fix: tune `--workers` or use custom executor.

### Two-Pass Indexing (Write Path)
**Decision:** Parse and embed in Pass 1 (streaming). Resolve graph edges in Pass 2 (batch).  
**Why:** Edges require both endpoints to exist as nodes before they can be inserted. Pass 1 builds the name_map (symbol → chunk_ids) needed for resolution. Pass 2 uses it.  
**Trade-off:** All raw_edges accumulate in RAM between passes. Estimated 3MB for a 10k-function repo. Acceptable for local.

### WAL (Write-Ahead Log) for Resumable Indexing
**Decision:** JSONL append-only file, per-file completion records, scoped per repo.  
**Why:** Indexing a large repo can take 10+ minutes. Crashes are real. WAL allows resume from last successful file, avoiding redundant embedding computation.  
**Key design:** WAL records completion AFTER both stores confirm write (not before). Crash during flush → file not marked done → retry on resume. Both LanceDB (merge_insert) and KuzuDB (MERGE) are idempotent — safe to retry.  
**Threading:** `threading.Lock()` protects file writes. Single uvicorn worker = threads, not processes. If `--workers N` is ever added, upgrade to `filelock` library for cross-process safety.  
**Key:** `op::repo::path=value` format for complete namespace isolation across repos.

### File-Level WAL Granularity (Not Chunk-Level)
**Decision:** WAL records one entry per file, not per chunk.  
**Why:** Chunk-level WAL would add thousands of entries per file with more I/O overhead than the retry cost. Files are the natural atomic unit: parsed together, embedded together, written together.  
**Large file exception:** Files with more chunks than `batch_size` are flushed in sub-batches, but WAL is recorded only after ALL sub-batches for that file succeed. Whole file is the WAL unit regardless.  
**Upgrade path if needed:** Add `chunk_offset` field to WAL entry to resume mid-file. Not built — YAGNI.

### MRL Cascade (Two-Stage Retrieval)
**Decision:** vector_128 IVF-PQ ANN for Stage 1 (candidates), vector_768 flat numpy cosine for Stage 2 (re-score), NO IVF-PQ on vector_768.  
**Why:** Stage 2 searches only the ~1000 Stage 1 candidates. Flat cosine on 1000 × 768 vectors is microseconds. IVF-PQ on 1000 vectors wastes disk space and hurts accuracy (quantization error on small candidate set is not worth it). Both vectors stored per chunk (768d + 128d = ~3.5KB/chunk, ~85MB for 25k chunks).  
**PQ sub-vectors:** `num_sub_vectors=32` for 128d (128/32 = 4d per sub-vector). More sub-vectors = better quantization quality than 16.

### Dynamic IVF-PQ Threshold
**Decision:** Skip IVF-PQ index creation if chunk count < 256. Use flat search instead.  
**Why:** IVF-PQ requires at least as many vectors as partitions to train KMeans. With `num_partitions=256`, repos with < 256 chunks crash on `create_index()`. Flat search on small repos is also mathematically 100% accurate (zero quantization loss) and faster than traversing an empty partition tree.  
**Threshold check:** `if self._table.count_rows() < 256: return` before calling `create_index()`.

### Single-Hop Graph Expansion
**Decision:** Expand only one hop from seed chunk_ids. Hard limit: 20 neighbors max.  
**Why:** Multi-hop = O(b^d) fan-out where b=branching factor, d=depth. At b=3, d=2: 9 neighbors per seed, 45 for k=5 seeds. "Lost in the middle" (Liu et al. 2023): LLMs attend strongly to start and end of context, weakly to middle. Stuffing 45 extra chunks means middle chunks are invisible.  
**Trade-off:** Some queries require two-hop reasoning (`login → fetch_user → execute_query`). User asks follow-up questions for deeper traversal. Speed and context size win for v1.

### Explicit Metadata Filters (No Auto-Detection)
**Decision:** Symbol, filename, language filters must be provided explicitly by the user (`--symbol`, `--file`, `--lang`).  
**Why:** Auto-detecting symbols from natural language adds latency and uncertainty. Developer queries are explicit — if they know the function name, they'll say it.  
**Symbol lookup threshold:** If exact symbol match returns > 10 results, fall back to hybrid search. Common names (`__init__`, `get`, `run`) return too many matches to be useful without semantic ranking.

### BoW Query Router
**Decision:** Keyword pattern matching (Bag of Words) to classify queries into code_search / conceptual / debugging / setup.  
**Why:** Developer queries are keyword-dense, not prose. BoW accuracy ~90% on this distribution. LLM-based routing would add +500ms latency + token cost per query.  
**Known failure:** Ambiguous queries ("why is this slow?" = debugging or conceptual), negation ("not the auth function"), cross-type queries. Acceptable for v1. See FAILURES.md.  
**Tie-breaking:** DEBUGGING wins on tie (misrouting a stack trace to conceptual = useless answer).

### Cosine Validator + Sampled LLM Judge
**Decision:** Run cosine similarity on every query (fast, ~5ms). Run LLM faithfulness judge on 10% sample only (expensive, ~500ms + tokens).  
**Why:** They measure different failure modes.  
- Low cosine → retrieval problem (wrong chunks retrieved)  
- High cosine + low LLM judge → generation problem (right chunks retrieved, LLM hallucinated)  

Sampling at 10% in production gives signal without doubling token usage. Set to 100% for offline eval runs.

### search_content Composite Column
**Decision:** Bake a composite FTS column `search_content = "Symbol: {name}\nDocs: {doc}\n{code}"` at ingestion time rather than multi-column FTS.  
**Why:** LanceDB native FTS only supports single-column string indexing without Rust/tantivy dependencies. Multi-column FTS throws `ValueError: field_names must be a string`.  
**Trade-off:** ~20% disk footprint increase (redundant storage). In exchange: extreme native retrieval speed, no heavy dependencies, domain model stays clean (CodeChunk.text untouched).

### Multi-Tenant Isolation (Four-Pillar Approach)
**Decision:** Namespace every data structure by `repo_name`.  
**Why:** System was originally built on single-tenant MVP assumptions. Moving to multi-repo indexing revealed three gaps: (1) WAL keys were global, (2) name_map had no repo prefix, (3) vector search had no repo filter.  
**Four pillars:**
1. WAL: keys = `op::repo::path=value`
2. name_map: keys = plain `symbol_name` (repo isolation via chunk_id prefix which is globally unique: `repo::path::symbol::hash`)
3. LanceDB: `repo_name` column + `WHERE repo_name = $repo` pre-filter on all searches
4. KuzuDB: `repo_name` property on every Chunk node + `MATCH (c:Chunk {repo_name: $repo})` on both sides of every edge query

### MODULE Chunk Type (Deferred — V2)
**Decision:** Do not implement file-level MODULE nodes in v1. Document as V2 requirement.  
**Why:** IMPORTS edges require both endpoints to be real nodes. `CodeChunk` models code blocks (functions/classes), not files. Synthetic file-level IDs were generated but had no matching nodes — all IMPORTS edges silently dropped.  
**V2 fix:** Add `ChunkType.MODULE` to represent entire files as nodes. Functions roll up to their parent MODULE. Files IMPORT other files via MODULE → MODULE edges. Not building now — requires schema migration and parser changes. See FAILURES.md Case Study 2.

### No LangChain / LlamaIndex
**Decision:** Build all components directly.  
**Why:** Full observability. Every step is measurable (latency, token count, vector scores, RRF scores). Frameworks abstract this away. "I can't see what's happening" is fatal for a system whose value proposition is transparency.

---

## Eval Strategy

### Golden Dataset: Symbol + Graph Driven Generation
**Decision:** Sample high-degree nodes from KuzuDB call graph, fetch 1-hop context, send to strong LLM for QA generation.  
**Why:** Manual golden dataset creation takes 2-3 hours for 30 pairs and is biased toward code the human already knows. High out-degree nodes are architecturally central — questions about them have richer `expected_chunk_ids`.  
**Known bias:** Only covers orchestrator functions, not leaf functions or error handling paths. Generated questions may be easier than real onboarding questions (LLM writes questions answerable from the same context window it sees). See FAILURES.md.

### Four Metrics
- **Precision@5:** `|retrieved[:5] ∩ expected| / 5` — retrieval signal-to-noise
- **Context Recall:** `|retrieved ∩ expected| / |expected|` — retrieval coverage
- **Answer Relevance:** cosine(embed(answer), embed(ground_truth)) — generation quality
- **Faithfulness:** LLM judge(question, answer, context) — hallucination detection

Together they localize failures: low precision/recall = retrieval issue; high both + low faithfulness = generation/prompt issue.