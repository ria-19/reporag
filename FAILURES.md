# FAILURES.md


## Case Studies from Live System Testing

### Case Study 1: The Multi-Tenant Catastrophe (Global State Collapse)

**Symptom:** After indexing `micrograd` then `fastapi`, micrograd successfully parsed but silently skipped graph building. After a simulated crash and restart, zero edges were written for the resumed job. A query for one codebase returned vector matches from another.

**Root Cause:** System was built on single-tenant MVP assumptions. Three structural gaps:
1. WAL keys were global (`pass2_completed`, `login_function`) — not tied to any repo
2. In-memory `name_map` died on crash with no hydration path; keys had no repo prefix
3. Vector search had no `repo_name` filter — searched across all indexed repos

**Fix:** Complete multi-tenant refactor across four pillars:
1. **WAL:** Keys = `op::repo::path=value`. Added `threading.Lock` to prevent concurrent write corruption
2. **name_map:** Standardized key format; added `fetch_name_map_for_repo()` to rebuild from DB on restart (hydration)
3. **LanceDB:** Added `repo_name` column to schema; forced `WHERE repo_name = $repo` pre-filter on all searches
4. **KuzuDB:** Stamped every Chunk node with `repo_name`; all Cypher queries enforce `{repo_name: $repo}` on both endpoints of every edge

**Lesson:** Multi-tenancy cannot be retrofitted cleanly. Design for it from the first data model.

---

### Case Study 2: The "Phantom Node" & Graph Edge Collapse

**Symptom:** `/health` returned `edges_imports: 0`. CALLS edges were drawn correctly. IMPORTS edges silently dropped to zero.

**Root Cause:** "Synthetic ID Trap." To represent IMPORTS edges, code generated a synthetic file-level ID: `f"{repo_name}::{raw_file.path}::imports"`. But no node with that ID existed in KuzuDB. The Cypher `MATCH` statement silently returned no rows — edge silently dropped.

**Fix (V1):** Documented. IMPORTS edges not written in v1. CALLS graph is complete.

**V2 Required Fix:** Implement `ChunkType.MODULE` — a node type representing an entire file. Functions roll up to their parent MODULE. Files IMPORT other files via MODULE → MODULE IMPORTS edges. Requires: new parser pass to emit MODULE chunks, schema migration, updated DDL, updated `resolve_edges()`.

**Status:** Tracked as V2 requirement. Not a regression — IMPORTS never worked.

---

### Case Study 3: The "Schema Wall" — Composite FTS Search

**Symptom:** `ValueError: field_names must be a string` when calling `create_fts_index(["text", "symbol_name", "docstring"])`.

**Root Cause:** LanceDB native FTS engine supports only single-column string indexing. Multi-column indexing requires heavy Rust/tantivy dependencies not in the default install.

**Fix:** `search_content` composite column baked at ingestion time:
```python
search_content = f"Symbol: {symbol_name}\nDocs: {docstring}\n{text}"
```
FTS index built on `search_content` only. Domain model (`CodeChunk.text`) untouched.

**Trade-off:** ~20% disk footprint increase (data redundancy). In exchange: zero new dependencies, native retrieval speed, clean domain model.

---

### Case Study 4: The "Small Data" KMeans Math Crash

**Symptom:** Indexing `micrograd` (35 functions) instantly crashed: `Lance error: Unprocessable: KMeans cannot train 256 centroids with 35 vectors`.

**Root Cause:** IVF-PQ index hardcoded `num_partitions=256`. KMeans cannot create more clusters than training vectors. Small repos fail immediately.

**Fix:** Dynamic threshold check in `LanceStore.create_indexes()`:
```python
if self._table.count_rows() < 256:
    logger.info("Skipping IVF-PQ (< 256 chunks) — using flat search")
    return
```
Flat search on small repos is also more accurate (zero quantization loss) and faster (no partition tree to traverse).

---

### Case Study 5: The "Silent 404" & Self-Healing Boot Sequence

**Symptom:** API booted cleanly, logged `OllamaLLM ready: llama3.2`, passed health checks. First RAG query returned `500: model 'llama3.2' not found`. Second symptom: fixing this triggered `KeyError: 'name'`.

**Root Causes:**
1. FastAPI lifespan checked if the Ollama **server** was running but did not verify model **weights** existed on disk
2. `ollama-python` SDK changed response schema: `m.model` (attribute) not `m['name']` (dict key)

**Fix:** `_ensure_model_ready()` injected into `OllamaLLM.__init__()`:
- Lists models via `ollama.list()`, uses `m.model` not `m['name']`
- If target model missing: calls `ollama.pull(model_name)` and blocks until complete
- API startup is now strictly ordered: model verified → ready to serve

**Lesson:** Health checks must verify **capability**, not just **presence**. "Ollama is running" ≠ "Ollama can run this query."

---

## Known Limitations

### Language Support
- Only Python, JavaScript, TypeScript supported. Fallback strategy (line-based chunking) used for all other languages.
- No semantic understanding of Go, Rust, Java, C++ — these get chunked as plaintext, losing symbol relationships.

### IMPORTS Graph (V2)
- File-level IMPORTS edges not implemented. Only function-level CALLS edges exist.
- Impact: "what does file X depend on?" queries can't traverse the dependency graph.

### Query Router Accuracy
- BoW keyword matching fails on: ambiguous queries ("why is this slow?"), negation ("not the auth function"), cross-type queries ("how do I debug the setup process?").
- Estimated accuracy: ~90% on developer keyword-dense queries.
- Fix cost: fine-tuned classifier or zero-shot LLM routing (+500ms, token cost per query). Add when eval shows routing errors hurting generation.

### Golden Dataset Bias
- QA pairs generated only from high-degree call graph nodes (orchestrator functions).
- Does not cover: leaf functions, error handling, setup/config code, JS-heavy paths.
- Generated questions may be easier than real onboarding questions (LLM writes questions from the same context it sees).
- Mitigation: manually review 5-10 pairs per eval run. Real fix: human-written pairs from actual onboarding sessions.

### WAL Multi-Worker Safety
- Current WAL uses `threading.Lock()` — protects against concurrent threads within one process.
- If `uvicorn --workers N` (multiple processes) is ever used, file-level locking (`filelock` library) is required.
- Running multiple workers with current WAL = potential file corruption on concurrent index jobs.

### Reranker Not Implemented
- Graph-expanded neighbors are appended to context with no quality ranking.
- If neighbor chunks are noisy, they may hurt generation quality.
- Measure faithfulness score first; add `bge-reranker-v2-m3` if score is low.

### LanceDB Incremental Indexing Gap
- New chunks added after initial IVF-PQ index creation are searched via flat scan (slow).
- `optimize()` merges deltas and reindexes, but runs on a background timer (default: every 5 minutes).
- Window between `add()` and next `optimize()`: slightly degraded query speed for new chunks.
- Acceptable for local system where indexing is infrequent. Production fix: dedicated indexing pipeline with index warming.

### Context Assembly — Simple Interleaving
- "Lost in the middle" mitigation uses a simple interleaving strategy: retrieved chunks at position 0 and -1, neighbors in middle.
- Does not account for token budget dynamically (uses character count approximation, 1 token ≈ 4 chars).
- Does not consider chunk diversity (may include redundant chunks from same file).

### No Chunk-Level Resume Within a File
- If a file has 200 functions and the server crashes after flushing sub-batches 1-29 of 30, the entire file is re-indexed on resume.
- Sub-batches 1-29 will be overwritten idempotently (LanceDB merge_insert, KuzuDB MERGE), but wasted computation.
- Fix: add `chunk_offset` field to WAL entry. Not built — YAGNI.