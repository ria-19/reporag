"""
LanceDB adapter — implements VectorStorePort.

Responsibilities:
  - Store CodeChunk records (vector + metadata + text)
  - Hybrid search: vector similarity + BM25 full-text
  - Fetch chunks by chunk_id list (for graph neighbor lookup)
  - Background index optimization

Schema: ChunkRecord (defined below)
  Two vector columns: vector_768 (precise), vector_128 (fast)
  Full-text search index on: text, symbol_name, docstring

WHY LanceDB over FAISS + JSON Lines (what we had before)?
  FAISS: vectors only, no metadata, no BM25, no persistence API.
  LanceDB: vectors + BM25 + metadata filtering + persistence,
  written in Rust, embedded (no server), Apache Arrow columnar.
  One dependency replaces three systems.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector

from src.core.models import CodeChunk, RetrievedChunk, ChunkType, Language
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# ════════════════════════════════════════════════════════
# SCHEMA
# Defined once. Changing requires re-indexing.
# ════════════════════════════════════════════════════════

class ChunkRecord(LanceModel):
    """
    LanceDB row. One per CodeChunk.

    WHY LanceModel over raw PyArrow schema?
    LanceModel auto-generates the Arrow schema from type hints.
    Pydantic validation on insert catches bad data early.
    Vector(N) is a LanceDB type that maps to fixed-size float32 array.
    """
    # Identity
    chunk_id:    str
    repo_name:   str
    symbol_name: str
    chunk_type:  str        # ChunkType.value — stored as string
    language:    str        # Language.value
    file_path:   str
    start_line:  int
    end_line:    int

    # Content — both indexed for BM25 full-text search
    text:      str
    docstring: str = ""

    # MRL vectors
    vector_768: Vector(768)   # precise re-scoring
    vector_128: Vector(128)   # fast candidate search

    # For name_map hydration
    calls_out_raw: str = "[]"   
    imports_raw:   str = "[]" 

    # ── COMPOSITE SEARCH FIELD For FTS ───────────────────
    search_content: str


# ════════════════════════════════════════════════════════
# STORE
# ════════════════════════════════════════════════════════

class LanceStore:
    """
    Implements VectorStorePort.
    All LanceDB interaction lives here.
    indexer.py never imports lancedb directly.
    """

    TABLE_NAME = "chunks"

    def __init__(
        self,
        db_path: str = None,
        optimize_interval_s: int = 300,
    ):
        db_path = db_path or settings.lance_db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(db_path)
        self._table = self._get_or_create_table()
        self._optimize_interval = optimize_interval_s

        # Background optimizer — merges deltas, reindexes new data
        self._start_optimizer()

        logger.info("LanceStore ready at %s", db_path)

    def _get_or_create_table(self):
        """
        Get existing table or create with schema.
        WHY check existence? Resume support — don't wipe on restart.
        """
        if self.TABLE_NAME in self._db.table_names():
            logger.info("LanceStore: opening existing table")
            return self._db.open_table(self.TABLE_NAME)

        logger.info("LanceStore: creating new table")
        table = self._db.create_table(
            self.TABLE_NAME,
            schema=ChunkRecord,   # LanceModel → Arrow schema
        )
        return table

    # ── Write ──────────────────────────────────────────

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """
        Write a batch of embedded chunks to LanceDB.
        Uses MERGE semantics on chunk_id — idempotent.
        WHY idempotent? WAL resume may replay a batch.
        Duplicate chunk_id = overwrite, not duplicate row.

        
        2. table.merge_insert("chunk_id").when_matched_update_all()
           .when_not_matched_insert_all().execute(records)
        WHY merge_insert over add()?
        add() creates duplicate rows on resume.
        merge_insert() upserts — safe to call multiple times.
        """
        if not chunks:
            return

        records = []
        for chunk in chunks:
            try:
                records.append(chunk.to_lance_record())
            except RuntimeError as e:
                # Chunk has no vector — skip and log
                # Should not happen if indexer calls embedder first
                logger.error("Skipping chunk without vector: %s", e)
                continue

        if not records:
            return

        try:
            (
                self._table.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )
            logger.debug("Successfully upserted %d chunks to LanceDB.", len(records))
        except Exception as e:
            logger.error("Failed to merge_insert chunks into LanceDB: %s", e)
            raise

    def is_indexed(self, chunk_id: str) -> bool:
        """
        Check if a chunk exists in the store.
        Used by WAL resume logic.
        """
        results = (
            self._table.search()
            .where(f"chunk_id = '{chunk_id}'")
            .limit(1)
            .to_list()
        )
        return len(results) > 0
         

    # ── Read ───────────────────────────────────────────
    
    def fetch_by_symbol(
        self,
        symbol_name: str,
        repo_name: str,
        filename: str | None = None,
        language: str | None = None
    ) -> list[CodeChunk]:
        """
        Exact symbol lookup by name.
        Used when query contains a specific function/class name.
        Faster and more precise than vector search for exact matches.

        WHY separate from hybrid_search?
        Vector search finds semantically similar chunks.
        This finds the exact symbol — different operation, different use case.
        pipeline.py checks for exact symbol match first,
        falls back to hybrid_search if not found.
        """
        conditions = [f"repo_name = '{repo_name}'", f"symbol_name = '{symbol_name}'"] 
        if language:
            conditions.append(f"language = '{language}'")
        if file_path_contains:
            conditions.append(f"file_path LIKE '%{file_path_contains}%'")

        where_clause = " AND ".join(conditions)
        results = (
            self._table
            .search()
            .where(where_clause)
            .to_pydantic(ChunkRecord)
        )
        return [self._record_to_chunk(r) for r in results]

    def fetch_name_map_for_repo(
        self,
        repo_name: str,
    ) -> dict[str, list[str]]:
        """
        Bulk fetch all chunks for a repo → rebuild name_map.
        Called ONCE at start of index_repo() before Pass 1 loop.

        WHY before Pass 1?
        Pass 1 extends name_map with new files.
        We need existing chunks in name_map so Pass 2 can
        resolve edges that cross old-file → new-file boundaries.

        WHY one query not N queries?
        N queries = one per skipped file = O(files) roundtrips.
        One query = full table scan filtered by repo_name = O(1) roundtrip.
        For 1000 files: 999 queries saved.

        Returns:
            {symbol_name: [chunk_id, ...]}
            Same structure as indexer's in-memory name_map.
        """
        import json

        results = (
            self._table
            .search()
            .where(f"repo_name = '{repo_name}'")
            .select(["symbol_name", "chunk_id"])
            .to_list()   # returns list of dicts
        )

        name_map: dict[str, list[str]] = {}
        for record in results:
            name_map.setdefault(
                f"{record.repo_name}::{record.symbol_name}", []
            ).append(record.chunk_id)
        
        logger.info(
            "Rebuilt name_map for repo '%s': %d symbols from %d chunks",
            repo_name, len(name_map), len(results),
        )
        return name_map

    def fetch_all_repo_metadata(self, repo_name: str) -> list[dict]:
        """
        Bulk fetch all metadata columns for a specific tenant.
        Returns raw dictionaries directly from LanceDB.
        """
        return (
            self._table.search()
            .where(f"repo_name = '{repo_name}'")
            .select(["chunk_id", "symbol_name", "calls_out_raw", "imports_raw"])
            .to_list()
    )
    
    def hybrid_search_filtered(
        self,
        vector_128:  list[float],
        vector_768:  list[float],
        query_text:  str,
        repo_name:   str,
        k:           int = None,
        candidate_k: int = None,
        symbol:      str | None = None,   
        filename:    str | None = None,  
        language: str | None = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """
        Two-stage MRL cascade + BM25 hybrid search with optional metadata pre-filter.

        Stage 1: vector_128 IVF_PQ ANN → candidate_k results  (fast)
        Stage 2: vector_768 FLAT cosine on candidates          (precise)
        BM25:    FTS search → candidate_k results
        Merge:   RRF on (Stage 2 ranks + BM25 ranks) → top k

        Stage 2 is flat numpy cosine — NOT a LanceDB index search.
        We fetch the stored vector_768 for each candidate and
        compute cosine similarity directly in numpy.
        WHY numpy and not LanceDB search?
        LanceDB's .search() on vector_768 would use its index (if it had one)
        or do a full table scan. We want to search ONLY the candidates.
        Numpy on 1000×768 is microseconds and exact.

        Pre-filter narrows the search space before ANN.
        WHY pre-filter not post-filter?
        Post-filter: search all N chunks, then discard non-matching.
        Wastes ANN computation on chunks we'll throw away.
        Pre-filter: restrict index scan to matching rows first.
        LanceDB supports this natively via .where() before .search().

        Args:
            vector_128:  fast search vector (128d, query-time)
            vector_768:  precise vector (768d, query-time)
            query_text:  raw query string for BM25
            k:           final results to return
            candidate_k: candidates from Stage 1 (default k*10)

        Returns:
            list[RetrievedChunk] ordered by final RRF score
        """
        
        k = k or settings.retrieval_k
        candidate_k = candidate_k or k * settings.retrieval_k_fetch_multiplier

        if not repo_name:
            raise ValueError("Security Violation: repo_name is required for all searches.")

        def clean(val: str) -> str:
            return val.replace("'", "''")

        conditions = [f"repo_name = '{clean(repo_name)}'"]
        
        if symbol:
            conditions.append(f"symbol_name = '{clean(symbol)}'")
            
        if filename:
            conditions.append(f"file_path LIKE '%{filename}'")
            
        if language:
            conditions.append(f"language = '{language}'")
            
        if "chunk_type" in kwargs and kwargs["chunk_type"]:
            conditions.append(f"chunk_type = '{kwargs['chunk_type']}'")

        where_clause = " AND ".join(conditions)

        search = (
            self._table
            .search(vector_128, vector_column_name="vector_128")
            .metric("cosine")
            .limit(candidate_k or settings.retrieval_k * 10)
        )
        if where_clause:
            search = search.where(where_clause)

        stage1_results = search.to_pydantic(ChunkRecord) # easy attribute access + validation
        # in RAM + already contains vector_768 inside each object (ChunkRecord)


        # BM25: full-text search with query_text
        bm25_results = (
            self._table
            .search(query_text, query_type="fts")
            .limit(candidate_k).to_pydantic(ChunkRecord)
        )


        # Stage 2: precise 768d re-score on Stage 1 candidates only
        # Fetch stored vector_768 for each candidate
        # Compute cosine similarity in numpy — flat, exact, fast on 1000 vecs        
        query_vec = np.array(vector_768, dtype=np.float32)
        stage2_scores: dict[str, float] = {}

        for record in stage1_results:
            stored_vec = np.array(record.vector_768, dtype=np.float32)
            score = float(np.dot(query_vec, stored_vec)) 
            stage2_scores[record.chunk_id] = score

        # Build ranked lists for RRF
        # Stage 2 ranking: sort by 768d cosine score
        stage2_ranked = sorted(stage2_scores.items(), key=lambda x : x[1], reverse=True)    

        # BM25 ranking: order from FTS results (already ranked by LanceDB)
        bm25_ranked = [(r.chunk_id, idx) for idx, r in enumerate(bm25_results)]

        # RRF merge — implemented in retrieval/hybrid.py
        # We call it here, passing both ranked lists
        # TODO: optimze using matrix, reducent for-loop removal + score population not done ye

        # For now, stub:
        final_ids = self._rrf_merge(stage2_ranked, bm25_ranked, k=k)

        # Build RetrievedChunk results
        record_map = {r.chunk_id: r for r in stage1_results}
        record_map.update({r.chunk_id: r for r in bm25_results})

        results = []
        for chunk_id, rrf_score in final_ids:
            record = record_map.get(chunk_id)
            if not record:
                continue
            chunk = self._record_to_chunk(record)
            results.append(RetrievedChunk(
                chunk=chunk,
                vector_score=stage2_scores.get(chunk_id),
                rrf_score=rrf_score,
                source="hybrid",
            ))

        return results

    def fetch_by_ids(self, chunk_ids: list[str]) -> list[CodeChunk]:
        """
        Fetch full chunk records by chunk_id list.
        Used after graph expansion — we have neighbor IDs,
        need their full text for reranking.
        """
        if not chunk_ids:
            return []
        ids_str = ", ".join(f"'{cid}'" for cid in chunk_ids)
        results = (
            self._table.search()
            .where(f"chunk_id IN ({ids_str})")
            .to_pydantic(ChunkRecord)
        )
        return [self._record_to_chunk(r) for r in results]

    def _rrf_merge(
        self,
        ranked_a: list[tuple[str, float]],
        ranked_b: list[tuple[str, int]],
        k: int,
        rrf_k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion.

        score(d) = Σ 1 / (rrf_k + rank(d))
        rrf_k=60: standard constant from original 2009 paper.
        WHY 60? Empirically found to be robust across datasets.
        Dampens the impact of very high ranks without ignoring them.

        ranked_a: (chunk_id, score) — we use position as rank
        ranked_b: (chunk_id, rank_index) — already positional
        """
        scores: dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(ranked_a):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rrf_k + rank)

        for rank, (chunk_id, _) in enumerate(ranked_b):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rrf_k + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def _record_to_chunk(self, record: ChunkRecord) -> CodeChunk:
        """Convert LanceDB record back to CodeChunk model."""
        return CodeChunk(
            chunk_id=record.chunk_id,
            repo_name=record.repo_name,
            symbol_name=record.symbol_name,
            chunk_type=ChunkType(record.chunk_type),
            language=Language(record.language),
            file_path=record.file_path,
            start_line=record.start_line,
            end_line=record.end_line,
            text=record.text,
            docstring=record.docstring or None,
        )
    
    def create_indexes(self) -> None:
        """
        Index strategy:
        vector_128: IVF_PQ — fast ANN over full corpus (Stage 1)
        vector_768: NO INDEX — flat search on Stage 1 candidates only
        text/symbol_name/docstring: FTS index for BM25
        
        Stage 2 re-scores only the ~1000 candidates from Stage 1.
        Flat cosine similarity on 1000 vectors is microseconds.
        IVF_PQ on 1000 vectors: wastes disk space + hurts accuracy
        (quantization error on small candidate set is not worth it).

        WHY IVF_PQ?
        IVF (Inverted File): partitions space into clusters,
        searches only nearby clusters → sublinear query time.
        PQ (Product Quantization): compresses vectors in memory,
        reduces RAM 4-8x with small accuracy loss.
        Together: fast and memory-efficient approximate search.

        num_partitions: number of IVF clusters
        Rule of thumb: sqrt(n_vectors)
        For 25k chunks: sqrt(25000) ≈ 158, round to 256

        num_sub_vectors: PQ compression
        Must divide embedding dimension evenly.
        128 / 32 = 4d per sub-vector for vector_128 
        768 / 8 = 96 sub-vectors for vector_768
        """
        num_rows = len(self._table)
        logger.info(f"Creating ANN indexes for {num_rows} vectors...")

        # 2. Logic: Only create IVF_PQ if we have enough data
        # Rule of thumb: You need at least ~10-20x more rows than partitions
        if num_rows > 256:
            num_partitions = 256
            logger.info(f"Building IVF_PQ index with {num_partitions} partitions")

            self._table.create_index(
                metric="cosine",
                vector_column_name="vector_128",
                index_type="IVF_PQ",
                num_partitions=num_partitions,
                num_sub_vectors=32,    # 128 / 32
                replace=True,          # rebuild if exists
            )
        else:
            # For small repos (like micrograd), a flat search is FASTER and 
            # more accurate than a partitioned search anyway.
            logger.info("Skipping IVF_PQ index (too few vectors). Using flat search.")


        # ALWAYS create the FTS index for BM25
        # Covers text, symbol_name, docstring
        logger.info("LanceDB: Creating FTS index on 'search_content'...")
        self._table.create_fts_index("search_content", replace=True)
        
        logger.info("Indexes created (vector_128 IVF_PQ + FTS)")


    def delete_repo_records(self, repo_name: str):
        """Cleanup for force re-indexing."""
        self._table.delete(f"repo_name = '{repo_name}'")
        logger.info(f"LanceDB: Records for {repo_name} purged.")

    # ── Background optimizer ───────────────────────────

    def _start_optimizer(self) -> None:
        """
        Background thread: periodically calls optimize().

        LanceDB incremental behavior:
          New data after index creation → flat scan (slow).
          optimize() → merges deltas, reindexes new data.

        WHY daemon thread?
        Dies automatically when main process exits.
        We don't want optimize() to block shutdown.
        """
        def _run():
            while True:
                time.sleep(self._optimize_interval)
                try:
                    self._table.optimize()
                    logger.debug("LanceDB optimize() completed")
                except Exception as e:
                    logger.warning("LanceDB optimize() failed: %s", e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.debug(
            "LanceDB background optimizer started (interval=%ds)",
            self._optimize_interval,
        )

    # ── Stats ──────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return basic stats for /health endpoint.
        """
        count = self._table.count_rows()
        return {
            "chunks_indexed": count,
            "table_name":     self.TABLE_NAME,
        }