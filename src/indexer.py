"""
Write path orchestrator.

Ties together: ingestion → chunking → embedding → storage.

Two passes:
  Pass 1 (streaming): parse files → embed in batches → write to Lance + Kuzu
  Pass 2 (resolution): resolve raw edges → write graph edges to Kuzu

Design decisions:
  - All dependencies injected via constructor (Dependency Injection)
    WHY: swap any component without touching this file
  - Streaming parse: constant memory per file
    WHY: RAM stays flat regardless of repo size
  - Batch embedding: accumulate BATCH_SIZE chunks, embed together
    WHY: SIMD optimization in sentence-transformers, higher throughput
  - WAL (Write-Ahead Log): record progress before each operation
    WHY: resume on failure without redundant computation

⚠️  MEMORY NOTE:
  ParsedRepo is NOT held in memory between passes.
  Only name_map (dict[str, list[str]]) and raw_edges (list[RawEdge])
  are kept between Pass 1 and Pass 2.
  Estimate: ~3MB for a 10k-function repo. Fine for local.
  Vectors: NOT held in memory — embedded per batch, written immediately.


⚠️ WAL GRANULARITY NOTE:
    We record completion at file level, not batch level.

    Small files (< batch_size chunks): grouped with other files,
    flushed together, WAL recorded once per file. Efficient.

    Large files (> batch_size chunks): flushed in sub-batches,
    WAL recorded only after ALL sub-batches succeed.
    If crash at sub-batch 29 of 30: entire file re-indexed on resume.
    Tradeoff accepted because:
    1. Files this large are rare (our loader already caps at 1MB)
    2. Sub-batches are idempotent (Lance + Kuzu use MERGE)
    3. Chunk-level WAL would add thousands of entries per file
        with more I/O overhead than the retry cost

    If this becomes a real problem: add chunk_offset to WAL entry
    and resume from last successful sub-batch within a file.
    Not building it now — YAGNI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import threading
from typing import Protocol

from src.core.models import (
    CodeChunk, RawEdge, GraphEdge,
    ParsedRepo,
)
from src.chunking.parser import CodeParser
from src.chunking.graph import build_name_map, resolve_edges
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════
# PORTS — what indexer.py needs, not what implements it
# These are the interfaces. Concrete classes in storage/
# implement them. This file never imports from storage/.
#
# WHY Protocol over ABC?
# Protocol = structural typing. Any class with the right
# methods satisfies the interface — no explicit inheritance.
# Makes mocking in tests trivial: just make a class with
# the right methods. No need to import the Protocol.
# ════════════════════════════════════════════════════════

class VectorStoreWritePort(Protocol):
    """Interface for vector + metadata storage (LanceDB)."""

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Write chunks with vectors to vector store."""
        ...

    def is_indexed(self, chunk_id: str) -> bool:
        """Check if chunk already exists — used for resume."""
        ...


class GraphStoreWritePort(Protocol):
    """Interface for graph storage (KuzuDB)."""

    def add_nodes(self, chunks: list[CodeChunk]) -> None:
        """Write chunk nodes to graph."""
        ...

    def add_edges(self, edges: list[GraphEdge]) -> None:
        """Write resolved edges to graph."""
        ...

    def node_exists(self, chunk_id: str) -> bool:
        """Check if node already exists — used for resume."""
        ...


class DocumentEmbedderPort(Protocol):
    """Interface for text → vector embedding."""

    def embed_batch(self, texts: list[str], is_query: bool = False) -> tuple[list[list[float]], list[list[float]]]:
        """
        Embed a batch of texts.
        Returns list of vectors, same order as input.
        """
        ...

    # def embed_query(
    #     self,
    #     text: str,
    # ) -> tuple[list[float], list[float]]:
    #     """Returns (vector_768, vector_128) for a single query."""
        ...

class LoaderPort(Protocol):
    """Interface for file ingestion."""

    def stream_files(self, repo_path: str, repo_name: str):
        """
        Yield RawFile objects one at a time.
        WHY generator: constant memory regardless of repo size.
        """
        ...


# ════════════════════════════════════════════════════════
# WAL — Write-Ahead Log
# Records progress so indexing can resume after failure.
#
# Format: one JSON line per operation
# {"op": "file_parsed", "path": "src/auth.py", "ts": 1234567890}
# {"op": "file_indexed", "path": "src/auth.py", "ts": 1234567891}
# {"op": "edges_written", "ts": 1234567892}
#
# WHY JSON Lines?
# Appendable without rewriting the file.
# Human readable — you can inspect it.
# Same reason vector_db.py chose JSONL for metadata.
# ════════════════════════════════════════════════════════

class WAL:
    """
    Write-Ahead Log for resumable indexing.
    """

    def __init__(self, wal_path: str):
        self.path = Path(wal_path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load existing WAL on startup — rebuilds done set."""
        if not self.path.exists():
            return
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = self._make_key(
                        entry.get("repo", ""), 
                        entry.get("op", ""), 
                        **entry.get("kwargs", {})
                    )
                    self._done.add(key)
                except json.JSONDecodeError:
                    continue   # corrupted line — skip, don't crash
        logger.info("WAL loaded: %d completed operations", len(self._done))

    def _make_key(self, repo: str, op: str, **kwargs) -> str:
        """
        Unique key per operation.
        file ops: repo_name + op + path (relative)
        edge ops: op only (one global operation)
        """
        parts = [repo, op] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        return "::".join(parts)

    def record(self, repo: str, op: str, **kwargs) -> None:
        """
        Append one completed operation to the WAL.
        Called AFTER the operation succeeds.

        WHY after and not before?
        WAL records completion, not intent.
        If we crash during the operation, the WAL won't have
        this entry → we retry on resume. Safe.
        """
        key = self._make_key(repo, op, **kwargs)

        entry = {
            "repo": repo,
            "op": op,
            "kwargs": kwargs,
            "ts": time.time()
        }
        
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._done.add(key)   # keep in-memory set in sync

    def is_done(self, repo: str, op: str, **kwargs) -> bool:
        """Check if an operation was already completed."""
        key = self._make_key(repo, op, **kwargs)
        print(f"DEBUG: Generated Key: '{key}' | Known Keys: {list(self._done)[:2]}")
        return key in self._done

    def clear(self, repo: str = "") -> None:
        """Delete/Clear WAL for one repo only — start fresh. Call before a full re-index.""" 
        with self.lock:
            if not repo:
                if self.path.exists():
                    self.path.unlink()
                self._done.clear()
                return
                
            if not self.path.exists():
                return

            remaining_entries = []
            with open(self.path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("repo") != repo:
                        remaining_entries.append(line)

            with open(self.path, "w") as f:
                f.writelines(remaining_entries)
            
            # Rebuild the in-memory set from the new file state
            self._done.clear()
            self._load()

        logger.info(f"WAL cleared for {repo}")


# ════════════════════════════════════════════════════════
# INDEXER
# ════════════════════════════════════════════════════════

class Indexer:
    """
    Orchestrates the full write path.

    All dependencies injected — this class never instantiates
    LanceDB, KuzuDB, or the embedder directly.

    WHY: Open/Closed Principle.
    Swap any component by passing a different implementation.
    This file never changes when storage backends change.
    """

    def __init__(
        self,
        loader:       LoaderPort,
        parser:       CodeParser,
        embedder:     DocumentEmbedderPort,
        vector_store: VectorStoreWritePort,
        graph_store:  GraphStoreWritePort,
        wal_path:     str = ".rag_wal/index.jsonl",
        batch_size:   int = None,
    ):
        self.loader       = loader
        self.parser       = parser
        self.embedder     = embedder
        self.vector_store = vector_store
        self.graph_store  = graph_store
        self.wal          = WAL(wal_path)
        self.batch_size   = batch_size or settings.embedding_batch_size

    def index_repo(
        self,
        repo_path: str,
        repo_name: str,
        force: bool = False,
    ) -> dict:
        """
        Full write path: ingest → chunk → embed → store.

        Args:
            repo_path: local path to the repo
            repo_name: identifier used in chunk_ids
            force:     if True, clear WAL and re-index everything

        Returns:
            summary dict with counts and timing
        """
        if force:
            logger.info(f"FORCE: Resetting state for {repo_name}...")
            self.wal.clear(repo=repo_name)
            self.vector_store.delete_repo_records(repo_name)
            self.graph_store.delete_repo_nodes(repo_name)

        start_time = time.time()

        # Accumulators for Pass 2
        # WHY only these two and not full chunks?
        # Chunks are written to storage immediately after embedding.
        # We only need what Pass 2 requires: name resolution.

        name_map: dict[str, list[str]] = {}
        all_raw_edges: list[RawEdge]    = []

        # HYDRATION: Rebuild the 'brain' if we are resuming/updating
        if not force:
            # ── Rebuild name_map from existing indexed chunks ──
            # WHY before Pass 1?
            # Skipped files (WAL) won't enter the Pass 1 loop.
            # Their symbols must still be in name_map for Pass 2.
            # One bulk query replaces N per-file queries.
            
            logger.info(f"Hydrating memory from storage for {repo_name}...")
            self._hydrate_indexer_state(repo_name, name_map, all_raw_edges)

        # Batch accumulator
        # Chunks accumulate here until batch_size is reached,
        # then embedded + written + discarded.
        cross_file_batch: list[CodeChunk] = []
        cross_file_batch_files: set[str] = set() # Tracks current batch

        # Counters for summary
        files_processed = 0
        chunks_total    = 0
        files_skipped   = 0

        # ── PASS 1: Stream, parse, embed in batches ──────────
        logger.info("Pass 1: streaming %s", repo_path)

        for raw_file in self.loader.stream_files(repo_path, repo_name):
            logger.info(f"Processing: {raw_file.path} ({raw_file.size_bytes / 1024:.1f} KB)")

            # WAL check — skip if already indexed. - [Read File -> Parse AST -> Extract Chunks] as a single, atomic, in-memory operation 
            # because re-doing it is practically instant + lives in RAM so either we write every function every class which would create I/O WAL file(disk) >> actual process

            print(f"DEBUG: Checking WAL for repo={repo_name}, path={raw_file.path}")
            if self.wal.is_done(repo_name, "file_indexed", path=raw_file.path):
                logger.debug("Skip (WAL): %s", raw_file.path)
                files_skipped += 1
                continue

            # Parse one file
            logger.debug(f"  -> Parsing {raw_file.path}...")
            chunks, raw_edges = self.parser.parse(raw_file, repo_name)
            logger.debug(f"  -> Parsed into {len(chunks)} chunks.")

            if not chunks:
                # Parser returned nothing — unsupported language
                # fallback returned empty (shouldn't happen) or file was
                # truly empty after stripping. Skip.
                continue

 
            # Update name_map with this file's chunks
            # WHY here and not after embedding?
            # name_map only needs chunk_id and symbol_name.
            # Both are set during parsing, before embedding.
            for chunk in chunks:
                key = f"{repo_name}::{chunk.symbol_name}"
                name_map.setdefault(key, []).append(chunk.chunk_id)

            # Accumulate raw edges
            all_raw_edges.extend(raw_edges)
            files_processed += 1
                
            if len(chunks) >= self.batch_size:
                # Large file: flush in sub-batches, WAL after all done
                # WAL NOT recorded per sub-batch — file is atomic unit
                for i in range(0, len(chunks), self.batch_size):
                    sub = chunks[i : i + self.batch_size]
                    self._flush_batch_raw(sub)
                    chunks_total += len(sub)
                
                self.wal.record(repo_name, "file_indexed", path=raw_file.path)
            else:
                # Normal file: accumulate into cross-file batch
                cross_file_batch.extend(chunks)
                cross_file_batch_files.add(raw_file.path)

                if len(cross_file_batch) >= self.batch_size:
                    flushed = self._flush_batch(repo_name, cross_file_batch, cross_file_batch_files)
                    chunks_total += flushed
                    cross_file_batch = []
                    cross_file_batch_files = set()

        # Flush remainder of cross-file batch
        if cross_file_batch:
            flushed = self._flush_batch(repo_name, cross_file_batch, cross_file_batch_files)
            chunks_total += flushed

        logger.info(
            "Pass 1 complete: %d files, %d chunks, %d skipped",
            files_processed, chunks_total, files_skipped,
        )

        # ── PASS 2: Resolve edges ─────────────────────────────
        if not self.wal.is_done(repo_name, "edges_written"):
            logger.info("Pass 2: resolving %d raw edges", len(all_raw_edges))
            
            # Delete partial edges from previous crashed run
            # WHY delete first? Crash mid-Pass-2 leaves partial edges.
            # name_map is now complete — safe to rewrite all edges.
            self.graph_store.delete_edges_for_repo(repo_name)
            
            edges, resolved, unresolved = resolve_edges(repo_name, all_raw_edges, name_map)
            
            self.graph_store.add_edges(edges)
            self.wal.record(repo_name, "edges_written")
            
            logger.info(
                "Pass 2 complete: %d edges written (%d unresolved/external)",
                len(edges), unresolved,
            )
        else:
            logger.info("Pass 2: skipped (WAL: edges already written)")

        # ── Final : Create Index  ─────────────────────────────
        logger.info("Pass 2 complete. Finalizing storage...")
        self.vector_store.create_indexes()

        elapsed = time.time() - start_time

        return {
            "repo":             repo_name,
            "files_processed":  files_processed,
            "files_skipped":    files_skipped,
            "chunks_indexed":   chunks_total,
            "elapsed_seconds":  round(elapsed, 2),
        }

    def _flush_batch_raw(self, batch: list[CodeChunk]) -> int:
        """
        Embed one batch and write to both stores.

        Order matters:
          1. Embed (adds vectors to chunks in-place)
          2. Write to LanceDB (needs vectors)
          3. Write nodes to KuzuDB (doesn't need vectors,
             but we do it here to keep writes together)

        WHY in-place vector assignment?
        Avoids creating new objects. CodeChunk.vector starts
        as None, embedder fills it. to_lance_record() checks
        it's not None before writing.

        Returns count of chunks written.
        """
        if not batch:
            return 0

        # Step 1: embed
        texts   = [chunk.text for chunk in batch]
        logger.debug(f"  -> Embedding {len(batch)} chunks...")
        vectors_768, vectors_128 = self.embedder.embed_batch(texts)
        logger.debug(f"  -> Embedding complete.")

        # Assign vectors in-place
        # WHY zip? Guarantees order alignment between chunks and vectors
        for chunk, v768, v128 in zip(batch, vectors_768, vectors_128):
            chunk.vector_768 = v768
            chunk.vector_128 = v128

        # Step 2: write to LanceDB
        self.vector_store.add_chunks(batch)

        # Step 3: write nodes to KuzuDB
        self.graph_store.add_nodes(batch)

        logger.debug("Flushed batch of %d chunks", len(batch))
        return len(batch)

    def _flush_batch(
        self,
        repo_name: str,
        batch: list[CodeChunk],
        batch_files: set[str],
    ) -> int:
        """
        Flush accumulated cross-file batch + record WAL.
        Used for normal-sized files that share a batch.
        """
        if not batch:
            return 0
        
        self._flush_batch_raw(batch)
        
        # Record file as indexed in WAL
        # WAL — record AFTER both stores confirm write
        # WHY after both? If we crash between Lance write and
        # Kuzu write, WAL won't have this entry → we retry
        # both on resume. Lance uses MERGE (idempotent).
        # Kuzu uses MERGE (idempotent). Safe to retry.
        for file_path in batch_files:
            self.wal.record(repo_name, "file_indexed", path=file_path)
        
        return len(batch)

    def _hydrate_indexer_state(
        self, 
        repo_name: str, 
        name_map: dict[str, list[str]], 
        all_raw_edges: list[RawEdge]
    ) -> None:
        """
        Rebuilds the name_map and all_raw_edges from existing storage.
        Ensures that files skipped by the WAL still participate in the 
        Pass 2 graph resolution.
        """
        import json
        from src.core.models import RawEdge, EdgeType

        logger.info(f"Hydrating memory from LanceDB for repo: {repo_name}")
        
        # 1. Fetch raw rows
        records = self.vector_store.fetch_all_repo_metadata(repo_name)
        
        for r in records:
            chunk_id = r["chunk_id"]
            symbol   = r.get("symbol_name")

            # --- A. HYDRATE TARGETS (name_map) ---
            if symbol:
                # We use the same namespacing logic as Pass 1
                key = f"{repo_name}::{symbol}"
                name_map.setdefault(key, []).append(chunk_id)

            # --- B. HYDRATE SOURCES (all_raw_edges) ---
            # LanceDB might return these as JSON strings or lists depending on the driver
            def _ensure_list(data) -> list[str]:
                if isinstance(data, str):
                    try:
                        return json.loads(data)
                    except:
                        return []
                return data if isinstance(data, list) else []

            calls_raw = _ensure_list(r.get("calls_out_raw", []))
            imports_raw = _ensure_list(r.get("imports_raw", []))

            for target in calls_raw:
                all_raw_edges.append(RawEdge(
                    source_id=chunk_id,
                    target_name=target,
                    edge_type=EdgeType.CALLS
                ))

            for target in imports_raw:
                all_raw_edges.append(RawEdge(
                    source_id=chunk_id,
                    target_name=target,
                    edge_type=EdgeType.IMPORTS
                ))

        logger.info(
            f"Hydration complete: {len(name_map)} symbols and {len(all_raw_edges)} raw edges loaded."
        )