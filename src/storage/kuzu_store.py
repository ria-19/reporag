"""
KuzuDB adapter — implements GraphStorePort.

Responsibilities:
  - Store Chunk nodes (metadata only, no vectors)
  - Store CALLS and IMPORTS edges
  - Graph expansion: given chunk_ids, return neighbor chunk_ids

WHY KuzuDB over Neo4j, NetworkX, SQLite adjacency table?
  Neo4j:     server process, expensive, pointer-chasing on disk
  NetworkX:  in-memory only, no persistence
  SQLite:    no graph query language, adjacency table = manual joins
  KuzuDB:    embedded (no server), columnar (no pointer-chasing),
             SIMD vectorized, Cypher query language, free.

Schema (defined once at DB creation):
  NODE: Chunk  — one per CodeChunk
  EDGE: CALLS  — directed, function A calls function B
  EDGE: IMPORTS — directed, file A imports from file B

Graph expansion query (single-hop, bounded):
  MATCH (a)-[:CALLS|IMPORTS]->(b)
  WHERE a.chunk_id IN $ids
  RETURN DISTINCT b.chunk_id
  LIMIT $limit

WHY single-hop?
  Multi-hop causes exponential fan-out (b^d growth).
  Lost-in-the-middle: LLMs attend weakly to middle context.
  Single-hop gives immediate dependencies — enough for most queries.
  User can ask follow-up questions for deeper traversal.

Edge validation: silent skip on missing endpoints.
Rationale: validating each edge against storage during indexing
couples the chunking layer to the storage layer. Instead, we
accept silent skips and handle cleanup in the repo update job.
Dangling edges are low risk — expand_neighbors() returns chunk_ids
that are then fetched from LanceDB. Missing chunks return empty
from fetch_by_ids(), which the pipeline handles gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import kuzu

from src.core.models import CodeChunk, GraphEdge, EdgeType
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════
# DDL — run once at DB creation
# ════════════════════════════════════════════════════════

_DDL = """
CREATE NODE TABLE IF NOT EXISTS Chunk (
    chunk_id    STRING,
    repo_name  STRING,
    symbol_name STRING,
    chunk_type  STRING,
    language    STRING,
    file_path   STRING,
    start_line  INT64,
    end_line    INT64,
    docstring   STRING,
    PRIMARY KEY (chunk_id)
);

CREATE REL TABLE IF NOT EXISTS CALLS (
    FROM Chunk TO Chunk
);

CREATE REL TABLE IF NOT EXISTS IMPORTS (
    FROM Chunk TO Chunk
);
"""
# WHY IF NOT EXISTS?
# DB may already exist on resume. Don't fail — be idempotent.
# Same principle as WAL: safe to run multiple times.


# ════════════════════════════════════════════════════════
# STORE
# ════════════════════════════════════════════════════════

class KuzuStore:
    """
    Implements GraphStorePort.
    All KuzuDB interaction lives here.
    Nothing outside this file imports kuzu directly.
    """

    # Max neighbors returned per expansion query.
    # WHY a hard limit?
    # Fan-out control: prevents runaway context growth.
    # k=5 retrieved × 20 neighbors = 100 max additional chunks.
    # All go to reranker — reranker picks best N.
    MAX_NEIGHBORS = 20

    def __init__(self, db_path: str = None):
        db_path = db_path or settings.kuzu_db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True) # Create/Check the parent directory (e.g., 'data/')

        # kuzu.Database opens or creates the file kuzu.db at this path: embedded
        # In-process: no server, no network, direct file access
        self._db   = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)

        self._init_schema()
        logger.info("KuzuStore ready at %s", db_path)

    def _init_schema(self) -> None:
        """
        Run DDL statements to create tables if they don't exist.
        Safe to call on every startup — IF NOT EXISTS is idempotent.
        """
        # KuzuDB executes one statement at a time
        # Split on semicolons, strip whitespace, skip empty
        statements = [
            s.strip()
            for s in _DDL.split(";")
            if s.strip()
        ]
        for stmt in statements:
            self._conn.execute(stmt)
        logger.debug("KuzuStore schema initialized")

    # ── Write ──────────────────────────────────────────

    def add_nodes(self, chunks: list[CodeChunk]) -> None:
        """
        Insert or update Chunk nodes.
        Uses MERGE semantics — idempotent on chunk_id.

        WHY MERGE not CREATE?
        WAL resume may replay a batch. MERGE upserts:
        if node exists → update properties.
        if not → create.
        CREATE would raise duplicate key error on resume.

        """
        if not chunks:
            return
        
        # MERGE query — one per node
        # WHY not batch insert? KuzuDB supports COPY FROM for bulk,
        # but MERGE for upsert requires individual statements. -> Read-Lock-Write
        # For our batch sizes (32 chunks), this is fast enough.
        # TODO: benchmark COPY vs MERGE at 10k chunks if slow.
        merge_query = """
            MERGE (n:Chunk {chunk_id: $chunk_id})
            SET n.repo_name   = $repo_name,
                n.symbol_name = $symbol_name,
                n.chunk_type  = $chunk_type,
                n.language    = $language,
                n.file_path   = $file_path,
                n.start_line  = $start_line,
                n.end_line    = $end_line,
                n.docstring   = $docstring
        """

        for chunk in chunks:
            params = chunk.to_kuzu_node()
            self._conn.execute(merge_query, params)
        
        logger.debug("KuzuStore upserted %d nodes", len(chunks))


    def add_edges(self, edges: list[GraphEdge]) -> None:
        """
        Insert CALLS and IMPORTS edges.

        Only insert if both endpoints exist as nodes.
        WHY check existence?
        graph.py may produce edges where target was not indexed
        (external library that resolved to a chunk_id by mistake).
        Inserting an edge with a missing endpoint → Kuzu error.
        """
        if not edges:
            return

        # MATCH silently returns no rows if either node is missing.
        # MERGE then has nothing to act on — edge is skipped, no error.
        # This is acceptable: silent skip is logged via RETURN 1 + has_next() check.
        calls_query = """
            MATCH (a:Chunk {chunk_id: $source_id}) 
            MATCH (b:Chunk {chunk_id: $target_id})
            MERGE (a)-[:CALLS]->(b)
            RETURN 1
        """
        imports_query = """
            MATCH (a:Chunk {chunk_id: $source_id})
            MATCH (b:Chunk {chunk_id: $target_id})
            MERGE (a)-[:IMPORTS]->(b)
            RETURN 1
        """

        count_upserted = 0

        for edge in edges:
            params = {
                "source_id": edge.source_id,
                "target_id": edge.target_id
            }
            if edge.edge_type == EdgeType.CALLS:
                result = self._conn.execute(calls_query, params)
            elif edge.edge_type == EdgeType.IMPORTS:
                result = self._conn.execute(imports_query, params)
            else:
                continue # Skip unknown edge types

            if result.has_next():
                count_upserted += 1
            

        count_skipped = len(edges) - count_upserted
        logger.debug(
            "KuzuStore upserted %s edges (skipped %s missing endpoints)", 
            count_upserted, 
            count_skipped
        )

    def node_exists(self, chunk_id: str) -> bool:
        """
        Check if a node exists — used by WAL resume logic.
        """
        exists_query = """
            MATCH (n:Chunk {chunk_id: $chunk_id})
            RETURN count(n) > 0 AS exists
        """

        result = self._conn.execute(exists_query, {"chunk_id": chunk_id})
        row = result.fetchone()
        return bool(row and row[0])

    def delete_edges_for_repo(self, repo_name: str) -> None:
        """
        Delete all CALLS and IMPORTS edges where source chunk
        belongs to this repo.

        WHY on source only not target?
        chunk_id format: "repo_name::path::symbol::hash"
        STARTS WITH repo_name identifies ownership unambiguously.
        We delete edges this repo OWNS (sourced from it).
        Cross-repo edges sourced by other repos are untouched.

        Called before Pass 2 on resume — clears partial edge writes
        so Pass 2 can write a clean complete set.
        """
        prefix = f"{repo_name}::"

        for rel in ("CALLS", "IMPORTS"):
            self._conn.execute(f"""
                MATCH (a:Chunk)-[r:{rel}]->()
                WHERE a.chunk_id STARTS WITH $prefix
                DELETE r
            """, {"prefix": prefix})

        logger.info(f"Cleared existing edges for {repo_name}")

    # ── Read ───────────────────────────────────────────

    def expand_neighbors(
        self,
        repo_name: str,
        chunk_ids: list[str],
        limit: int = None,
    ) -> list[str]:
        """
        Single-hop graph expansion.
        Given a list of chunk_ids, return their direct neighbors.

        Traverses both CALLS and IMPORTS edges.
        Returns neighbor chunk_ids only — caller fetches full
        text from LanceDB via fetch_by_ids().

        WHY single-hop only?
          Multi-hop: O(b^d) fan-out, b=branching factor, d=depth.
          At d=2, b=3: 9 neighbors per seed → 45 for k=5 seeds.
          Lost-in-the-middle: LLM attends weakly to middle context.
          Single-hop gives immediate dependencies — sufficient for
          most "how does X work?" queries. User asks follow-ups
          for deeper traversal. Speed and context size win.

        WHY DISTINCT?
          Multiple seeds may share neighbors.
          login → fetch_user, validate → fetch_user
          Without DISTINCT: fetch_user returned twice.

        WHY exclude seeds from results?
          Seeds are already in the retrieved set.
          Returning them as neighbors wastes reranker capacity.

        Args:
            chunk_ids: seed chunk_ids from hybrid search
            limit:     max neighbors to return

        Returns:
            list of neighbor chunk_ids (not including seeds)
        """
        limit = limit or self.MAX_NEIGHBORS

        if not chunk_ids:
            return []

        # Parameterize the IN clause safely
        # WHY not f-string? SQL/Cypher injection.
        # KuzuDB doesn't support list parameters in WHERE IN yet,
        # so we build a safe literal list of quoted strings.
        # chunk_ids come from our own system (chunk_id format enforced
        # by make_chunk_id) — no user input reaches here directly.
        # Still: validate format before interpolating.
        # safe_ids = self._safe_id_list(chunk_ids)

        params = {"repo_name": repo_name, "ids": chunk_ids, "limit": limit}

        # NOTE: KuzuDB >= 0.x supports native list parameters.
        # Using $ids directly instead of building a string literal.
        # _safe_id_list() kept below for reference — not used.

        query = """
            MATCH (a:Chunk)-[:CALLS|IMPORTS]->(b:Chunk)
            WHERE a.repo_name = $repo_name 
              AND b.repo_name = $repo_name
              AND a.chunk_id IN $ids
              AND NOT (b.chunk_id IN $ids)
            RETURN DISTINCT b.chunk_id
            LIMIT $limit
        """.strip()

        try:
            result = self._conn.execute(query, params)
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"KuzuDB expansion failed: {e}")
            raise e

    def get_callers(self, chunk_id: str) -> list[str]:
        """
        Reverse traversal: who calls this chunk?
        Used for impact analysis: "what breaks if I change X?"

        Same single-hop limit applies.

        Not used in main query pipeline yet.
        Exposed for /query endpoint optional parameter.
        """

        limit = self.MAX_NEIGHBORS 

        callers_query = f"""
            MATCH (a:Chunk)-[:CALLS]->(b:Chunk {{chunk_id: $chunk_id}})
            RETURN DISTINCT a.chunk_id AS chunk_id
            LIMIT {self.MAX_NEIGHBORS}
        """

        params = {"chunk_id": chunk_id}
        result = self._conn.execute(callers_query, params)
        return [row[0] for row in result]


    # ── Stats ──────────────────────────────────────────

    def stats(self) -> dict:
        """
        Node and edge counts for /health endpoint.
        """
        nodes_res = self._conn.execute("MATCH (n:Chunk) RETURN count(n)")
        node_cnt = nodes_res.get_next()[0] if nodes_res.has_next() else 0

        calls_res = self._conn.execute("MATCH ()-[e:CALLS]->() RETURN count(e)")
        calls_cnt = calls_res.get_next()[0] if calls_res.has_next() else 0

        imports_res = self._conn.execute("MATCH ()-[e:IMPORTS]->() RETURN count(e)")
        imports_cnt = imports_res.get_next()[0] if imports_res.has_next() else 0

        return {
            "nodes": node_cnt,
            "edges_calls": calls_cnt,
            "edges_imports": imports_cnt,
            "edges_total": calls_cnt + imports_cnt
        }

    def delete_repo_nodes(self, repo_name: str):
        """
        Removes all nodes for a specific repo and all associated edges.
        """
        # DETACH DELETE is the safest way to wipe a tenant.
        # It finds the nodes, clips all their edges, and then deletes the nodes.
        query = "MATCH (n:Chunk {repo_name: $repo_name}) DETACH DELETE n"
    
        try:
            self._conn.execute(query, {"repo_name": repo_name})
            logger.info(f"KuzuDB: Full purge for repo '{repo_name}' successful.")
        except Exception as e:
            logger.error(f"KuzuDB: Failed to purge repo '{repo_name}': {e}")
            raise

    # ── Helpers ────────────────────────────────────────

    # def _safe_id_list(self, chunk_ids: list[str]) -> str:
    #     """
    #     Build a safe Cypher list literal from chunk_ids.

    #     Input:  ["repo::src.auth::login::a1b2", "repo::src.db::fetch::c3d4"]
    #     Output: "'repo::src.auth::login::a1b2', 'repo::src.db::fetch::c3d4'"

    #     WHY validate?
    #     chunk_ids are system-generated but we still check format.
    #     Rejects anything with quotes to prevent injection.
    #     """
    #     validated = []
    #     for cid in chunk_ids:
    #         if "'" in cid or '"' in cid:
    #             logger.warning("Rejecting suspicious chunk_id: %s", cid)
    #             continue
    #         validated.append(f"'{cid}'")
    #     return ", ".join(validated)