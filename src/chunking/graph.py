"""
Pass 2: Resolve RawEdges → GraphEdges

Runs after all files are parsed and chunks are written to storage.

Input:
  raw_edges: list[RawEdge]      — collected during streaming Pass 1
  name_map:  dict[str, list[str]] — symbol_name → list[chunk_id] 
                                    built during Pass 1 from all chunks

Output:
  list[GraphEdge]  — resolved edges, ready for KuzuDB
"""

from src.core.models import RawEdge, GraphEdge, EdgeType
from src.logger import get_logger

logger = get_logger(__name__)


def build_name_map(chunks) -> dict[str, list[str]]:
    """
    Build repo_name::symbol_name → list[chunk_id] lookup.

    Called by indexer.py during streaming Pass 1.
    Updated incrementally as each file is parsed.

    WHY list[str] not str?
    Same symbol name can exist in multiple files per repo.
    We keep all candidates, resolve ambiguity at retrieval time.

    Args:
        chunks: list[CodeChunk] from one file's parse result

    Returns:
        Partial name_map for these chunks.
        indexer.py merges these into the global name_map.
    """
    partial: dict[str, list[str]] = {}
    for chunk in chunks:
        key = f"{chunk.repo_name}::{chunk.symbol_name}"
        partial.setdefault(key, []).append(chunk.chunk_id)
    return partial


def resolve_edges(
    repo_name: str,
    raw_edges: list[RawEdge],
    name_map: dict[str, list[str]],
) -> tuple[list[GraphEdge], int, int]:
    """
    Resolve raw symbol names to chunk_ids.

    Three outcomes per RawEdge:
      1. target_name not in name_map → external library, skip
      2. exactly one candidate → unambiguous, create one edge
      3. multiple candidates → same name in N files,
         create N edges, let reranker sort it out at query time

    Args:
        raw_edges: unresolved edges from Pass 1
        name_map:  complete symbol → chunk_id map (all files parsed)

    Returns:
        edges:       resolved GraphEdge list
        resolved:    count of successfully resolved edges
        unresolved:  count of skipped (external) edges

    WHY return counts?
    indexer.py logs these. High unresolved count =
    lots of external library calls = normal for most repos.
    Low resolved count on a large repo = parser missing calls.
    Both are useful signals for debugging.
    """
    edges: list[GraphEdge] = []
    resolved = 0
    unresolved = 0

    for raw_edge in raw_edges:
        # Extract base name for lookup
        # "self.validator" → "validator"
        # "os.path.join"   → "os" (external, will be skipped)
        # "fetch_user"     → "fetch_user"
        target_base_name = _extract_base_name(raw_edge.target_name)
        target_name = f"{repo_name}::{target_base_name}"

        candidates = name_map.get(target_name, [])

        if not candidates:
            # External library or unresolvable — skip silently
            # We log at debug level only — this is expected behavior
            logger.debug("Unresolved: %s (external or not indexed)", target_name)
            unresolved += 1
            continue

        for candidate_id in candidates:
            # Skip self-loops — a chunk calling itself is recursion,
            # valid but not useful as a graph edge for our purposes
            if candidate_id == raw_edge.source_id:
                continue

            edges.append(GraphEdge(
                source_id=raw_edge.source_id,
                target_id=candidate_id,
                edge_type=raw_edge.edge_type,
            ))

        resolved += 1 # Only counding : how many RawEdges found at least one destination != len(edges)

    logger.info(
        "Edge resolution: %d resolved, %d unresolved (external)",
        resolved, unresolved,
    )
    return edges, resolved, unresolved


def _extract_base_name(raw_name: str) -> str:
    """
    Extract the base symbol name for map lookup.

    "fetch_user"      → "fetch_user"   direct call
    "self.validate"   → "validate"     method call on self
    "obj.method"      → "method"       attribute call
    "os.path.join"    → "os"           external (will miss in name_map)
    "this.handleClick" -> handleClick

    WHY take the last part for self.X but first part for os.X?
    We can't know at this stage. Taking the last dotted segment
    works for self.method, this.method and obj.method.
    os.path.join → "join" would incorrectly match any local "join".
    Taking first segment → "os" → not in name_map → false-positive so skipped. Safer.

    Current approach: take last segment if starts with "self" or "this" or
    otherwise take first segment.

    WHY not always last segment?
    "module.fetch_user" → "fetch_user" would match any fetch_user
    in any file. Over-resolves. First segment → "module" → not in
    name_map → safely skipped.
    """
    if raw_name.startswith("self.") or raw_name.startswith("this."):
        # self.validate → validate
        return raw_name.split(".", 1)[1]
    # os.path.join → os (likely external)
    # fetch_user   → fetch_user (direct)
    return raw_name.split(".")[0]