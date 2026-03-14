# src/core/models.py
"""
Foundation of the entire system. Zero external deps except pydantic.

Data flow:
  RawFile       → produced by ingestion/loaders.py
  CodeChunk     → produced by chunking/parser.py (Pass 1)
  RawEdge       → produced by chunking/parser.py (Pass 1)
  ParsedRepo    → held in memory between Pass 1 and Pass 2
  GraphEdge     → produced by chunking/graph.py (Pass 2)
  RetrievedChunk→ produced by retrieval/
  QueryResult   → produced by query/pipeline.py
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator
import hashlib
import json


# ════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════

class ChunkType(str, Enum):
    FUNCTION = "function"   # standalone function or method
    CLASS    = "class"      # class header chunk only
    MODULE   = "module"     # top-level code outside any def/class


class EdgeType(str, Enum):
    CALLS   = "CALLS"       # function A calls function B
    IMPORTS = "IMPORTS"     # file A imports from file B


class Language(str, Enum):
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN    = "unknown"  # triggers fallback char-split


class QueryRoute(str, Enum):
    CODE_SEARCH = "code_search"
    CONCEPTUAL  = "conceptual"
    DEBUGGING   = "debugging"
    SETUP       = "setup"


# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════

def make_chunk_id(
    repo: str,
    filepath: str,
    classname: str | None,
    funcname: str,
    signature: str,
) -> str:
    """
    Stable, unique chunk ID.

    Stable: same code after body refactor = same ID
    Unique: same funcname in different class/file = different ID

    Format: {repo}::{safe_path}::{class_or_module}.{func}::{sig_hash}
    """
    class_part = classname or "module"
    sig_hash = hashlib.sha256(signature.encode()).hexdigest()[:8]
    safe_path = filepath.replace("/", ".").replace("\\", ".")
    return f"{repo}::{safe_path}::{class_part}.{funcname}::{sig_hash}"


# ════════════════════════════════════════════════════════
# WRITE PATH — INGESTION
# ════════════════════════════════════════════════════════

class RawFile(BaseModel):
    """
    Output of ingestion/loaders.py.
    Filters (ignore dirs, extensions, size) applied BEFORE
    this model is created. If a RawFile exists, it's valid.
    """
    repo_name:  str        # Tenant Id -> for making WAL repo based namespace instead of global
    path:       str       # relative from repo root
    content:    str       # full file text utf-8
    language:   Language  # detected from extension
    size_bytes: int

    @field_validator("content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("RawFile content cannot be empty")
        return v


# ════════════════════════════════════════════════════════
# WRITE PATH — CHUNKING
# ════════════════════════════════════════════════════════

class CodeChunk(BaseModel):
    """
    Atomic unit of the system. One function, one class header,
    or one module-level block.

    Two-pass population:
      Pass 1 (parser.py):  all fields except vector
      Embedder:            vector added
      Pass 2 (graph.py):   edges resolved separately as GraphEdge
    """
    # ── Identity ──────────────────────────────────────────
    repo_name:   str
    chunk_id:    str
    symbol_name: str
    chunk_type:  ChunkType
    language:    Language
    parent_class: str | None = None
    # Set when chunk_type=FUNCTION and function is a method.
    # Tells LLM: this function lives inside this class.

    # ── Location ──────────────────────────────────────────
    file_path:   str
    start_line:  int        # 1-indexed inclusive
    end_line:    int        # 1-indexed inclusive

    # ── Content ───────────────────────────────────────────
    text:        str        # full source of this chunk
    docstring:   str | None = None

    # ── Raw dependency names (Pass 1 output) ──────────────
    calls_out_raw: list[str] = Field(default_factory=list)
    # Raw symbol names this chunk calls: ["fetch_user", "hash_password"]
    # NOT chunk_ids yet — resolved in Pass 2 by graph.py
    # WHY keep raw names on the chunk?
    # graph.py needs them for resolution. After Pass 2 creates
    # GraphEdges, these raw names have served their purpose.

    imports_raw: list[str] = Field(default_factory=list)
    # Raw import strings: ["from src.auth import verify", "import os"]
    # Used to build IMPORTS edges in Pass 2

    # ── Vector (added by embedder, before storage) ────────
    vector_768: list[float] | None = None
    vector_128: list[float] | None = None
    # None until embedding/embedder.py runs.
    # to_lance_record() raises if this is still None.

    @field_validator("start_line", "end_line")
    @classmethod
    def lines_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Line numbers are 1-indexed, must be >= 1")
        return v

    @model_validator(mode="after")
    def end_after_start(self) -> CodeChunk:
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line {self.end_line} < start_line {self.start_line}"
            )
        return self

    def to_lance_record(self) -> dict[str, Any]:
        """Projection for LanceDB. Vector must be set."""
        if self.vector_768 is None or self.vector_128 is None:
            raise RuntimeError(
                f"Chunk {self.chunk_id} missing vectors. Embed before storing."
            )
        
        # Symbol + Docstring + Code 
        composite = "\n".join([   
            f"Symbol: {self.symbol_name}",
            f"Docs: {self.docstring or ''}",
            self.text
        ])

        return {
            "chunk_id":     self.chunk_id,
            "symbol_name":  self.symbol_name,
            "repo_name":    self.repo_name,
            "chunk_type":   self.chunk_type.value,
            "language":     self.language.value,
            "file_path":    self.file_path,
            "start_line":   self.start_line,
            "end_line":     self.end_line,
            "text":         self.text,
            "docstring":    self.docstring or "",
            "vector_768":   self.vector_768,
            "vector_128":   self.vector_128,
            "calls_out_raw": json.dumps(self.calls_out_raw), 
            "imports_raw":   json.dumps(self.imports_raw),
            "search_content": composite
        }

    def to_kuzu_node(self) -> dict[str, Any]:
        """
        Projection for KuzuDB node.
        No vector — Kuzu is for graph structure only.
        No calls_out_raw — edges are stored as GraphEdge, not here.
        """
        return {
            "chunk_id":    self.chunk_id,
            "symbol_name": self.symbol_name,
            "repo_name":   self.repo_name,
            "chunk_type":  self.chunk_type.value,
            "language":    self.language.value,
            "file_path":   self.file_path,
            "start_line":  self.start_line,
            "end_line":    self.end_line,
            "docstring":   self.docstring or "",
        }


class RawEdge(BaseModel):
    """
    Unresolved edge from Pass 1.
    source_id known. target_name is a symbol string, not yet a chunk_id.
    graph.py resolves these into GraphEdge in Pass 2.
    """
    source_id:   str        # chunk_id of caller — known after Pass 1
    target_name: str        # raw symbol name — "fetch_user", "os.path"
    edge_type:   EdgeType


class GraphEdge(BaseModel):
    """
    Resolved edge. Both endpoints are chunk_ids.
    Written to KuzuDB by storage/kuzu_store.py.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType


class ParsedRepo(BaseModel):
    """
    Intermediate state between Pass 1 and Pass 2.
    Lives in memory only. Never written to disk.

    Pass 1 fills chunks + raw_edges.
    Pass 2 (graph.py) reads this, resolves raw_edges → GraphEdge.
    """
    repo_name: str
    chunks:    list[CodeChunk]    # all chunks, vector=None at this point
    raw_edges: list[RawEdge]      # unresolved, target is symbol name


# ════════════════════════════════════════════════════════
# READ PATH
# ════════════════════════════════════════════════════════

class RetrievedChunk(BaseModel):
    """
    A chunk that came back from retrieval, with scores.
    Wraps CodeChunk rather than inheriting — scores are
    read-path metadata, not part of the chunk's identity.
    """
    chunk:          CodeChunk
    vector_score:   float | None = None
    bm25_score:     float | None = None
    rrf_score:      float | None = None
    reranker_score: float | None = None
    source: str = "hybrid"
    # "vector", "bm25", "graph_expand", "hybrid"
    # tracked so eval can measure graph expansion contribution


# ════════════════════════════════════════════════════════
# OBSERVABILITY
# ════════════════════════════════════════════════════════

class StepMetrics(BaseModel):
    step_name:       str
    latency_ms:      float
    input_count:     int
    output_count:    int
    memory_delta_mb: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)


class QueryMetrics(BaseModel):
    embed_query:   StepMetrics | None = None
    hybrid_search: StepMetrics | None = None
    graph_expand:  StepMetrics | None = None
    fetch_chunks:  StepMetrics | None = None
    reranker:      StepMetrics | None = None
    llm_generate:  StepMetrics | None = None
    validation:    StepMetrics | None = None

    total_latency_ms:  float = 0.0
    tokens_used:       int   = 0
    chunks_retrieved:  int   = 0
    graph_nodes_added: int   = 0
    reranker_scores:   list[float] = Field(default_factory=list)
    mrl_dimensions:    int   = 768


# ════════════════════════════════════════════════════════
# RESPONSE
# ════════════════════════════════════════════════════════

class QueryResult(BaseModel):
    question:         str
    answer:           str
    repo_name:        str
    route:            QueryRoute
    sources:          list[RetrievedChunk]
    metrics:          QueryMetrics
    faithfulness:     float | None = None
    answer_relevance: float | None = None