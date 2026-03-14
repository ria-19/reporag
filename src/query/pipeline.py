"""
Read path orchestrator.

Receives a QueryRequest, returns a QueryResult.
Knows the ORDER of steps, not the implementation of any step.
Each step is a black box injected via constructor.

Steps:
  1.  Parse filters from request
  2.  Exact symbol lookup (if --symbol provided)
  3.  Route query → prompt template
  4.  Embed query → (vector_128, vector_768)
  5.  Hybrid search with optional pre-filters
  6.  Graph expand neighbors
  7.  Fetch neighbor text from LanceDB
  8.  Merge + deduplicate all chunks
  9.  Assemble context string
  10. Generate answer
  11. Validate (cosine always, LLM judge sampled)
  12. Collect metrics, return QueryResult

WHY pipeline.py does not instantiate anything:
  Dependency Injection — every component is passed in.
  Swap LLM: pass different LLMPort. This file never changes.
  Test: pass mock stores. No real DB needed in unit tests.
"""

from __future__ import annotations

import time
from typing import Protocol

from src.core.models import (
    CodeChunk, RetrievedChunk,
    QueryResult, QueryMetrics, QueryRoute,
)
from src.query.router import QueryRouter, ContextAssembler
from src.query.validator import CosineValidator, LLMJudge
from src.observability.metrics import Timer
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# ════════════════════════════════════════════════════════
# PORTS — what pipeline needs from the outside world
# Defined here because pipeline owns the read path.
# Concrete implementations live in storage/ and llm.py.
# ════════════════════════════════════════════════════════

class VectorStoreReadPort(Protocol):
    def hybrid_search_filtered(
        self,
        vector_128:         list[float],
        vector_768:         list[float],
        query_text:         str,
        repo_name:          str,
        k:                  int | None,
        candidate_k:        int | None,
        symbol:             str | None = None,
        filename:           str | None = None,
        language:           str | None = None, 
        chunk_type:         str | None = None
    ) -> list[RetrievedChunk]: ...

    def fetch_by_symbol(
        self,
        symbol_name: str,
        repo_name: str,
        filename:    str | None = None, 
        language:    str | None = None,
    ) -> list[CodeChunk]: ...

    def fetch_by_ids(
        self,
        chunk_ids: list[str],
    ) -> list[CodeChunk]: ...


class GraphStoreReadPort(Protocol):
    def expand_neighbors(
        self,
        repo_name: str,
        chunk_ids: list[str],
        limit: int,
    ) -> list[str]: ...


class QueryEmbedderPort(Protocol):
    def embed_query(
        self,
        text: str,
    ) -> tuple[list[float], list[float]]: ...
    # Returns (vector_768, vector_128)


class LLMPort(Protocol):
    def generate(
        self,
        prompt: str,
    ) -> tuple[str, int]: ...
    # Returns (answer_text, tokens_used)


# ════════════════════════════════════════════════════════
# QUERY REQUEST
# Input shape for the read path.
# Lives here not in core/models.py because it is an
# interface concern — different interfaces (HTTP, CLI, MCP)
# may have different input shapes that all funnel into
# the same pipeline.
# api.py constructs QueryRequest from HTTP body.
# mcp_server.py constructs it from tool call args.
# ════════════════════════════════════════════════════════

from pydantic import BaseModel

class QueryRequest(BaseModel):
    """
    Validated input to QueryPipeline.query().

    question:  the user's natural language query
    symbol:    explicit symbol filter (--symbol calculate_tax)
    filename:  explicit file filter   (--file billing/tax.py)
    language:  explicit lang filter   (--lang python)
    k:         number of final chunks to retrieve
    """
    question:  str
    repo_name: str
    symbol:    str | None = None
    filename:  str | None = None
    language:  str | None = None
    k:         int        = 5

    # WHY no repo filter here?
    # For v1: one repo indexed at a time.
    # Multi-repo support: add repo: str | None = None later.


# ════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════

# Max symbol results before we fall back to hybrid search.
# WHY 10? More than 10 exact matches = ambiguous symbol name
# (e.g. __init__, get, run). Hybrid search ranks better
# than returning 10 unordered exact matches.
MAX_SYMBOL_RESULTS = 10


class QueryPipeline:
    """
    Orchestrates the full read path.
    Stateless across requests — all per-request state is local
    to query(). Safe for concurrent use.
    """

    def __init__(
        self,
        embedder:     QueryEmbedderPort,
        vector_store: VectorStoreReadPort,
        graph_store:  GraphStoreReadPort,
        llm:          LLMPort,
        cosine_validator: CosineValidator,
        llm_judge:        LLMJudge,
    ):
        # Injected — never instantiated here
        self.embedder     = embedder
        self.vector_store = vector_store
        self.graph_store  = graph_store
        self.llm          = llm
        self.cosine_validator = cosine_validator
        self.llm_judge        = llm_judge

        # These are stateless utilities — safe to share across requests
        # WHY instantiate here and not inject?
        # QueryRouter and ContextAssembler have no external dependencies.
        # No reason to inject them — they can't be swapped meaningfully.
        self.router    = QueryRouter()
        self.assembler = ContextAssembler()

    def query(self, request: QueryRequest) -> QueryResult:
        """
        Execute full read path for one query.

        All per-request state (metrics, chunks, timers) is local. else Global State Leak
        WHY local? Concurrent requests must not share state.
        If metrics were on self, two parallel queries corrupt each other.
        """
        # Fresh metrics object per request
        metrics = QueryMetrics()
        total_start = time.perf_counter()

        # ── Step 1: Parse filters ─────────────────────────
        # Already parsed by Pydantic — request.symbol, .filename, .language
        # No processing needed. Pydantic validated on construction.
        k           = request.k
        candidate_k = k * settings.retrieval_k_fetch_multiplier

        # ── Step 2: Exact symbol lookup ───────────────────
        # Only if user explicitly provided --symbol
        # WHY explicit only? Auto-detection adds latency + uncertainty.
        # User already knows the symbol name — let them tell us.
        symbol_chunks: list[CodeChunk] = []

        if request.symbol:
            with Timer("symbol_lookup") as t:
                symbol_chunks = self.vector_store.fetch_by_symbol(
                    symbol_name=request.symbol,
                    repo_name=request.repo_name,
                    filename=request.filename,
                    language=request.language
                )

            if len(symbol_chunks) > MAX_SYMBOL_RESULTS:
                # Too broad — __init__, get, run etc.
                # Fall through to hybrid search
                logger.debug(
                    "Symbol '%s' → %d results (> %d threshold), "
                    "falling back to hybrid search",
                    request.symbol, len(symbol_chunks), MAX_SYMBOL_RESULTS,
                )
                symbol_chunks = []
            else:
                logger.debug(
                    "Symbol '%s' → %d exact matches",
                    request.symbol, len(symbol_chunks),
                )

        # ── Step 3: Route query ───────────────────────────
        route, prompt_template = self.router.route(request.question)
        logger.debug("Route: %s", route.value)

        # ── Step 4: Embed query ───────────────────────────
        # Always embed — needed for hybrid search and cosine validation
        # Even if symbol lookup succeeded, we need vectors for validation
        with Timer("embed_query") as t:
            v768, v128 = self.embedder.embed_query(request.question)

        metrics.embed_query = t.to_step_metrics(
            input_count=1,
            output_count=2,  # two vectors produced
            model=settings.embedding_model,
            dimensions_768=len(v768),
            dimensions_128=len(v128),
        )

        # ── Step 5: Hybrid search ─────────────────────────
        # Skip if exact symbol lookup gave us enough results
        # WHY skip? Symbol results are already exact matches.
        # Hybrid search would add noise, not signal.
        hybrid_chunks: list[RetrievedChunk] = []

        if not symbol_chunks:
            with Timer("hybrid_search") as t:
                hybrid_chunks = self.vector_store.hybrid_search_filtered(
                    vector_128=v128,
                    vector_768=v768,
                    query_text=request.question,
                    repo_name=request.repo_name,
                    k=k,
                    candidate_k=candidate_k,
                    symbol=request.symbol,
                    filename=request.filename,
                    language=request.language,
                )

            metrics.hybrid_search = t.to_step_metrics(
                input_count=1,
                output_count=len(hybrid_chunks),
            )

        # Seed chunks for graph expansion
        # If symbol lookup succeeded: use those
        # Otherwise: use hybrid results
        seed_chunks = (
            [RetrievedChunk(chunk=c, source="symbol_lookup")
             for c in symbol_chunks]
            if symbol_chunks
            else hybrid_chunks
        )
        seed_ids = [rc.chunk.chunk_id for rc in seed_chunks]

        # ── Step 6: Graph expand ──────────────────────────
        # Find direct neighbors (callees + importees) of seed chunks
        neighbor_ids: list[str] = []

        with Timer("graph_expand") as t:
            neighbor_ids = self.graph_store.expand_neighbors(
                repo_name=request.repo_name,
                chunk_ids=seed_ids,
                limit=settings.graph_expand_limit,
            )

        metrics.graph_expand = t.to_step_metrics(
            input_count=len(seed_ids),
            output_count=len(neighbor_ids),
        )
        metrics.graph_nodes_added = len(neighbor_ids)

        # ── Step 7: Fetch neighbor chunks ─────────────────
        # Get full text for neighbor IDs from LanceDB
        neighbor_chunks: list[RetrievedChunk] = []

        if neighbor_ids:
            with Timer("fetch_chunks") as t:
                fetched = self.vector_store.fetch_by_ids(neighbor_ids)

            neighbor_chunks = [
                RetrievedChunk(chunk=c, source="graph_expand")
                for c in fetched
            ]

            metrics.fetch_chunks = t.to_step_metrics(
                input_count=len(neighbor_ids),
                output_count=len(neighbor_chunks),
            )

        # ── Step 8: Merge + deduplicate ───────────────────
        # Combine seed chunks + neighbor chunks
        # Dedup on chunk_id — graph expand may return chunks
        # already in seed set (common in dense call graphs)
        all_chunks = seed_chunks + neighbor_chunks
        seen_ids: set[str] = set()
        deduped: list[RetrievedChunk] = []

        for rc in all_chunks:
            if rc.chunk.chunk_id not in seen_ids:
                deduped.append(rc)
                seen_ids.add(rc.chunk.chunk_id)

        metrics.chunks_retrieved = len(deduped)
        logger.debug(
            "Chunks after dedup: %d (seeds=%d, neighbors=%d)",
            len(deduped), len(seed_chunks), len(neighbor_chunks),
        )

        # ── Step 9: Assemble context ──────────────────────
        context_str, chunks_included = self.assembler.assemble(
            chunks=deduped,
            query=request.question,
        )

        # Build final prompt
        prompt = prompt_template.format(
            context=context_str,
            question=request.question,
        )

        # ── Step 10: Generate ─────────────────────────────
        with Timer("llm_generate") as t:
            answer, tokens_used = self.llm.generate(prompt)

        metrics.llm_generate = t.to_step_metrics(
            input_count=len(prompt),
            output_count=len(answer),
            tokens_used=tokens_used,
            route=route.value,
            model=settings.llm_model,
        )
        metrics.tokens_used = tokens_used

        # ── Step 11: Validate ─────────────────────────────
        # Cosine: always — fast, no LLM call
        # LLM judge: sampled — expensive, runs on sample_rate fraction
        cosine_score = self.cosine_validator.score(
            answer=answer,
            context=context_str,
        )

        llm_faithfulness = self.llm_judge.score_faithfulness(
            question=request.question,
            answer=answer,
            context=context_str,
        )
        llm_relevance = self.llm_judge.score_relevance(
            question=request.question,
            answer=answer,
        )

        # ── Step 12: Finalize metrics ─────────────────────
        metrics.total_latency_ms = (
            (time.perf_counter() - total_start) * 1000
        )
        metrics.mrl_dimensions = len(v128)

        logger.info(
            "Query complete | route=%s | chunks=%d | "
            "latency=%.0fms | tokens=%d | cosine=%.2f",
            route.value,
            metrics.chunks_retrieved,
            metrics.total_latency_ms,
            metrics.tokens_used,
            cosine_score,
        )

        return QueryResult(
            question=request.question,
            answer=answer,
            repo_name=request.repo_name,
            route=route,
            sources=deduped,
            metrics=metrics,
            faithfulness=cosine_score,
            # LLM scores: None if not sampled this request
            answer_relevance=llm_relevance,
        )
