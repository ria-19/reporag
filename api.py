"""
HTTP interface. Four endpoints:
  POST /index   — ingest + index a repo
  POST /query   — answer a question
  GET  /eval    — run golden dataset, return metrics
  GET  /health  — system stats

Owns:
  IndexRequest  — write path interface model
  QueryRequest  — read path interface model (imported from pipeline.py)

Does NOT own:
  Business logic — all delegated to Indexer and QueryPipeline
  Domain models  — imported from src/core/models.py

Startup (lifespan):
  All heavy objects constructed once:
  NomicEmbedder, LanceStore, KuzuStore, Indexer, QueryPipeline
  Attached to app.state — shared across all requests.
  WHY once? NomicEmbedder loads a 500MB model.
  LanceStore opens a file handle.
  Recreating per-request = 2s latency per call.

Teardown (lifespan, after yield):
  Close DB connections, release resources.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.loaders import GitHubRepoLoader
from src.chunking.parser import CodeParser
from src.embedding.embedder import NomicEmbedder
from src.storage.lance_store import LanceStore
from src.storage.kuzu_store import KuzuStore
from src.query.validator import CosineValidator, LLMJudge
from src.query.pipeline import QueryPipeline, QueryRequest
from src.indexer import Indexer
from src.llm import build_llm
from src.config import settings
from src.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════
# LIFESPAN
# Runs once at startup, once at shutdown.
# Everything between startup and yield is available
# for the entire lifetime of the app.
# Everything after yield is teardown.
#
# WHY asynccontextmanager?
# FastAPI lifespan must be an async context manager.
# Our setup is I/O bound (opening files, loading models)
# — async allows other tasks to run during setup if needed.
# In practice our setup is sequential, but the interface
# requires async.
# ════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Build and attach all shared objects to app.state.
    Yield → app serves requests.
    After yield → teardown.
    """
    setup_logging()                      
    settings.validate_for_startup()      
    logger.info("Starting RepoRAG API")

    # ── Build shared components ───────────────────────
    # Order matters: components that depend on others
    # must be built after their dependencies.

    logger.info("Loading embedding model...")
    embedder = NomicEmbedder(device=settings.embedding_device)
    # ~2s, loads 500MB model into RAM or VRAM
    # Done once here — never again during request handling

    logger.info("Opening LanceDB...")
    lance_store = LanceStore(db_path=settings.lance_db_path)

    logger.info("Opening KuzuDB...")
    kuzu_store = KuzuStore(db_path=settings.kuzu_db_path)

    logger.info("Building LLM...")
    llm = build_llm()
    # Reads settings.llm_provider → OllamaLLM or GeminiLLM

    # Validators share the embedder — no second model load
    cosine_validator = CosineValidator(embedder=embedder)
    llm_judge        = LLMJudge(
        llm=llm,
        sample_rate=settings.llm_judge_sample_rate,
    )

    # Write path
    indexer = Indexer(
        loader=GitHubRepoLoader(),
        parser=CodeParser(),
        embedder=embedder,
        vector_store=lance_store,
        graph_store=kuzu_store,
    )

    # Read path
    pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=lance_store,
        graph_store=kuzu_store,
        llm=llm,
        cosine_validator=cosine_validator,
        llm_judge=llm_judge,
    )

    # Attach to app.state — accessible in every route handler
    # via request.app.state.<name>
    app.state.indexer  = indexer
    app.state.pipeline = pipeline
    app.state.lance    = lance_store
    app.state.kuzu     = kuzu_store

    logger.info("RepoRAG API ready")

    yield
    # ── Teardown ──────────────────────────────────────
    # Code here runs after the last request is handled
    # and the server is shutting down.
    logger.info("Shutting down RepoRAG API")
    # LanceDB and KuzuDB close file handles when GC collects them.
    # Explicit close if they expose a .close() method:
    # lance_store.close()
    # kuzu_store.close()


# ════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════

app = FastAPI(
    title="RepoRAG",
    description="Local-first codebase intelligence system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════
# INTERFACE MODELS
# Input/output shapes for HTTP — not domain models.
# Live here because they are interface concerns.
# ════════════════════════════════════════════════════════

class IndexRequest(BaseModel):
    """
    Write path input.
    User tells us what to index.
    """
    github_url: str
    repo_name:  str
    force:      bool  = False
    # force=True: clear WAL, re-index from scratch


class IndexResponse(BaseModel):
    repo:            str
    files_processed: int
    files_skipped:   int     
    chunks_indexed:  int
    elapsed_seconds: float


class QueryResponse(BaseModel):
    """
    Wraps QueryResult for HTTP serialization.
    We don't expose the full QueryResult directly —
    we control exactly what the API returns.
    """
    question:         str
    answer:           str
    route:            str
    sources:          list[dict]   # simplified — file_path + symbol_name + snippet
    metrics:          dict         # full QueryMetrics as dict
    faithfulness:     float | None
    answer_relevance: float | None


# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

@app.post("/index", response_model=IndexResponse)
def index_repo(req: IndexRequest, request: Request):
    """
    Clone and index a GitHub repository.

    Steps (handled by Indexer, not here):
      git clone → parse → embed → write to Lance + Kuzu

    Returns indexing summary with chunk count and timing.
    """
    indexer: Indexer = request.app.state.indexer

    try:
        # TODO: git clone req.github_url to a temp dir first
        # For now: assume repo_name is a local path
        # git clone support: add to loaders.py
        result = indexer.index_repo(
            repo_path=req.github_url,   # swap with cloned path
            repo_name=req.repo_name,
            force=req.force,
        )
    except Exception as e:
        logger.error("Indexing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return IndexResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    """
    Answer a question about the indexed codebase.

    Accepts optional filters:
      symbol:   exact function/class name
      filename: partial file path match
      language: python | javascript | typescript

    Returns answer + sources + full observability metrics.
    """
    pipeline: QueryPipeline = request.app.state.pipeline

    try:
        result = pipeline.query(req)
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Serialize sources — expose only what's useful to the caller
    sources = [
        {
            "file_path":    rc.chunk.file_path,
            "symbol_name":  rc.chunk.symbol_name,
            "chunk_type":   rc.chunk.chunk_type.value,
            "start_line":   rc.chunk.start_line,
            "end_line":     rc.chunk.end_line,
            "snippet":      rc.chunk.text[:200] + "..."
                            if len(rc.chunk.text) > 200
                            else rc.chunk.text,
            "rrf_score":    rc.rrf_score,
            "source":       rc.source,
        }
        for rc in result.sources
    ]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        route=result.route.value,
        sources=sources,
        metrics=result.metrics.model_dump(),
        faithfulness=result.faithfulness,
        answer_relevance=result.answer_relevance,
    )


@app.get("/health")
def health(request: Request):
    """
    System stats: chunk counts, graph counts, last indexed.
    Used to verify the index is populated before querying.

    Returns immediately — no heavy computation.
    """
    lance: LanceStore = request.app.state.lance
    kuzu:  KuzuStore  = request.app.state.kuzu

    lance_stats = lance.stats()
    kuzu_stats  = kuzu.stats()

    return {
        "status": "ok",
        "lance":  lance_stats,
        "kuzu":   kuzu_stats,
    }


@app.get("/eval")
def eval_endpoint(request: Request):
    """
    Run golden dataset evaluation.
    Returns Precision@5, Context Recall, Faithfulness, Answer Relevance.

    Delegates to eval.py logic exposed as a function.
    WHY not inline here? eval.py is also a standalone script.
    Same function, two entry points: CLI and HTTP.
    """
    pipeline: QueryPipeline = request.app.state.pipeline

    try:
        from eval import run_eval
        results = run_eval(pipeline)
    except Exception as e:
        logger.error("Eval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return results
