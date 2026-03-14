"""
Central configuration via environment variables.

All settings read from environment or .env file.
Pydantic BaseSettings:
  - reads from environment variables automatically
  - validates types (int, float, bool) — no manual casting
  - raises on startup if required vars are missing
  - .env file loaded via model_config

WHY one settings object not scattered os.getenv() calls?
  Single source of truth.
  Type-validated at startup — fail fast, not at query time.
  Testable: swap settings by setting env vars in test.
  Documentable: all config in one place.

Usage:
  from src.config import settings
  settings.lance_db_path
  settings.llm_provider

Never instantiate Settings() directly outside this file.
Import the singleton: from src.config import settings
"""

from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from pathlib import Path


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # WHY extra="ignore"?
        # .env files often have comments or unrelated vars.
        # Don't crash on unknown fields.
        extra="ignore",
    )

    # ── Paths ─────────────────────────────────────────
    lance_db_path: str = Field(
        default="data/lance",
        description="LanceDB storage directory",
    )
    kuzu_db_path: str = Field(
        default="data/kuzu.db",
        description="KuzuDB storage file",
    )
    wal_path: str = Field(
        default="data/wal.jsonl",
        description="Write-ahead log file path",
    )

    # ── Embedding ─────────────────────────────────────
    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="HuggingFace model ID for embeddings",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding model: cpu | cuda | mps",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Chunks per embedding forward pass",
    )

    # ── Chunking ──────────────────────────────────────
    chunk_size: int = Field(
        default=3000,
        description="Max chars per chunk (fallback strategy)",
    )
    batch_size: int = Field(
        default=32,
        description="Chunks per storage flush in indexer",
    )

    # ── LLM ───────────────────────────────────────────
    llm_provider: str = Field(
        default="ollama",
        description="LLM backend: ollama | gemini",
    )
    llm_model: str = Field(
        default="llama3.2",
        description="Model name for Ollama",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model name when provider=gemini",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Required when llm_provider=gemini",
    )

    # ── Retrieval ─────────────────────────────────────
    retrieval_k: int = Field(
        default=5,
        description="Final chunks returned per query",
    )
    retrieval_k_fetch_multiplier: int = Field(
        default=10,
        description=(
            "Stage 1 fetches k * multiplier candidates. "
            "Higher = better recall, slower Stage 1."
        ),
    )
    graph_expand_limit: int = Field(
        default=20,
        description="Max neighbor chunk_ids from single-hop expansion",
    )

    # ── Validation ────────────────────────────────────
    llm_judge_sample_rate: float = Field(
        default=0.1,
        description=(
            "Fraction of queries evaluated by LLM judge. "
            "0.1 = 10%. Set 0.0 to disable, 1.0 for all queries."
        ),
    )

    # ── API ───────────────────────────────────────────
    allowed_origins: str = Field(
        default="*",
        description=(
            "Comma-separated CORS origins. "
            "Use * for local dev, restrict in production."
        ),
    )

    # ── Logging ───────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR",
    )

    # ── Computed ──────────────────────────────────────
    @computed_field
    @property
    def allowed_origins_list(self) -> list[str]:
        """
        Split comma-separated origins string into list.
        api.py needs list[str] for CORSMiddleware.
        WHY computed? CORSMiddleware takes list, env vars are strings.
        Parse once here, use everywhere.

        "http://localhost:3000,http://localhost:8080"
        → ["http://localhost:3000", "http://localhost:8080"]
        "*" → ["*"]
        """
        return [o.strip() for o in self.allowed_origins.split(",")]

    def validate_for_startup(self) -> None:
        """
        Fail-fast checks that can't be expressed as field validators.
        Call once in lifespan() before building any components.

        WHY separate method not validators?
        Some checks require cross-field logic (gemini_api_key required
        only when llm_provider=gemini). Pydantic field validators
        run per-field without access to other fields easily.
        model_validator exists but this is clearer.
        """
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY must be set when LLM_PROVIDER=gemini. "
                "Add it to .env or set as environment variable."
            )

        if self.embedding_device not in ("cpu", "cuda", "mps"):
            raise ValueError(
                f"EMBEDDING_DEVICE must be cpu | cuda | mps, "
                f"got: {self.embedding_device}"
            )

        if not 0.0 <= self.llm_judge_sample_rate <= 1.0:
            raise ValueError(
                "LLM_JUDGE_SAMPLE_RATE must be between 0.0 and 1.0"
            )

        # Ensure data directory for Lance exist
        Path(self.lance_db_path).mkdir(parents=True, exist_ok=True)

        Path(self.kuzu_db_path).parent.mkdir(parents=True, exist_ok=True)

        wal_parent = Path(self.wal_path).parent
        wal_parent.mkdir(parents=True, exist_ok=True)


# Singleton — import this, never instantiate Settings() directly
settings = Settings()