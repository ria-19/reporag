"""
Text → vectors using nomic-embed-text-v1.5.

Produces two vectors per chunk:
  vector_768: full MRL embedding, used for precise re-scoring
  vector_128: truncated + re-normalized, used for fast candidate search

WHY re-normalize after truncation?
  After taking first 128 dims, the vector is no longer unit length.
  Cosine similarity assumes unit vectors.
  L2 normalize restores this property.
  Without normalization: similarity scores are wrong.

WHY outside LanceDB?
  1. Transparency: we time this step independently
  2. MRL control: truncation logic lives here, not inside the DB
  3. Flexibility: swap model by changing this file only

NOTE: nomic-embed-text-v1.5 applies LayerNorm internally
via sentence-transformers pooling config before we see the vectors.
If switching to raw HuggingFace transformers API, you must call
F.layer_norm(hidden_state, hidden_state.shape) explicitly
before truncation. sentence-transformers handles this for us.
"""

from __future__ import annotations

import time
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger
from src.observability.metrics import StepMetrics


logger = get_logger(__name__)


# Nomic requires a task prefix on the text before embedding.
# WHY prefixes?
# The model was trained with prefixes to distinguish task types.
# Wrong prefix = wrong embedding space = worse retrieval.
# "search_document" for chunks being indexed.
# "search_query"    for queries at search time.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX    = "search_query: "

# MRL dimensions we use
DIM_FULL = 768
DIM_FAST = 128

class NomicEmbedder:
    """
    Wraps nomic-embed-text-v1.5.
    Produces (vector_768, vector_128) pairs per text.
    """

    def __init__(self, device: str = None):
        device = device or settings.embedding_device
        logger.info(
            "Loading embedding model: %s on %s",
            settings.embedding_model, device
        )
        # trust_remote_code required by nomic model
        # WHY? Nomic uses custom model code not in transformers core.
        # Safe here — we're loading a known model from HuggingFace.
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=device,
            trust_remote_code=True,
        )
        logger.info("Embedder ready")

    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Embed a batch of texts.
        Returns (vectors_768, vectors_128) — parallel lists.

        Args:
            texts:    list of raw text strings
            is_query: True for query-time embedding,
                      False for indexing (different prefix)

        Returns:
            vectors_768: list of 768d vectors
            vectors_128: list of 128d vectors, normalized
        
        WHY return both from one call?
        One forward pass produces 768d.
        128d is just a slice — free to compute.
        Two separate calls would double the inference cost.
        """
        if not texts:
            return [], []

        t0 = time.perf_counter()

        # Add task prefix — required by nomic model
        prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
        prefixed = [prefix + t for t in texts]

        # Single forward pass — produces full 768d embeddings
        # normalize_embeddings=True: unit vectors for cosine similarity
        embeddings = self.model.encode(
            prefixed,
            batch_size=len(texts),
            normalize_embeddings=True,   # L2 normalize full vectors (768d)
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # embeddings shape: (len(texts), 768)

        # Full 768d vectors — already normalized internally
        vectors_768 = embeddings.tolist()

        # 128d truncation + re-normalize
        # Step 1: slice first 128 dimensions
        truncated = embeddings[:, :DIM_FAST]   # shape: (N, 128)

        # Step 2: re-normalize each vector to unit length
        # WHY: slicing breaks unit length property
        # np.linalg.norm axis=1 → per-row norm
        # keepdims=True → (N,1) so division broadcasts correctly
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors (shouldn't happen)
        norms = np.maximum(norms, 1e-10)
        normalized_128 = (truncated / norms).tolist()

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Embedded %d texts in %.1fms (%.2fms/text)",
            len(texts), latency_ms, latency_ms / len(texts),
        )

        return vectors_768, normalized_128

    def embed_query(self, text: str) -> tuple[list[float], list[float]]:
        """
        Embed a single query string.
        Returns (vector_768, vector_128).
        Used at search time — different prefix than documents.
        """
        vecs_768, vecs_128 = self.embed_batch([text], is_query=True)
        return vecs_768[0], vecs_128[0]