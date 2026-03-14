"""
Response quality scoring.

Two validators, different cost/signal tradeoff:

1. CosineValidator (fast, ~5ms)
   Embeds answer + context, computes cosine similarity.
   Measures: semantic overlap between answer and retrieved context.
   Diagnostic: LOW score → retrieval problem (wrong chunks).
   Always runs.

2. LLMJudge (slow, ~500ms + tokens)
   Sends (question, answer, context) to LLM.
   Asks: is this answer grounded in the context?
   Diagnostic: HIGH cosine + LOW judge → hallucination problem.
   Runs on sample_rate fraction of queries only.

WHY both?
   Together they localize failure:
   Low cosine:              retrieval issue
   High cosine + low judge: generation/prompt issue
   Both high:               pipeline working correctly
"""

from __future__ import annotations

import random
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class CosineValidator:
    """
    Fast semantic overlap scoring.
    Reuses the embedding model — no extra model to load.
    """

    def __init__(self, embedder):
        # Receive embedder — don't instantiate here
        # WHY: embedder is already loaded in pipeline.py
        # Loading it again wastes 500MB RAM and 2s startup
        self._embedder = embedder

    def score(
        self,
        answer:  str,
        context: str,
    ) -> float:
        """
        Cosine similarity between answer and context embeddings.
        Returns float in [-1, 1], typically [0.3, 0.95] for good answers.

        Interpretation:
          > 0.7: answer well-grounded in context
          0.5-0.7: partial overlap, some grounding
          < 0.5: answer diverged from context — retrieval likely off
        """
        if not answer.strip() or not context.strip():
            return 0.0

        # Use document prefix for both — we're comparing content, not query
        vec_answer,  _ = self._embedder.embed_batch([answer])
        vec_context, _ = self._embedder.embed_batch([context[:2000]])
        # WHY truncate context? Context can be 32k chars.
        # Embedding a 32k string is slow and diluted.
        # First 2000 chars (the highest-scored chunks) are representative.

        a = np.array(vec_answer[0])
        c = np.array(vec_context[0])
        # Both unit vectors (normalized in embedder) → dot = cosine
        return float(np.dot(a, c))


class LLMJudge:
    """
    LLM-based faithfulness scorer.
    Expensive — runs on sample_rate fraction of queries only.

    Faithfulness: is the answer supported by the context?
    Answer relevance: does the answer address the question?
    """

    # Prompt designed for a single-token response: a float 0.0-1.0
    # WHY single token? Minimize latency and tokens used.
    _FAITHFULNESS_PROMPT = """\
You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Rate the faithfulness of the answer on a scale from 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.5: Answer is partially supported, some claims not in context
- 0.0: Answer contradicts or ignores the context entirely

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

    _RELEVANCE_PROMPT = """\
You are evaluating whether an answer addresses the question asked.

Question: {question}

Answer: {answer}

Rate the relevance on a scale from 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.5: Answer partially addresses the question
- 0.0: Answer does not address the question at all

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

    def __init__(self, llm, sample_rate: float = 0.1):
        self._llm         = llm
        self._sample_rate = sample_rate
        # WHY 0.1 default? 10% sampling = reasonable signal
        # without doubling token usage. Adjust in config.

    def should_run(self) -> bool:
        """Probabilistic sampling — run on sample_rate fraction."""
        return random.random() < self._sample_rate

    def score_faithfulness(
        self,
        question: str,
        answer:   str,
        context:  str,
    ) -> float | None:
        """
        Returns faithfulness score [0,1] or None if not sampled.
        None means: not evaluated this query, not a failure.
        """
        if not self.should_run():
            return None

        prompt = self._FAITHFULNESS_PROMPT.format(
            context=context[:3000],  # truncate for token budget
            question=question,
            answer=answer,
        )
        try:
            raw, _ = self._llm.generate(prompt)
            return float(raw.strip())
        except (ValueError, TypeError):
            logger.warning("LLM judge returned non-numeric: %s", raw[:50])
            return None

    def score_relevance(
        self,
        question: str,
        answer:   str,
    ) -> float | None:
        """Returns answer relevance score [0,1] or None if not sampled."""
        if not self.should_run():
            return None

        prompt = self._RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
        )
        try:
            raw, _ = self._llm.generate(prompt)
            return float(raw.strip())
        except (ValueError, TypeError):
            logger.warning("LLM judge returned non-numeric: %s", raw[:50])
            return None
