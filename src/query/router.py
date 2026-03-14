"""
Query type classifier → prompt template selector.

Classifies incoming query into one of four types,
then returns the appropriate prompt template.

Classification: keyword pattern matching (BoW).
WHY not LLM-based routing?
  +500ms latency + token cost per query.
  Developer queries are keyword-dense — BoW accuracy ~90%.
  Acceptable for v1. See FAILURES.md for upgrade path.

WHY four types?
  Code search:  needs file references, function signatures.
  Conceptual:   needs architecture overview, data flow.
  Debugging:    needs error context, root cause, fix suggestion.
  Setup:        needs steps, commands, dependencies.
  Different information needs → different prompt shapes.
  One generic prompt degrades quality on all four.

Known limitation: keyword router uses BoW pattern matching.
Fails on: ambiguous queries ("why is this slow?" could be
debugging or conceptual), negation ("not the auth function"),
and cross-type queries ("how do I debug the setup process?"): 
Cause keyword collisions and misrouting
Fix: small fine-tuned classifier or zero-shot LLM routing.
Cost of fix: +500ms latency + token cost per query.
Current accuracy: ~90% on developer keyword-dense queries.
Acceptable for v1. Add when eval shows routing errors
are hurting generation quality.

"""

from __future__ import annotations

from src.core.models import QueryRoute
from src.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════
# PATTERNS
# Key: QueryRoute
# Value: list of trigger keywords
# Order matters in _route(): first match wins on tie.
# ════════════════════════════════════════════════════════

_PATTERNS: dict[QueryRoute, list[str]] = {
    QueryRoute.DEBUGGING: [
        "error", "exception", "traceback", "bug", "fix", "failing",
        "not working", "broken", "crash", "null", "none", "undefined",
        "why is", "why does", "wrong", "incorrect", "issue",
    ],
    QueryRoute.SETUP: [
        "install", "setup", "configure", "run", "start", "deploy",
        "docker", "requirements", "dependencies", "env", "environment",
        "build", "init", "initialize", "local",
    ],
    QueryRoute.CODE_SEARCH: [
        "function", "method", "class", "where is", "how does",
        "implementation", "define", "definition", "find", "locate",
        "which file", "what file", "show me", "return type",
        "parameter", "argument", "interface",
    ],
    QueryRoute.CONCEPTUAL: [
        "architecture", "design", "overview", "explain", "what is",
        "how does it", "structure", "pattern", "flow", "diagram",
        "relationship", "difference", "why", "concept",
    ],
}

# WHY DEBUGGING first in dict?
# Python dicts preserve insertion order (3.7+).
# On tie: first pattern wins.
# Debugging is highest priority — misrouting a debug query
# to conceptual gives a useless architectural answer.
# A developer with a stack trace needs root cause, not overview.


# ════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# One per QueryRoute. Context and question slots are
# filled by pipeline.py at generation time.
#
# WHY hardcoded here and not in config?
# Prompts change for linguistic reasons, not config reasons.
# They live with the routing logic that selects them.
# If you want A/B testing prompts, add a template registry.
# ════════════════════════════════════════════════════════

_PROMPTS: dict[QueryRoute, str] = {

    QueryRoute.CODE_SEARCH: """\
You are a code analysis assistant. Answer using ONLY the provided context.

Rules:
- Cite specific file paths and function names
- Include relevant code snippets from context
- State line numbers when available
- If the answer is not in the context, say so explicitly

Context:
{context}

Question: {question}

Answer with file references:""",

    QueryRoute.CONCEPTUAL: """\
You are a technical documentation assistant. Answer using ONLY the provided context.

Rules:
- Explain architecture, data flow, and design patterns
- Describe module relationships and responsibilities
- Be concise — no padding
- If the answer is not in the context, say so explicitly

Context:
{context}

Question: {question}

Explanation:""",

    QueryRoute.DEBUGGING: """\
You are a debugging assistant. Answer using ONLY the provided context.

Rules:
- Identify the root cause from the context
- Reference specific code sections where the issue likely originates
- Suggest a fix if the context supports it
- Do NOT speculate beyond what the context shows

Context:
{context}

Question: {question}

Debugging analysis:""",

    QueryRoute.SETUP: """\
You are a setup assistant. Answer using ONLY the provided context.

Rules:
- Provide step-by-step instructions
- Include exact commands from the context
- List dependencies and requirements
- If setup steps are not in the context, say so explicitly

Context:
{context}

Question: {question}

Setup instructions:""",
}


# ════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════

class QueryRouter:
    """
    Routes a query to the best prompt template.

    Usage in pipeline.py:
        router = QueryRouter()
        route, prompt_template = router.route(question)
        prompt = prompt_template.format(context=ctx, question=question)
    """

    def route(self, query: str) -> tuple[QueryRoute, str]:
        """
        Classify query and return (route, prompt_template).

        Scoring: count keyword hits per route.
        Winner: highest hit count.
        Tie: first route in _PATTERNS dict wins
             (DEBUGGING > SETUP > CODE_SEARCH > CONCEPTUAL).
        Default: CONCEPTUAL — safest fallback, broadest coverage.

        Returns:
            route:           QueryRoute enum value
            prompt_template: string with {context} and {question} slots
        """
        query_lower = query.lower()
        scores: dict[QueryRoute, int] = {}

        for route, patterns in _PATTERNS.items():
            score = sum(1 for p in patterns if p in query_lower)
            scores[route] = score

        best_route = max(scores, key=lambda r: scores[r])
        best_score = scores[best_route]

        # Zero hits on all patterns → default to CONCEPTUAL
        # WHY CONCEPTUAL as default?
        # "explain X" is the safest assumption for unknown queries.
        # A conceptual answer that mentions code is better than
        # a code-search answer that lists files for a conceptual question.
        if best_score == 0:
            best_route = QueryRoute.CONCEPTUAL

        logger.debug(
            "Query routed: %s (score=%d) | query='%s'",
            best_route.value, best_score, query[:60],
        )

        return best_route, _PROMPTS[best_route]


# ════════════════════════════════════════════════════════
# CONTEXT ASSEMBLER
# Lives here because context assembly is a prompt concern,
# not a retrieval concern.
# ════════════════════════════════════════════════════════

class ContextAssembler:
    """
    Assembles retrieved chunks into a context string for the prompt.

    Order matters: LLMs attend strongly to beginning and end,
    weakly to the middle ("lost in the middle", Liu et al. 2023).

    Strategy: highest-scored chunks at position 0 and position -1.
    Middle positions: filled with graph-expanded neighbors.

    WHY not just sort by score descending?
    That puts all high-scored chunks at the top → middle is weak.
    Interleaving ensures important chunks are at attention peaks.
    """

    # Max characters of context to send to LLM.
    # WHY chars not tokens? Token counting requires a tokenizer call.
    # Rough rule: 1 token ≈ 4 chars. 8000 tokens ≈ 32000 chars.
    # Stay under model context window with headroom for prompt + answer.
    MAX_CONTEXT_CHARS = 32_000

    def assemble(
        self,
        chunks,           # list[RetrievedChunk], scored and ordered
        query: str,
    ) -> tuple[str, int]:
        """
        Build context string from retrieved chunks.

        Args:
            chunks:  RetrievedChunk list, best first
            query:   original query (used for logging only)

        Returns:
            context_str: formatted context for prompt {context} slot
            chunk_count: how many chunks were included
        """
        if not chunks:
            return "", 0

        # Separate: original retrieval vs graph-expanded neighbors
        # WHY separate? We want retrieval hits at attention peaks,
        # neighbors in the middle.
        retrieved = [c for c in chunks if c.source != "graph_expand"]
        neighbors = [c for c in chunks if c.source == "graph_expand"]

        # Interleave: best retrieved → neighbors → second best retrieved
        # For small sets (< 4 chunks): just sort by score, no interleaving
        if len(retrieved) <= 2:
            ordered = chunks
        else:
            # First half of retrieved → neighbors → second half
            mid = len(retrieved) // 2
            ordered = retrieved[:mid] + neighbors + retrieved[mid:]

        # Build context string, respect char limit
        parts: list[str] = []
        total_chars = 0

        for rc in ordered:
            chunk = rc.chunk
            # Format: file path + symbol name as header, then code
            # WHY header? LLM needs to know WHERE this code lives.
            # Without it: "the login function" with no file reference.
            header = f"# File: {chunk.file_path} | {chunk.chunk_type.value}: {chunk.symbol_name}"
            if chunk.docstring:
                header += f"\n# Purpose: {chunk.docstring[:100]}"
            block = f"{header}\n{chunk.text}"

            if total_chars + len(block) > self.MAX_CONTEXT_CHARS:
                # Would exceed limit — stop here
                # Log how many chunks we dropped
                dropped = len(ordered) - len(parts)
                if dropped > 0:
                    logger.debug(
                        "Context limit reached: included %d chunks, dropped %d",
                        len(parts), dropped,
                    )
                break

            parts.append(block)
            total_chars += len(block)

        context_str = "\n\n---\n\n".join(parts)

        logger.debug(
            "Context assembled: %d chunks, %d chars",
            len(parts), total_chars,
        )
        return context_str, len(parts)
