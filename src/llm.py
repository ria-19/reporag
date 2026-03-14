"""
LLM provider abstraction.

pipeline.py calls generate(prompt) — never knows which LLM runs.
Provider selected from settings.llm_provider at startup.

WHY not in pipeline.py?
Pipeline's job: orchestrate steps.
LLM provider switching: configuration concern.
Mixing them violates single responsibility.
If Ollama changes its API: touch llm.py only.

WHY not in config.py?
config.py holds values (strings, ints, bools).
This holds behavior (a class with methods).
Values and behavior are different things.
"""

from __future__ import annotations
from typing import Protocol
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class LLMPort(Protocol):
    def generate(self, prompt: str) -> tuple[str, int]:
        """
        Generate a response from a prompt.
        Returns (answer_text, tokens_used).
        WHY return tokens? Observability — pipeline.py
        adds this to QueryMetrics.
        """
        ...


class OllamaLLM:
    """Local Ollama inference. No API key, no network call."""

    def __init__(self):
        import ollama
        self._client = ollama.Client()
        self._model  = settings.llm_model
        
        # ── Self-Healing: Physical Download ──────────
        self._ensure_model_ready()
        logger.info("OllamaLLM ready: %s", self._model)
    
    def _ensure_model_ready(self):
        """Checks if model exists; pulls it if not."""
        try:
            local_models_response = self._client.list()
            exists = any(
                m.model.startswith(self._model) 
                for m in local_models_response.models
            )
            if not exists:
                logger.info("Model '%s' not found. Pulling now...", self._model)
                self._client.pull(model=self._model)
                logger.info("Successfully pulled '%s'", self._model)
            else:
                logger.debug("Model '%s' already exists locally.", self._model)
                
        except Exception as e:
            logger.error(f"Ollama pre-flight check failed: {e}")
            raise RuntimeError(f"Ollama model {self._model} unavailable.") from e
    
    def generate(self, prompt: str) -> tuple[str, int]:
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
        )
        text   = response["response"]
        tokens = response.get("eval_count", 0)
        return text, tokens


class GeminiLLM:
    """Google Gemini via API key."""

    def __init__(self):
        import google.generativeai as genai
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for Gemini provider")
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)
        logger.info("GeminiLLM ready: %s", settings.gemini_model)
        
        # ── Self-Healing: Connectivity Check ─────────
        self._verify_connection()
        logger.info("GeminiLLM ready: %s", self._model_name)

    def _verify_connection(self):
        """Simple ping to ensure the API key works and service is up."""
        try:
            # We send a tiny prompt to verify the key/model status
            self._model.generate_content("ping", generation_config={"max_output_tokens": 1})
        except Exception as e:
            logger.error("Gemini pre-flight check failed. Check your API key or Quota: %s", e)
            raise RuntimeError(f"Gemini API unreachable for model {self._model_name}") from e

    def generate(self, prompt: str) -> tuple[str, int]:
        response = self._model.generate_content(prompt)
        text     = response.text
        tokens   = response.usage_metadata.total_token_count
        return text, tokens


def build_llm() -> LLMPort:
    """
    Factory function — reads config, returns correct implementation.
    Called once at startup in pipeline.py constructor.
    WHY factory not class? One function, one job.
    """
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaLLM()
    if provider == "gemini":
        return GeminiLLM()
    raise ValueError(
        f"Unknown llm_provider: {provider}. "
        f"Set LLM_PROVIDER=ollama or LLM_PROVIDER=gemini in .env"
    )