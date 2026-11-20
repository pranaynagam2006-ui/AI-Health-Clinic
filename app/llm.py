from typing import Any

from app.config import Settings
from app.errors import LLMUnavailableError, MissingAPIKeyError


def create_llm(settings: Settings) -> Any:
    """Create the configured LangChain chat model; fail explicitly if unavailable."""
    try:
        if settings.llm_provider == "gemini":
            if not settings.google_api_key:
                raise MissingAPIKeyError(
                    "GOOGLE_API_KEY is required when LLM_PROVIDER=gemini. Set it in your environment or .env file."
                )
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=0,
                max_retries=0,
                request_timeout=20,
            )
        if settings.llm_provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
    except MissingAPIKeyError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(f"Could not initialize {settings.llm_provider}: {exc}") from exc
    raise LLMUnavailableError(f"Unsupported LLM_PROVIDER={settings.llm_provider!r}; use gemini or ollama.")


def structured_planner(llm: Any):
    try:
        from app.schemas import PlannerDecision

        return llm.with_structured_output(PlannerDecision)
    except Exception as exc:
        raise LLMUnavailableError(f"LLM backend does not support structured tool planning: {exc}") from exc
