"""Central configuration. Everything env-driven, provider-agnostic."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Reasoning LLM (litellm model string — swap providers freely)
    # e.g. "deepseek/deepseek-v4-flash", "anthropic/claude-sonnet-5", "openai/gpt-4o"
    llm_model: str = "anthropic/claude-sonnet-5"
    deepseek_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Vision model — OCR fallback ONLY. Claude models have vision, so the
    # same provider/key can cover both roles.
    vision_model: str = "anthropic/claude-sonnet-5"
    gemini_api_key: str | None = None

    # Firecrawl web search/fetch
    firecrawl_api_key: str | None = None

    # openFDA (optional; works without a key at lower rate limits)
    openfda_api_key: str | None = None

    # OCR
    ocr_confidence_threshold: float = 0.75

    # Limits
    max_drugs_per_request: int = 15
    pair_concurrency: int = 5
    http_timeout: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
