from __future__ import annotations
import os
from langchain_core.language_models import BaseChatModel

_DEFAULTS = {
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-2.5-flash",
}

_API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def create_llm(provider: str = "groq", model: str | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the given provider.

    Args:
        provider: "groq" or "google"
        model:    model name; falls back to the provider default if omitted

    API keys are read from env:
        GROQ_API_KEY   — required when provider="groq"
        GOOGLE_API_KEY — required when provider="google"
    """
    provider = provider.lower()

    if provider not in _DEFAULTS:
        raise ValueError(
            f"Unknown provider={provider!r}. Supported: {list(_DEFAULTS)}"
        )

    key_env = _API_KEY_ENV[provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise EnvironmentError(f"{key_env} is not set in the environment.")

    resolved_model = model or _DEFAULTS[provider]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=resolved_model, temperature=0, groq_api_key=api_key)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=resolved_model, temperature=0, google_api_key=api_key)
