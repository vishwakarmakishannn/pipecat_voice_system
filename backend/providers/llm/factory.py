import os


def _selected_provider() -> str:
    return os.getenv("LLM_PROVIDER", "google").strip().lower()


def get_llm():
    provider = _selected_provider()

    if provider == "google":
        from .google_llm import get_google_llm
        return get_google_llm()
    if provider == "groq":
        from .groq_llm import get_groq_llm
        return get_groq_llm()
    if provider == "openai":
        from .openai_llm import get_openai_llm
        return get_openai_llm()
    if provider == "local":
        from providers.local.llm.local_llm import get_local_llm
        return get_local_llm()
    raise ValueError(
        "Unsupported LLM provider: "
        f"{provider!r}. Expected google, groq, openai, or local."
    )


async def warm_llm_provider() -> None:
    """Warm startup-only resources for the selected LLM provider."""
    if _selected_provider() == "local":
        from providers.local.llm.runtime import warm_local_llm_runtime
        await warm_local_llm_runtime()


async def shutdown_llm_provider() -> None:
    """Release process-wide resources for the selected LLM provider."""
    if _selected_provider() == "local":
        from providers.local.llm.runtime import shutdown_local_llm_runtime
        await shutdown_local_llm_runtime()
