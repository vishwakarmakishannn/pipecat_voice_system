import os


def _selected_provider() -> str:
    return os.getenv("LLM_PROVIDER", "google").strip().lower()


def get_llm(system_instruction: str | None = None):
    provider = _selected_provider()
    instruction_kwargs = (
        {"system_instruction": system_instruction}
        if system_instruction is not None
        else {}
    )

    if provider == "google":
        from .google_llm import get_google_llm
        return get_google_llm(**instruction_kwargs)
    if provider == "groq":
        from .groq_llm import get_groq_llm
        return get_groq_llm(**instruction_kwargs)
    if provider == "openai":
        from .openai_llm import get_openai_llm
        return get_openai_llm(**instruction_kwargs)
    if provider == "local":
        from providers.local.llm.local_llm import get_local_llm
        return get_local_llm(**instruction_kwargs)
    raise ValueError(
        "Unsupported LLM provider: "
        f"{provider!r}. Expected google, groq, openai, or local."
    )


async def warm_llm_provider() -> None:
    """Warm startup-only resources for the selected LLM provider."""
    provider = _selected_provider()
    if provider == "local":
        from providers.local.llm.runtime import warm_local_llm_runtime
        await warm_local_llm_runtime()
    elif provider == "groq":
        from providers.llm.groq_runtime import warm_groq_runtime
        await warm_groq_runtime()


async def shutdown_llm_provider() -> None:
    """Release process-wide resources for the selected LLM provider."""
    provider = _selected_provider()
    if provider == "local":
        from providers.local.llm.runtime import shutdown_local_llm_runtime
        await shutdown_local_llm_runtime()
    elif provider == "groq":
        from providers.llm.groq_runtime import shutdown_groq_runtime
        await shutdown_groq_runtime()
