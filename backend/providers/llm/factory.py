import os


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "google").strip().lower()

    if provider == "google":
        from .google_llm import get_google_llm
        return get_google_llm()
    if provider == "groq":
        from .groq_llm import get_groq_llm
        return get_groq_llm()
    if provider == "openai":
        from .openai_llm import get_openai_llm
        return get_openai_llm()
    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. Expected google, groq, or openai."
    )
