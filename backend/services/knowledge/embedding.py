"""Provider-versioned embeddings for the Mswipe corpus."""

import os

from core.knowledge_config import (
    KNOWLEDGE_EMBEDDING_BATCH_SIZE,
    KNOWLEDGE_EMBEDDING_DIMENSION,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_EMBEDDING_PROVIDER,
)

_google_client = None
_openai_client = None


def embedding_identity() -> tuple[str, str, int]:
    return (
        KNOWLEDGE_EMBEDDING_PROVIDER,
        KNOWLEDGE_EMBEDDING_MODEL,
        KNOWLEDGE_EMBEDDING_DIMENSION,
    )


async def embed_knowledge_texts(values: list[str]) -> list[list[float] | None]:
    """Embed a batch without silently crossing provider/model vector spaces."""
    normalized = [" ".join((value or "").split()) for value in values]
    if KNOWLEDGE_EMBEDDING_PROVIDER == "disabled":
        return [None] * len(normalized)
    nonempty = [(index, value) for index, value in enumerate(normalized) if value]
    output: list[list[float] | None] = [None] * len(values)
    if not nonempty:
        return output

    payload = [value for _, value in nonempty]
    vectors: list[list[float]] = []
    for start in range(0, len(payload), KNOWLEDGE_EMBEDDING_BATCH_SIZE):
        batch = payload[start : start + KNOWLEDGE_EMBEDDING_BATCH_SIZE]
        vectors.extend(await _embed_batch(batch))

    if len(vectors) != len(nonempty):
        raise RuntimeError("Embedding provider returned an incomplete batch")
    for (index, _), vector in zip(nonempty, vectors, strict=True):
        if len(vector) != KNOWLEDGE_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding dimension {len(vector)} does not match configured "
                f"dimension {KNOWLEDGE_EMBEDDING_DIMENSION}"
            )
        output[index] = list(vector)
    return output


async def _embed_batch(payload: list[str]) -> list[list[float]]:
    if KNOWLEDGE_EMBEDDING_PROVIDER == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for knowledge embeddings")
        global _google_client
        if _google_client is None:
            from google import genai

            _google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        from google.genai import types

        response = await _google_client.aio.models.embed_content(
            model=KNOWLEDGE_EMBEDDING_MODEL,
            contents=payload,
            config=types.EmbedContentConfig(
                output_dimensionality=KNOWLEDGE_EMBEDDING_DIMENSION
            ),
        )
        return [list(item.values) for item in response.embeddings]
    elif KNOWLEDGE_EMBEDDING_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for knowledge embeddings")
        global _openai_client
        if _openai_client is None:
            from openai import AsyncOpenAI

            _openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = await _openai_client.embeddings.create(
            input=payload,
            model=KNOWLEDGE_EMBEDDING_MODEL,
            dimensions=KNOWLEDGE_EMBEDDING_DIMENSION,
        )
        return [
            list(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]
    else:  # guarded by configuration validation
        raise RuntimeError("Unsupported knowledge embedding provider")



async def embed_knowledge_text(value: str) -> list[float] | None:
    return (await embed_knowledge_texts([value]))[0]
