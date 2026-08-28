import asyncio

from services.knowledge import embedding


def test_embedding_requests_are_bounded_batches(monkeypatch):
    batches = []

    async def fake_embed_batch(values):
        batches.append(list(values))
        return [[float(len(value)), 0.0] for value in values]

    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(embedding, "_embed_batch", fake_embed_batch)
    result = asyncio.run(embedding.embed_knowledge_texts(["one", "two", "three", ""]))
    assert batches == [["one", "two"], ["three"]]
    assert result == [[3.0, 0.0], [3.0, 0.0], [5.0, 0.0], None]
