import asyncio

import pytest

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


@pytest.mark.anyio
async def test_query_embedding_cache_normalizes_and_returns_defensive_copies(monkeypatch):
    calls = []

    async def fake_embed_batch(values):
        calls.append(list(values))
        return [[1.0, 2.0]]

    embedding.reset_knowledge_embedding_cache_for_tests()
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(embedding, "KNOWLEDGE_QUERY_CACHE_SIZE", 8)
    monkeypatch.setattr(embedding, "KNOWLEDGE_QUERY_CACHE_TTL_SECONDS", 60.0)
    monkeypatch.setattr(embedding, "_embed_batch", fake_embed_batch)

    first = await embedding.embed_knowledge_text("  How   does Mswipe work? ")
    first[0] = 99.0
    second = await embedding.embed_knowledge_text("How does Mswipe work?")

    assert calls == [["How does Mswipe work?"]]
    assert second == [1.0, 2.0]
    assert all("How does Mswipe work?" not in key for key in embedding._query_cache)


@pytest.mark.anyio
async def test_concurrent_identical_queries_share_one_embedding_request(monkeypatch):
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_embed_batch(_values):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return [[3.0, 4.0]]

    embedding.reset_knowledge_embedding_cache_for_tests()
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(embedding, "KNOWLEDGE_QUERY_CACHE_SIZE", 8)
    monkeypatch.setattr(embedding, "KNOWLEDGE_QUERY_CACHE_TTL_SECONDS", 60.0)
    monkeypatch.setattr(embedding, "_embed_batch", fake_embed_batch)

    first = asyncio.create_task(embedding.embed_knowledge_text("same question"))
    await started.wait()
    second = asyncio.create_task(embedding.embed_knowledge_text("same question"))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == [[3.0, 4.0], [3.0, 4.0]]
    assert calls == 1


@pytest.mark.anyio
async def test_query_embedding_rejects_wrong_dimension(monkeypatch):
    async def fake_embed_batch(_values):
        return [[1.0]]

    embedding.reset_knowledge_embedding_cache_for_tests()
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(embedding, "_embed_batch", fake_embed_batch)

    with pytest.raises(RuntimeError, match="dimension"):
        await embedding.embed_knowledge_text("question")


@pytest.mark.anyio
async def test_warmup_failure_degrades_instead_of_blocking_voice_startup(monkeypatch):
    async def failed_embed(_values):
        raise RuntimeError("provider quota exhausted")

    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "_embed_batch", failed_embed)

    assert await embedding.warm_knowledge_embedding() is False


@pytest.mark.anyio
async def test_repeated_timeouts_open_circuit_and_skip_dense_deadline(monkeypatch):
    async def slow_embed(_values):
        await asyncio.sleep(0.02)
        return [[1.0, 2.0]]

    embedding.reset_knowledge_embedding_cache_for_tests()
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_QUERY_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_CIRCUIT_FAILURES", 2)
    monkeypatch.setattr(embedding, "_embed_batch", slow_embed)

    first = await embedding.query_knowledge_embedding("first")
    second = await embedding.query_knowledge_embedding("second")
    third = await embedding.query_knowledge_embedding("third")
    await asyncio.sleep(0.03)

    assert first.failure_class == "timeout"
    assert second.circuit_state == "open"
    assert third.failure_class == "circuit_open"
    assert third.duration_ms < 10
    assert embedding._query_inflight == {}


@pytest.mark.anyio
async def test_cancelled_waiter_does_not_leak_shared_embedding_task(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_embed(_values):
        started.set()
        await release.wait()
        return [[1.0, 2.0]]

    embedding.reset_knowledge_embedding_cache_for_tests()
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(embedding, "_embed_batch", controlled_embed)

    waiter = asyncio.create_task(embedding.embed_knowledge_text("cancel me"))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert embedding._query_inflight == {}
    assert len(embedding._query_cache) == 1


@pytest.mark.anyio
async def test_inflight_capacity_is_bounded(monkeypatch):
    release = asyncio.Event()

    async def controlled_embed(_values):
        await release.wait()
        return [[1.0, 2.0]]

    embedding.reset_knowledge_embedding_cache_for_tests()
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(embedding, "KNOWLEDGE_EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(embedding, "KNOWLEDGE_QUERY_INFLIGHT_MAX", 1)
    monkeypatch.setattr(embedding, "_embed_batch", controlled_embed)

    first = asyncio.create_task(embedding.embed_knowledge_text("first"))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="capacity"):
        await embedding.embed_knowledge_text("second")
    release.set()
    assert await first == [1.0, 2.0]
