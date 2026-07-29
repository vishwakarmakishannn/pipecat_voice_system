import pytest

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.deepgram.tts import DeepgramTTSService

from providers.tts.deepgram_tts import InstrumentedDeepgramTTSService


@pytest.mark.anyio
async def test_deepgram_tts_records_request_to_provider_audio(monkeypatch):
    service = InstrumentedDeepgramTTSService(api_key="test")
    logs = []

    async def parent_run_tts(_self, _text, _context_id):
        yield None

    async def parent_append(_self, _context_id, _frame):
        return None

    monkeypatch.setattr(DeepgramTTSService, "run_tts", parent_run_tts)
    monkeypatch.setattr(DeepgramTTSService, "append_to_audio_context", parent_append)
    monkeypatch.setattr("providers.tts.deepgram_tts.logger.info", lambda *args: logs.append(args))

    assert [frame async for frame in service.run_tts("hello", "turn-1")] == [None]
    await service.append_to_audio_context(
        "turn-1", TTSAudioRawFrame(b"\x00\x00", 24000, 1, context_id="turn-1")
    )

    assert logs[0][0].startswith("voice_tts stage=request_started")
    assert isinstance(logs[1][-1], float)
    assert logs[1][-1] >= 0

    await service.append_to_audio_context("turn-1", TTSStoppedFrame(context_id="turn-1"))
    assert service._request_started_at == {}
    assert service._first_audio_seen == set()
