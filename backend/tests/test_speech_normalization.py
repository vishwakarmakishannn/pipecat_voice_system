from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from core.processors import SpeechNormalizationProcessor
from core.speech_normalization import normalize_speech_text


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("## Features\n- Fast setup\n- Safe payments", "Features. Fast setup. Safe payments"),
        ("Use **Mswipe** — it is reliable.", "Use Mswipe, it is reliable."),
        ("Open [support](https://example.com/help).", "Open support."),
        ("Visit https://example.com/help for details.", "Visit the provided link for details."),
        ("Success is 99%.", "Success is 99 percent."),
        ("One\u00a0two\u200b three", "One two three"),
    ],
)
def test_speech_normalization_golden_cases(raw, spoken):
    assert normalize_speech_text(raw) == spoken


@pytest.mark.anyio
async def test_spoken_frames_and_customer_transcript_are_identical(monkeypatch):
    frames = []
    processor = SpeechNormalizationProcessor()

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(
        LLMFullResponseStartFrame(),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(
        LLMTextFrame("## Answer\n- Use **Mswipe** — "),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(
        LLMTextFrame("it supports safe payments."),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(
        LLMFullResponseEndFrame(),
        FrameDirection.DOWNSTREAM,
    )

    spoken = "".join(
        frame.text for frame in frames if isinstance(frame, LLMTextFrame)
    )
    transcript_payloads = [
        frame.message["data"]["payload"]
        for frame in frames
        if isinstance(frame, OutputTransportMessageUrgentFrame)
        and frame.message["data"]["type"] == "assistant_transcript"
    ]
    final = [payload for payload in transcript_payloads if payload.get("final")]

    assert spoken == "Answer. Use Mswipe, it supports safe payments."
    assert final[0]["text"] == spoken


@pytest.mark.anyio
async def test_normalization_diagnostic_contains_hashes_not_raw_text(monkeypatch):
    frames = []
    events = []
    recorder = SimpleNamespace(record=lambda **payload: events.append(payload))
    processor = SpeechNormalizationProcessor(diagnostic_recorder=recorder)

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame("**private-looking text**"), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert events[0]["code"] == "tts.speech_normalized"
    assert len(events[0]["details"]["raw_sha256"]) == 64
    assert "private-looking" not in str(events[0])
