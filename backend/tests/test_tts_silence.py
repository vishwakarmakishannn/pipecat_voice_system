import pytest

from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection

from core.processors import LeadingSilenceTrimmerProcessor


def _pcm(samples):
    return b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples)


@pytest.mark.anyio
async def test_leading_silence_is_dropped_with_preroll(monkeypatch):
    trimmer = LeadingSilenceTrimmerProcessor(enabled=True, threshold=100, preroll_ms=20)
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(trimmer, "push_frame", capture)
    await trimmer.process_frame(TTSStartedFrame(context_id="turn"), FrameDirection.DOWNSTREAM)
    await trimmer.process_frame(
        TTSAudioRawFrame(_pcm([0] * 100), 1000, 1, context_id="turn"),
        FrameDirection.DOWNSTREAM,
    )
    audible = TTSAudioRawFrame(_pcm(([0] * 10) + ([500] * 5)), 1000, 1, context_id="turn")
    await trimmer.process_frame(audible, FrameDirection.DOWNSTREAM)

    assert len(delivered) == 2
    assert delivered[1] is audible
    assert audible.audio == _pcm(([0] * 20) + ([500] * 5))


@pytest.mark.anyio
async def test_audio_after_first_audible_frame_passes_unchanged(monkeypatch):
    trimmer = LeadingSilenceTrimmerProcessor(enabled=True, threshold=100, preroll_ms=0)
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(trimmer, "push_frame", capture)
    await trimmer.process_frame(TTSStartedFrame(context_id="turn"), FrameDirection.DOWNSTREAM)
    first = TTSAudioRawFrame(_pcm([500]), 24000, 1, context_id="turn")
    following = TTSAudioRawFrame(_pcm([0, 0]), 24000, 1, context_id="turn")
    stopped = TTSStoppedFrame(context_id="turn")
    await trimmer.process_frame(first, FrameDirection.DOWNSTREAM)
    await trimmer.process_frame(following, FrameDirection.DOWNSTREAM)
    await trimmer.process_frame(stopped, FrameDirection.DOWNSTREAM)

    assert delivered == [delivered[0], first, following, stopped]
    assert isinstance(delivered[0], TTSStartedFrame)


@pytest.mark.anyio
async def test_silence_scan_uses_absolute_signed_pcm_threshold(monkeypatch):
    trimmer = LeadingSilenceTrimmerProcessor(enabled=True, threshold=100, preroll_ms=0)
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(trimmer, "push_frame", capture)
    frame = TTSAudioRawFrame(_pcm([0, -101, 500]), 24000, 1, context_id="turn")
    await trimmer.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert delivered == [frame]
    assert frame.audio == _pcm([-101, 500])
