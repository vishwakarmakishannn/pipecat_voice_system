import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer

from core.turn_analyzer import (
    SharedModelSileroVADAnalyzer,
    SharedModelSmartTurnAnalyzerV3,
)


def test_smart_turn_instances_share_only_model_session():
    first = SharedModelSmartTurnAnalyzerV3()
    second = SharedModelSmartTurnAnalyzerV3()
    try:
        assert first is not second
        assert first._session is second._session
        assert first._audio_buffer is not second._audio_buffer
        assert first._executor is not second._executor
    finally:
        first._executor.shutdown(wait=False, cancel_futures=True)
        second._executor.shutdown(wait=False, cancel_futures=True)


def test_silero_instances_share_only_model_session():
    first = SharedModelSileroVADAnalyzer(sample_rate=16000)
    second = SharedModelSileroVADAnalyzer(sample_rate=16000)
    assert first is not second
    assert first._model is not second._model
    assert first._model.session is second._model.session
    assert first._model._state is not second._model._state


def test_silero_diagnostics_are_bounded_and_transcript_free():
    analyzer = SharedModelSileroVADAnalyzer(sample_rate=16000)
    with analyzer._diagnostics_lock:
        analyzer._diagnostic_samples.extend(
            [(0.9, 0.7), (0.4, 0.3), (0.1, 0.05)]
        )

    diagnostics = analyzer.diagnostics()

    assert diagnostics == {
        "window_frames": 3.0,
        "confidence_min": 0.1,
        "confidence_max": 0.9,
        "confidence_avg": 0.4667,
        "confidence_at_stop": 0.1,
        "volume_min": 0.05,
        "volume_max": 0.7,
        "volume_avg": 0.35,
        "volume_at_stop": 0.05,
    }


def test_silero_diagnostics_accept_one_element_array_without_changing_vad_value(
    monkeypatch,
):
    analyzer = object.__new__(SharedModelSileroVADAnalyzer)
    analyzer._diagnostics_lock = threading.Lock()
    analyzer._last_confidence = 0.0
    analyzer._diagnostic_samples = deque(maxlen=64)
    confidence = np.array([0.8125], dtype=np.float32)

    monkeypatch.setattr(
        SileroVADAnalyzer,
        "voice_confidence",
        lambda _self, _buffer: confidence,
    )

    returned = analyzer.voice_confidence(b"audio")

    assert returned is confidence
    assert analyzer._last_confidence == pytest.approx(0.8125)


def test_silero_diagnostics_skip_non_scalar_values_without_breaking_vad(
    monkeypatch,
):
    analyzer = object.__new__(SharedModelSileroVADAnalyzer)
    analyzer._diagnostics_lock = threading.Lock()
    analyzer._last_confidence = 0.25
    analyzer._diagnostic_samples = deque(maxlen=64)
    confidence = np.array([0.4, 0.6], dtype=np.float32)

    monkeypatch.setattr(
        SileroVADAnalyzer,
        "voice_confidence",
        lambda _self, _buffer: confidence,
    )

    returned = analyzer.voice_confidence(b"audio")

    assert returned is confidence
    assert analyzer._last_confidence == 0.25


@pytest.mark.anyio
async def test_smart_turn_logs_model_probability(monkeypatch):
    analyzer = object.__new__(SharedModelSmartTurnAnalyzerV3)
    analyzer._pending_completion_reason = None
    metrics = SimpleNamespace(e2e_processing_time_ms=12.34, probability=0.87654)

    async def analyze_end_of_turn(_self):
        return EndOfTurnState.COMPLETE, metrics

    monkeypatch.setattr(
        LocalSmartTurnAnalyzerV3,
        "analyze_end_of_turn",
        analyze_end_of_turn,
    )
    messages = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        state, returned_metrics = await analyzer.analyze_end_of_turn()
    finally:
        logger.remove(sink_id)

    assert state == EndOfTurnState.COMPLETE
    assert returned_metrics is metrics
    assert any(
        "state=COMPLETE reason=model_prediction probability=0.8765" in message
        for message in messages
    )


@pytest.mark.anyio
async def test_smart_turn_logs_forced_silence_timeout_as_effective_completion(monkeypatch):
    analyzer = object.__new__(SharedModelSmartTurnAnalyzerV3)
    analyzer._pending_completion_reason = None

    monkeypatch.setattr(
        BaseSmartTurn,
        "append_audio",
        lambda _self, _buffer, _is_speech: EndOfTurnState.COMPLETE,
    )

    async def analyze_end_of_turn(_self):
        return EndOfTurnState.INCOMPLETE, None

    monkeypatch.setattr(
        LocalSmartTurnAnalyzerV3,
        "analyze_end_of_turn",
        analyze_end_of_turn,
    )
    messages = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        assert analyzer.append_audio(b"", False) == EndOfTurnState.COMPLETE
        state, metrics = await analyzer.analyze_end_of_turn()
    finally:
        logger.remove(sink_id)

    assert state == EndOfTurnState.INCOMPLETE
    assert metrics is None
    assert analyzer._pending_completion_reason is None
    assert any(
        "state=COMPLETE reason=forced_silence_timeout probability=None" in message
        for message in messages
    )
