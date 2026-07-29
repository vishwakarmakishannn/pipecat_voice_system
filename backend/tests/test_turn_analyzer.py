from core.turn_analyzer import SharedModelSileroVADAnalyzer, SharedModelSmartTurnAnalyzerV3


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
