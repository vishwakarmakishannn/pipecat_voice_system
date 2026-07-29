import threading
import time

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.utils.env import env_truthy
from pipecat.audio.vad.silero import SileroOnnxModel, SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer


class SharedModelSmartTurnAnalyzerV3(LocalSmartTurnAnalyzerV3):
    """Per-call turn state with one process-wide immutable ONNX session."""

    _shared_session = None
    _session_lock = threading.Lock()

    def __init__(self, **kwargs):
        BaseSmartTurn.__init__(self, **kwargs)
        self._log_data = env_truthy("PIPECAT_SMART_TURN_LOG_DATA", default=False)
        with self._session_lock:
            if self.__class__._shared_session is None:
                loader = LocalSmartTurnAnalyzerV3(**kwargs)
                self.__class__._shared_session = loader._session
                loader._executor.shutdown(wait=False, cancel_futures=True)
        self._session = self.__class__._shared_session


_warm_analyzer = None


def warm_smart_turn_model():
    global _warm_analyzer
    if _warm_analyzer is None:
        _warm_analyzer = SharedModelSmartTurnAnalyzerV3()
    return _warm_analyzer


class SharedModelSileroVADAnalyzer(SileroVADAnalyzer):
    """Per-call recurrent VAD state backed by one immutable ONNX session."""

    _shared_session = None
    _session_lock = threading.Lock()

    def __init__(self, **kwargs):
        VADAnalyzer.__init__(self, **kwargs)
        with self._session_lock:
            if self.__class__._shared_session is None:
                loader = SileroVADAnalyzer(**kwargs)
                self.__class__._shared_session = loader._model.session
        model = object.__new__(SileroOnnxModel)
        model.session = self.__class__._shared_session
        model.sample_rates = [8000, 16000]
        model.reset_states()
        self._model = model
        self._last_reset_time = time.time()


_warm_vad = None


def warm_silero_vad_model():
    global _warm_vad
    if _warm_vad is None:
        _warm_vad = SharedModelSileroVADAnalyzer()
    return _warm_vad
