import threading
import time
from collections import deque

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
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
        self._pending_completion_reason: str | None = None
        self._log_data = env_truthy("PIPECAT_SMART_TURN_LOG_DATA", default=False)
        with self._session_lock:
            if self.__class__._shared_session is None:
                loader = LocalSmartTurnAnalyzerV3(**kwargs)
                self.__class__._shared_session = loader._session
                loader._executor.shutdown(wait=False, cancel_futures=True)
        self._session = self.__class__._shared_session

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Remember when silence, rather than the model, forced completion."""
        state = super().append_audio(buffer, is_speech)
        if state == EndOfTurnState.COMPLETE:
            self._pending_completion_reason = "forced_silence_timeout"
        return state

    async def analyze_end_of_turn(self):
        """Expose analyzer latency, confidence, and the effective stop reason."""
        started_at = time.perf_counter()
        state, metrics = await super().analyze_end_of_turn()
        forced_reason = self._pending_completion_reason
        self._pending_completion_reason = None
        effective_state = EndOfTurnState.COMPLETE if forced_reason else state
        reason = forced_reason or "model_prediction"
        logger.info(
            "voice_endpointing stage=smart_turn_analysis state={} "
            "reason={} probability={} duration_ms={} model_ms={}",
            getattr(effective_state, "name", str(effective_state)),
            reason,
            (
                round(metrics.probability, 4)
                if metrics is not None
                else None
            ),
            round((time.perf_counter() - started_at) * 1000, 1),
            (
                round(metrics.e2e_processing_time_ms, 1)
                if metrics is not None
                else None
            ),
        )
        return state, metrics


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
        self._diagnostics_lock = threading.Lock()
        self._last_confidence = 0.0
        self._diagnostic_samples: deque[tuple[float, float]] = deque(
            maxlen=64
        )
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

    @staticmethod
    def _diagnostic_scalar(value) -> float | None:
        """Best-effort scalar conversion that cannot break live VAD.

        Pipecat's Silero wrapper currently returns a one-element NumPy array
        even though ``voice_confidence`` is annotated as returning ``float``.
        Diagnostics must tolerate either representation and must never change
        the value flowing through the VAD implementation.
        """
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def voice_confidence(self, buffer):
        confidence = super().voice_confidence(buffer)
        diagnostic_confidence = self._diagnostic_scalar(confidence)
        if diagnostic_confidence is not None:
            with self._diagnostics_lock:
                self._last_confidence = diagnostic_confidence
        return confidence

    def _get_smoothed_volume(self, audio: bytes):
        volume = super()._get_smoothed_volume(audio)
        diagnostic_volume = self._diagnostic_scalar(volume)
        if diagnostic_volume is not None:
            with self._diagnostics_lock:
                self._diagnostic_samples.append(
                    (self._last_confidence, diagnostic_volume)
                )
        return volume

    def diagnostics(self) -> dict[str, float]:
        """Return a bounded, transcript-free view of the recent VAD window."""
        with self._diagnostics_lock:
            samples = list(self._diagnostic_samples)
        if not samples:
            return {}
        confidences = [sample[0] for sample in samples]
        volumes = [sample[1] for sample in samples]
        return {
            "window_frames": float(len(samples)),
            "confidence_min": round(min(confidences), 4),
            "confidence_max": round(max(confidences), 4),
            "confidence_avg": round(sum(confidences) / len(confidences), 4),
            "confidence_at_stop": round(confidences[-1], 4),
            "volume_min": round(min(volumes), 4),
            "volume_max": round(max(volumes), 4),
            "volume_avg": round(sum(volumes) / len(volumes), 4),
            "volume_at_stop": round(volumes[-1], 4),
        }


_warm_vad = None


def warm_silero_vad_model():
    global _warm_vad
    if _warm_vad is None:
        _warm_vad = SharedModelSileroVADAnalyzer()
    return _warm_vad
