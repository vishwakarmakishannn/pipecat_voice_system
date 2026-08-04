"""Process-wide Kokoro model, download, and inference runtime."""

import asyncio
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from loguru import logger

from .kokoro_config import KokoroConfig, load_kokoro_config


ModelFactory = Callable[[KokoroConfig], Any]
Downloader = Callable[[str, Path, float], None]


def _download_file(url: str, destination: Path, timeout_seconds: float) -> None:
    """Atomically download one official model asset into the shared cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    request = Request(url, headers={"User-Agent": "AuraVoiceKokoro/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty Kokoro asset from {url}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _create_model(config: KokoroConfig):
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Kokoro requires the "pipecat-ai[kokoro]" runtime dependencies'
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = config.intra_op_threads
    options.inter_op_num_threads = config.inter_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry(
        "session.intra_op.allow_spinning",
        "1" if config.allow_spinning else "0",
    )
    session = ort.InferenceSession(
        str(config.model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return Kokoro.from_session(session, str(config.voices_path))


class KokoroRuntime:
    """Own one optimized ONNX session shared by all voice-call services."""

    def __init__(
        self,
        config: KokoroConfig,
        *,
        model_factory: ModelFactory = _create_model,
        downloader: Downloader = _download_file,
    ):
        self.config = config
        self._model_factory = model_factory
        self._downloader = downloader
        self._model = None
        self._loaded = False
        self._warmed = False
        self._closed = False
        self._load_lock = threading.RLock()
        self._warm_lock = asyncio.Lock()
        # Serial inference avoids two call sessions oversubscribing the same
        # four Apple performance cores and producing unpredictable tail latency.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="kokoro-runtime",
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def warmed(self) -> bool:
        return self._warmed

    def _ensure_asset(
        self,
        path: Path,
        url: str,
        asset: str,
    ) -> None:
        if path.is_file():
            return
        logger.info(
            "voice_startup stage=kokoro_download asset={} precision={} path={}",
            asset,
            self.config.precision,
            path,
        )
        self._downloader(url, path, self.config.download_timeout_seconds)

    def _ensure_model_sync(self):
        with self._load_lock:
            if self._closed:
                raise RuntimeError("Kokoro runtime is closed")
            if self._model is not None:
                return self._model

            started = time.perf_counter()
            self._ensure_asset(
                self.config.model_path,
                self.config.model_url,
                "model",
            )
            self._ensure_asset(
                self.config.voices_path,
                self.config.voices_url,
                "voices",
            )
            self._model = self._model_factory(self.config)
            self._loaded = True
            logger.info(
                "voice_startup stage=model_loaded service=kokoro_tts "
                "precision={} intra_op_threads={} inter_op_threads={} "
                "allow_spinning={} duration_ms={}",
                self.config.precision,
                self.config.intra_op_threads,
                self.config.inter_op_threads,
                self.config.allow_spinning,
                round((time.perf_counter() - started) * 1000, 1),
            )
            return self._model

    def _warm_sync(self) -> None:
        model = self._ensure_model_sync()
        if self._warmed or not self.config.warmup_enabled:
            return
        started = time.perf_counter()
        samples, _ = model.create(
            "Ready.",
            voice=self.config.voice,
            lang=self.config.language_code,
            speed=1.0,
        )
        self._warmed = True
        logger.info(
            "voice_startup stage=kokoro_warmup precision={} duration_ms={} "
            "samples={}",
            self.config.precision,
            round((time.perf_counter() - started) * 1000, 1),
            len(samples),
        )

    async def warm(self) -> None:
        """Download, load, and optionally exercise the selected model once."""
        if self._closed:
            raise RuntimeError("Kokoro runtime is closed")
        async with self._warm_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._warm_sync)

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        language: str,
        speed: float,
    ) -> tuple[bytes, int]:
        model = self._ensure_model_sync()
        samples, sample_rate = model.create(
            text,
            voice=voice,
            lang=language,
            speed=speed,
        )
        audio = (samples * 32767).astype(np.int16).tobytes()
        return audio, sample_rate

    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        language: str,
        speed: float = 1.0,
    ) -> tuple[bytes, int]:
        """Run phonemization and inference away from the asyncio event loop."""
        if self._closed:
            raise RuntimeError("Kokoro runtime is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            voice,
            language,
            speed,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
        with self._load_lock:
            self._model = None
            self._loaded = False
            self._warmed = False


_runtime: KokoroRuntime | None = None
_runtime_lock = threading.Lock()


def get_kokoro_runtime(config: KokoroConfig | None = None) -> KokoroRuntime:
    """Return the process-wide runtime for the configured Kokoro variant."""
    global _runtime
    config = config or load_kokoro_config()
    with _runtime_lock:
        if _runtime is None or _runtime._closed:
            _runtime = KokoroRuntime(config)
        elif _runtime.config != config:
            raise RuntimeError(
                "Kokoro runtime is already initialized with different settings; "
                "restart the backend after changing KOKORO_* variables"
            )
        return _runtime


async def warm_kokoro_runtime() -> None:
    await get_kokoro_runtime().warm()


async def shutdown_kokoro_runtime() -> None:
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        await runtime.close()
