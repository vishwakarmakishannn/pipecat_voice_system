#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "moonshine-voice==0.1.0",
# ]
# ///
"""Run Moonshine STT against a WAV file or the system microphone.

Examples:
  uv run stt_test.py --download-only
  uv run stt_test.py --wav path/to/speech.wav --runs 3
  uv run stt_test.py
  uv run stt_test.py --providers CoreML,CPU
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from moonshine_voice import (
    MicTranscriber,
    ModelArch,
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
    load_wav_file,
)

ARCHITECTURES = {
    "tiny": ModelArch.TINY,
    "base": ModelArch.BASE,
    "tiny-streaming": ModelArch.TINY_STREAMING,
    "base-streaming": ModelArch.BASE_STREAMING,
    "small-streaming": ModelArch.SMALL_STREAMING,
    "medium-streaming": ModelArch.MEDIUM_STREAMING,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test local Moonshine speech-to-text and print latency metrics."
    )
    parser.add_argument(
        "--wav",
        type=Path,
        help="Transcribe this WAV file. Without this option, use the microphone.",
    )
    parser.add_argument(
        "--language", default="en", help="Model language (default: en)."
    )
    parser.add_argument(
        "--model-arch",
        choices=ARCHITECTURES,
        default="tiny-streaming",
        help="Moonshine architecture (default: tiny-streaming).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Use an already-downloaded model directory instead of the model cache.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Download/read model assets under this directory instead of the OS cache.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the selected model, print its path, and exit.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timed passes in WAV mode (default: 1).",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.25,
        help="Seconds between live partial transcript updates (default: 0.25).",
    )
    parser.add_argument("--device", type=int, help="Microphone device index.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--providers",
        help='Ordered ONNX providers, for example "CoreML,CPU" (default: CPU).',
    )
    parser.add_argument(
        "--coreml-cache-dir",
        type=Path,
        help="Persistent CoreML compilation cache (used with --providers CoreML,CPU).",
    )
    parser.add_argument(
        "--log-ort-runs",
        action="store_true",
        help="Ask Moonshine to log individual ONNX Runtime calls.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.update_interval <= 0:
        parser.error("--update-interval must be greater than zero")
    if args.model_path and args.cache_root:
        parser.error("--model-path and --cache-root cannot be used together")
    return args


def list_input_devices() -> None:
    import sounddevice as sd

    default_input = sd.default.device[0]
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) > 0:
            marker = "*" if index == default_input else " "
            print(
                f"{marker} #{index}: {device['name']} "
                f"({int(device['max_input_channels'])} in, "
                f"{int(device['default_samplerate'])} Hz native)"
            )


def resolve_model(args: argparse.Namespace) -> tuple[Path, ModelArch]:
    architecture = ARCHITECTURES[args.model_arch]
    if args.model_path:
        model_path = args.model_path.expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")
        return model_path, architecture

    cache_root = args.cache_root.expanduser().resolve() if args.cache_root else None
    started_at = time.perf_counter()
    model_path, resolved_architecture = get_model_for_language(
        args.language,
        architecture,
        cache_root=cache_root,
    )
    resolution_ms = (time.perf_counter() - started_at) * 1000.0
    print(f"Model lookup/download: {resolution_ms:.1f} ms")
    return Path(model_path), resolved_architecture


def transcriber_options(args: argparse.Namespace) -> dict[str, str]:
    options: dict[str, str] = {}
    if args.providers:
        options["ort_providers"] = args.providers
    if args.coreml_cache_dir:
        cache_dir = args.coreml_cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        options["coreml_cache_dir"] = str(cache_dir)
    if args.log_ort_runs:
        options["log_ort_runs"] = "true"
    return options


def transcript_text(transcript) -> str:
    lines = sorted(transcript.lines, key=lambda line: line.start_time)
    return " ".join(line.text.strip() for line in lines if line.text.strip())


def run_wav(args: argparse.Namespace, model_path: Path, model_arch: ModelArch) -> None:
    wav_path = args.wav.expanduser().resolve()
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file does not exist: {wav_path}")

    audio_data, sample_rate = load_wav_file(wav_path)
    audio_duration = len(audio_data) / sample_rate
    print(f"Audio: {wav_path}\nFormat: mono, {sample_rate} Hz, {audio_duration:.3f} s")

    load_started_at = time.perf_counter()
    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch,
        options=transcriber_options(args),
    )
    load_ms = (time.perf_counter() - load_started_at) * 1000.0
    print(f"Model load: {load_ms:.1f} ms")

    latencies: list[float] = []
    last_text = ""
    try:
        for run_number in range(1, args.runs + 1):
            started_at = time.perf_counter()
            transcript = transcriber.transcribe_without_streaming(
                audio_data, sample_rate=sample_rate
            )
            elapsed = time.perf_counter() - started_at
            latencies.append(elapsed * 1000.0)
            last_text = transcript_text(transcript)
            rtf = elapsed / audio_duration if audio_duration else float("inf")
            speed = audio_duration / elapsed if elapsed else float("inf")
            print(
                f"Run {run_number}: latency={elapsed * 1000.0:.1f} ms | "
                f"RTF={rtf:.3f} | speed={speed:.1f}x realtime"
            )
    finally:
        transcriber.close()

    print(f"Transcript: {last_text or '(no speech recognized)'}")
    if len(latencies) > 1:
        print(
            f"Latency summary: min={min(latencies):.1f} ms | "
            f"median={statistics.median(latencies):.1f} ms | "
            f"mean={statistics.fmean(latencies):.1f} ms"
        )


class LatencyListener(TranscriptEventListener):
    """Print live text and estimate delay relative to captured audio timestamps."""

    def __init__(self) -> None:
        self.session_started_at = 0.0
        self.first_partial_seen: set[int] = set()

    def start_session_clock(self) -> None:
        self.session_started_at = time.perf_counter()

    def _timings(self, line) -> tuple[float, float]:
        elapsed = time.perf_counter() - self.session_started_at
        from_onset_ms = max(0.0, elapsed - line.start_time) * 1000.0
        end_offset = line.start_time + line.duration
        behind_audio_ms = max(0.0, elapsed - end_offset) * 1000.0
        return from_onset_ms, behind_audio_ms

    def on_line_started(self, event) -> None:
        print(f"\n[speech] detected at {event.line.start_time:.2f} s", flush=True)

    def on_line_text_changed(self, event) -> None:
        line = event.line
        from_onset_ms, behind_audio_ms = self._timings(line)
        first_marker = (
            " first-partial" if line.line_id not in self.first_partial_seen else ""
        )
        self.first_partial_seen.add(line.line_id)
        print(
            f"[partial{first_marker}] {line.text}\n"
            f"  from-speech-onset={from_onset_ms:.1f} ms | "
            f"recognition-lag={behind_audio_ms:.1f} ms",
            flush=True,
        )

    def on_line_completed(self, event) -> None:
        line = event.line
        from_onset_ms, finalization_ms = self._timings(line)
        print(
            f"[final] {line.text or '(no speech recognized)'}\n"
            f"  utterance-e2e={from_onset_ms:.1f} ms | "
            f"finalization-latency={finalization_ms:.1f} ms",
            flush=True,
        )
        self.first_partial_seen.discard(line.line_id)

    def on_error(self, event) -> None:
        print(f"[Moonshine error] {event}", file=sys.stderr, flush=True)


def run_microphone(
    args: argparse.Namespace, model_path: Path, model_arch: ModelArch
) -> None:
    load_started_at = time.perf_counter()
    transcriber = MicTranscriber(
        model_path=str(model_path),
        model_arch=model_arch,
        update_interval=args.update_interval,
        device=args.device,
        samplerate=args.sample_rate,
        channels=1,
        blocksize=args.block_size,
        options=transcriber_options(args),
    )
    load_ms = (time.perf_counter() - load_started_at) * 1000.0
    print(f"Model load: {load_ms:.1f} ms")

    listener = LatencyListener()
    transcriber.add_listener(listener)
    print("Listening. Speak normally, pause to finalize, and press Ctrl+C to stop.")
    listener.start_session_clock()
    transcriber.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        transcriber.stop()
        transcriber.close()


def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return

    model_path, model_arch = resolve_model(args)
    print(f"Model: {model_path}")
    print(f"Architecture: {model_arch.name} ({model_arch.value})")
    if args.download_only:
        return

    if args.wav:
        run_wav(args, model_path, model_arch)
    else:
        run_microphone(args, model_path, model_arch)


if __name__ == "__main__":
    main()
