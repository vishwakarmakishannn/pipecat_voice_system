# Pipecat RAG Backend

This is the backend for the Pipecat RAG voice agent.

## Local llama.cpp LLM

The backend can use a separately managed, OpenAI-compatible `llama-server`.
The tested development configuration is Qwen3-4B-Instruct-2507 Q4_K_M with
two parallel slots:

```bash
./llama.cpp/build/bin/llama-server \
  -m /absolute/path/to/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --alias qwen3-4b-local \
  --host 127.0.0.1 --port 8080 \
  -c 16384 -np 2 -ngl 99 -fa on --jinja
```

Select it in `.env`:

```dotenv
LLM_PROVIDER=local
VOICE_MAX_CONCURRENT_SESSIONS=2
LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_LLM_MODEL=qwen3-4b-local
LOCAL_LLM_API_KEY=local-no-key
LOCAL_LLM_TEMPERATURE=0.7
LOCAL_LLM_TOP_P=0.8
LOCAL_LLM_TOP_K=20
LOCAL_LLM_MIN_P=0.0
LOCAL_LLM_PRESENCE_PENALTY=0.0
LOCAL_LLM_MAX_TOKENS=512
LOCAL_LLM_WARMUP_TIMEOUT_SECONDS=30
LOCAL_LLM_MAX_CONCURRENT_SESSIONS=2
```

The backend does not launch or stop `llama-server`. It verifies the configured
model and runs a short warmup completion during application startup. Startup
fails if the server or model is unavailable; local requests never fall back to
a cloud LLM. The process shares one HTTP client across at most two admitted
voice sessions.

Set `LLM_PROVIDER=groq`, `google`, or `openai` to restore a cloud provider.
When voice uses the local provider, keep deferred memory classification off the
voice model with `MEMORY_LLM_PROVIDER=groq` (or `google`/`openai`). Set it to
`local` only when sharing llama.cpp capacity is intentional.
Those providers retain their existing credential and model settings.

Set the assistant's default IANA timezone independently of the selected LLM:

```dotenv
VOICE_TIMEZONE=Asia/Kolkata
```

The backend injects the current date and timezone once into each session's
durable system instruction. Exact clock
time, timezone conversion, and deadline questions use the always-available
`get_current_datetime` tool so changing seconds do not invalidate prompt-cache
prefixes. `tavily_search` is also advertised on every ordinary turn; both
read-only tools use automatic semantic selection rather than phrase matching.
The write-capable complaint tool remains scoped to its confirmation workflow.

## Live conversation context

Live sessions use Pipecat's automatic context summarization instead of hard
dropping older turns. Stable authenticated facts and the canonical database
summary are placed in the session system instruction; only recent dialogue is
seeded into mutable context. A dedicated Groq model summarizes older clean
user/assistant turns, while raw tool results and query-scoped RAG/memory blocks
are excluded. Successfully applied summaries update the same canonical
`Conversation.summary` field used when a session is reopened.

```dotenv
VOICE_CONTEXT_SUMMARIZATION_ENABLED=true
VOICE_CONTEXT_SUMMARY_MAX_TOKENS=3000
VOICE_CONTEXT_SUMMARY_MAX_MESSAGES=20
VOICE_CONTEXT_SUMMARY_TARGET_TOKENS=900
VOICE_CONTEXT_SUMMARY_KEEP_MESSAGES=8
VOICE_CONTEXT_SUMMARY_TIMEOUT_SECONDS=6
VOICE_CONTEXT_SUMMARY_RETRY_COOLDOWN_SECONDS=30
VOICE_CONTEXT_EMERGENCY_MAX_MESSAGES=40
VOICE_CONTEXT_EMERGENCY_MAX_CHARS=24000
GROQ_CONTEXT_SUMMARY_MODEL=llama-3.1-8b-instant
```

When summarization is enabled, `GROQ_API_KEY` is required even if the live
voice model is Google, OpenAI, or local. The summarizer never falls back to the
latency-sensitive voice model. `VOICE_CONTEXT_MAX_MESSAGES=12` and
`VOICE_CONTEXT_MAX_CHARS=6000` remain the legacy bounds used only when the
feature flag is disabled. The larger emergency bounds protect a live session
if Groq is temporarily unavailable; failed attempts observe a retry cooldown.

## Speech-to-text providers

The backend supports four switchable STT providers:

```dotenv
STT_PROVIDER=deepgram    # Cloud streaming
STT_PROVIDER=whisper     # Local pywhispercpp
STT_PROVIDER=mlxwhisper  # Local Apple MLX
STT_PROVIDER=moonshine   # Local Moonshine true streaming
```

For the highest-accuracy English Moonshine v2 streaming model:

```dotenv
STT_PROVIDER=moonshine
AUDIO_INPUT_SAMPLE_RATE=16000
MOONSHINE_MODEL=medium-streaming
MOONSHINE_LANGUAGE=en
MOONSHINE_UPDATE_INTERVAL_SECONDS=0.25
MOONSHINE_TTFS_P99_SECONDS=0.75
# MOONSHINE_MODEL_DIR=/absolute/path/to/an/unpacked/model
# MOONSHINE_VOICE_CACHE=/absolute/path/to/a/model/cache
```

`medium-streaming` is the English-only 245M-parameter architecture. The model
is downloaded and loaded during backend startup, then shared across voice
sessions; every session receives an independent streaming state. Interim text
is emitted during speech and Moonshine's native streaming endpointing completes
each line. The stream intentionally remains continuous across Pipecat VAD
events so brief pauses and trailing audio are not split into false turns.
Docker Compose persists downloads in the `moonshine-cache` volume.
`MOONSHINE_TTFS_P99_SECONDS` is the measured P99
speech-end-to-final-transcript budget used only as Pipecat's missing-transcript
safety timeout; tune it from production telemetry if the default is too low.

Moonshine removes the need to wait until an entire utterance has been buffered
before recognition begins. That normally reduces speech-end-to-final-text
latency versus the local Whisper providers, but total turn latency still
depends on VAD end detection, hardware throughput, and LLM/TTS latency. Measure
the existing latency telemetry on the deployment hardware before choosing a
final update interval or session limit.

### Local Whisper

For local `whisper.cpp`:

```dotenv
STT_PROVIDER=whisper
AUDIO_INPUT_SAMPLE_RATE=16000
WHISPER_MODEL=small
WHISPER_LANGUAGE=auto
WHISPER_THREADS=4
# WHISPER_MODELS_DIR=/absolute/path/to/a/model/cache
```

Run `uv sync` after dependency changes. The selected model is downloaded into
pywhispercpp's platform cache on first startup unless `WHISPER_MODELS_DIR` is
set.

For Apple Silicon MLX Whisper:

```dotenv
STT_PROVIDER=mlxwhisper
AUDIO_INPUT_SAMPLE_RATE=16000
MLX_WHISPER_MODEL=mlx-community/whisper-small-mlx
MLX_WHISPER_LANGUAGE=auto
MLX_WHISPER_TEMPERATURE=0
MLX_WHISPER_NO_SPEECH_THRESHOLD=0.6
```

The Whisper providers download their selected model on first use and warm it
before the application reports ready.

## Local Kokoro TTS

Kokoro runs through one process-wide ONNX session. Each call still gets an
independent Pipecat service for interruption and metrics state, but calls no
longer construct or warm their own model:

```dotenv
TTS_PROVIDER=kokoro
KOKORO_VOICE_ID=af_heart
KOKORO_LANGUAGE=en-US
KOKORO_MODEL_PRECISION=fp16
AUDIO_OUTPUT_SAMPLE_RATE=24000
```

Supported language values are `en`, `en-US`, `en-GB`, `es`, `fr`, `hi`, `it`,
`ja`, `pt`, and `zh`. `KOKORO_MODEL_PRECISION` accepts `fp32`, `fp16`, or
`int8`. Kokoro uses low-latency phrase aggregation by default: the first
complete phrase is dispatched at about 12 characters, while later chunks
target 80 characters to keep playback buffered. The backend loads and warms
the selected model during process startup, so sessions do not pay ONNX's cold
start cost.

The latency controls can be tuned or disabled:

```dotenv
KOKORO_LOW_LATENCY=true
KOKORO_WARMUP_ENABLED=true
KOKORO_FIRST_CHUNK_CHARS=12
KOKORO_FIRST_CHUNK_MIN_WORDS=1
KOKORO_CHUNK_CHARS=80
KOKORO_MIN_CHUNK_WORDS=2
KOKORO_INTRA_OP_THREADS=4
KOKORO_INTER_OP_THREADS=1
KOKORO_ALLOW_SPINNING=false
```

`TTS_TEXT_AGGREGATION_MODE=token` remains an explicit escape hatch, but sends
raw LLM token fragments to Kokoro and can produce choppy speech.

Run `uv sync` after dependency changes. At startup, the backend downloads the
selected official model and `voices-v1.0.bin` into
`~/.cache/pipecat/kokoro-onnx`. Set `KOKORO_CACHE_DIR` to use another cache.
Docker Compose persists the default cache in the `kokoro-cache` volume. To
populate and warm it before accepting voice traffic:

```bash
docker compose run --rm -e TTS_PROVIDER=kokoro backend \
  python -c "import asyncio; from providers.tts.factory import warm_tts_provider; asyncio.run(warm_tts_provider())"
```

Production deployments can instead provide pre-downloaded files:

```dotenv
KOKORO_MODEL_PRECISION=fp16
KOKORO_MODEL_PATH=/models/kokoro/kokoro-v1.0.fp16.onnx
KOKORO_VOICES_PATH=/models/kokoro/voices-v1.0.bin
```

Both paths must be configured together and must point to existing files.
Kokoro uses no API key. Restart the backend after changing any `KOKORO_*`
model or runtime setting.

## RAG ingestion worker

Uploads and links are stored as durable `queued` database jobs. In local
development, run the ingestion worker in a second terminal:

```bash
uv run rag_worker.py
```

Install its parsing dependencies with `uv sync --extra rag-worker`. Docker
Compose starts the separate `rag-worker` service automatically; the voice API
image intentionally excludes Docling and browser-extraction dependencies.
