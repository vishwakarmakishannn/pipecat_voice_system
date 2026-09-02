# Pipecat Mswipe Voice Backend

This is the backend for the Pipecat Mswipe voice agent.

The independent, release-controlled Mswipe production knowledge system is
documented in [MSWIPE_KNOWLEDGE.md](MSWIPE_KNOWLEDGE.md).

## Local llama.cpp LLM

The backend can use a separately managed, OpenAI-compatible `llama-server`.
The tested development configuration is Qwen3-4B-Instruct-2507 Q4_K_M with
two parallel slots:

```bash
./llama.cpp/build/bin/llama-server \
  -m /absolute/path/to/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --alias qwen3-4b-local \
  --host 127.0.0.1 --port 8080 \
  -c 16384 -np 2 -ngl all -fa on --jinja \
  --cache-prompt --cache-ram 2048 --load-mode mmap+mlock \
  --log-disable --no-perf
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
model and warms the exact production system/tool prefix on every explicit
llama.cpp slot during application startup. Startup
fails if the server or model is unavailable; local requests never fall back to
a cloud LLM. The process shares one HTTP client across at most two admitted
voice sessions.

Set `LLM_PROVIDER=groq`, `google`, or `openai` to restore a cloud provider.
When voice uses the local provider, keep deferred memory classification off the
voice model with `MEMORY_LLM_PROVIDER=groq` (or `google`/`openai`). Set it to
`local` only when sharing llama.cpp capacity is intentional.
Those providers retain their existing credential and model settings.

### Groq live-voice transport and model gate

Groq uses one process-scoped HTTP connection pool with SDK retries disabled.
The voice service owns at most one observable transient retry and keeps both
attempts inside `VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS`; it never retries after
text or a native tool-call delta has been released. Relevant controls are:

```dotenv
GROQ_LIVE_MAX_ATTEMPTS=2
GROQ_LIVE_FIRST_ATTEMPT_TIMEOUT_SECONDS=2.5
GROQ_CONNECT_TIMEOUT_SECONDS=1.5
GROQ_POOL_TIMEOUT_SECONDS=0.5
GROQ_WRITE_TIMEOUT_SECONDS=5
GROQ_READ_TIMEOUT_SECONDS=30
VOICE_LLM_RETRY_RESERVE_SECONDS=0.35
```

The selected Groq model is a deployment decision, not an application
hardcoding. Before promoting a model, run the live planning-only capability
gate below. It checks timeless answers, ambiguity, Mswipe knowledge selection,
contextual Mswipe follow-ups, complaint selection,
current-fact limitations when web search is disabled, and simulated tool markup
without executing any tool:

```bash
GROQ_EVALUATION_MODEL=openai/gpt-oss-20b \
  uv run python -m scripts.evaluate_groq_voice_model
```

Treat a nonzero exit status as a failed production-promotion gate, then compare
passing candidates using recorded p50/p90 first-output latency from test calls.

Set the assistant's default IANA timezone independently of the selected LLM:

```dotenv
VOICE_TIMEZONE=Asia/Kolkata
```

The backend injects the current date and timezone once into each session's
durable system instruction. Exact clock
time, timezone conversion, and deadline questions use the always-available
`get_current_datetime` tool so changing seconds do not invalidate prompt-cache
prefixes. `search_mswipe_knowledge` is advertised when
`MSWIPE_KNOWLEDGE_ENABLED=true`. `tavily_search` is advertised only when
`WEB_SEARCH_ENABLED=true`; the Mswipe deployment keeps it disabled. Tool
descriptions and the complete conversation drive semantic selection rather
than keyword routing. While a complaint draft is active, only the complaint
state-machine tool is available and a validated confirmation is required.

```dotenv
MSWIPE_KNOWLEDGE_ENABLED=true
WEB_SEARCH_ENABLED=false
VOICE_TOOL_FILLER_ENABLED=true
VOICE_TOOL_FILLER_DELAY_MS=0
VOICE_TOOL_FILLER_TEXT="Let me look that up for you."
```

With the zero-millisecond delay, the configured filler is sent to the live
transcript and TTS before each tool's in-progress event. Set
`VOICE_TOOL_FILLER_ENABLED=false` to disable it without changing code.

## Voice 2.0 call lifecycle and context

Every voice start creates a new opaque UUID call. Calls may reconnect only to
their still-running transport during the 30-second grace window; a completed,
failed, cancelled, or abandoned call is immutable and cannot be reopened.

Each call receives a fresh Pipecat context. Only stable authenticated facts are
placed in its system instruction. Prior-call summaries and transcript excerpts
are retrieved later only when the active turn asks for relevant recall; an
older transcript is never used as the new call's mutable history. During an
active call, Pipecat automatic context summarization compacts clean user and
assistant turns. Raw tool results, diagnostics, recording metadata, and
query-scoped RAG/memory blocks are excluded. Successfully applied summaries
update `Call.summary` for later selective recall and analysis.

The breaking migration is guarded. It will not remove populated Voice 1.x
tables unless the intended database has first been reset with the literal
confirmation below. The command prints a credential-redacted database target;
verify it before proceeding:

```bash
uv run python -m scripts.reset_voice2_database --dry-run
uv run python -m scripts.reset_voice2_database \
  --confirm RESET_ALL_APPLICATION_DATA
uv run alembic upgrade head
```

The dashboard is split into `/playground`, `/calls`, `/calls/:callId`,
`/memories`. Historical call data is served by the owner-scoped
`/api/calls` list/detail/timeline/turn/client-event/recording/delete/restore
endpoints. There is no conversation-resume API.

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
GROQ_CONTEXT_SUMMARY_MODEL=openai/gpt-oss-20b
```

When summarization is enabled, `GROQ_API_KEY` is required even if the live
voice model is Google, OpenAI, or local. The summarizer never falls back to the
latency-sensitive voice model. `VOICE_CONTEXT_MAX_MESSAGES=12` and
`VOICE_CONTEXT_MAX_CHARS=6000` remain emergency bounds used only when the
feature flag is disabled. The larger emergency bounds protect a live session
if Groq is temporarily unavailable; failed attempts observe a retry cooldown.

## Recording, diagnostics, and retention

Every successfully created call is recorded as a combined 16 kHz mono stream.
Audio is written in bounded one-second chunks to a persistent PCM spool,
encoded as a 64 kbps MP3 with PyAV/libmp3lame, uploaded privately, and removed
from the spool after successful processing. Configure local paths with:

```dotenv
RECORDING_SPOOL_DIR=/var/lib/aura/recording-spool
RECORDING_STORAGE_DIR=/var/lib/aura/recordings
RECORDING_ACCESS_TTL_SECONDS=300
```

Docker Compose mounts both paths in the durable `call-artifacts` volume and
runs `call_maintenance_worker.py` for interrupted-recording recovery, stale
call abandonment, and expired-call purge. S3 deployments store private object
keys and return short-lived presigned URLs; local development uses a signed,
authenticated Range-capable media endpoint.

Calls are soft-deleted for 30 days before their transcript, turns, operations,
diagnostics, memory chunks, spool, metadata, and MP3 are permanently purged.
Customer-visible diagnostics are sanitized and intentionally exclude secrets,
complete prompts, provider bodies, stack traces, headers, and audio bytes.
Internal operational logs remain separate and should be shipped to the
deployment's normal log platform.

Recording is always enabled by the current product policy
(`always-on-v1`). Deployment operators are responsible for giving legally
required recording notice, obtaining consent, configuring encryption at rest
and in transit, and choosing retention appropriate to each jurisdiction.

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
MOONSHINE_UPDATE_INTERVAL_SECONDS=0.10
MOONSHINE_VAD_WINDOW_DURATION_SECONDS=0.25
MOONSHINE_FINALIZE_GRACE_SECONDS=0.35
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
If a native line does not complete within
`MOONSHINE_FINALIZE_GRACE_SECONDS`, a bounded safety flush recovers the turn;
this timeout is not added to normally completed turns.
Docker Compose persists downloads in the `moonshine-cache` volume.
`MOONSHINE_VAD_WINDOW_DURATION_SECONDS` reduces Moonshine's native VAD
averaging window from its 0.5-second default. Values below 0.25 seconds should
be validated against short utterances and clipped trailing words before use.
`MOONSHINE_TTFS_P99_SECONDS` is the measured P99
speech-end-to-final-transcript budget used only as Pipecat's missing-transcript
safety timeout; tune it from production telemetry if the default is too low.

Moonshine removes the need to wait until an entire utterance has been buffered
before recognition begins. That normally reduces speech-end-to-final-text
latency versus the local Whisper providers, but total turn latency still
depends on VAD end detection, hardware throughput, and LLM/TTS latency. Measure
the existing latency telemetry on the deployment hardware before choosing a
final update interval or session limit.

Moonshine forced finalization is submitted as soon as VAD stops and runs in
parallel with downstream SmartTurn analysis. Final transcript delivery remains
gated until the VAD-stop frame has reached the turn aggregator, so the reduced
latency does not reorder the frames. The `stt_finalization_ms` telemetry object
splits force-update queue/run time, SmartTurn-side VAD residence time, and final
callback delivery, alongside Moonshine's native final-transcription time.

The semantic SmartTurn gate remains the default. An explicitly faster
silence-timeout strategy is available for deployments that can tolerate a
higher chance of ending a turn during a natural pause:

```dotenv
TURN_STOP_STRATEGY=smart_turn       # safe default
# TURN_STOP_STRATEGY=speech_timeout # lower latency, more cutoff risk
SPEECH_TIMEOUT_SECS=0.2
VAD_STOP_SECS=0.15
SMART_TURN_STOP_SECS=0.7
```

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

## Mswipe knowledge worker

The only document-knowledge path is the release-controlled Mswipe subsystem.
Run its control-plane worker with:

```bash
uv sync --extra knowledge-worker
uv run knowledge_worker.py
```

Docker Compose starts `knowledge-worker` automatically. Source registration,
crawling, review, release publication, validation, search, and rollback are
documented in [MSWIPE_KNOWLEDGE.md](MSWIPE_KNOWLEDGE.md).

For the normal local workflow, run only PostgreSQL and the durable knowledge
worker in Docker, then keep the backend and frontend in terminals for readable
logs:

```bash
# Project root
docker compose up -d db knowledge-worker

# Terminal 1
cd backend
uv run main.py

# Terminal 2
cd frontend
npm run dev
```

Port `8080` belongs to the frontend only when the Compose `frontend` service is
running. With `npm run dev`, use the Vite URL printed in the frontend terminal.

## Tests

The default suite is hermetic and skips tests that require PostgreSQL:

```bash
uv run pytest -q
```

To run the database integration suite, start PostgreSQL with pgvector, migrate
the database selected by `DATABASE_URL`, and opt in explicitly:

```bash
docker compose up -d db
DATABASE_URL=postgresql+asyncpg://pipecat_user:pipecat_password@localhost:5434/pipecat_db \
  uv run alembic upgrade head
RUN_DATABASE_TESTS=1 \
DATABASE_URL=postgresql+asyncpg://pipecat_user:pipecat_password@localhost:5434/pipecat_db \
  uv run pytest -q tests/test_calls.py tests/test_calls_api.py
```
