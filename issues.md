# Project issue remediation ledger

Issues are ordered from highest to lowest priority. An item is marked `done` only after its implementation and listed verification have succeeded.

1. **Critical — CORS policy is effectively permissive** — `done`
   - Pipecat's runner installs its own CORS middleware from `PIPECAT_ALLOWED_ORIGINS`, overriding the intent of the application's `ALLOWED_ORIGINS` configuration.
   - Verification: configuration/unit test proves untrusted origins are excluded and Pipecat receives the explicit allow-list.

2. **Critical — Voice session authentication fails open** — `done`
   - Missing or invalid voice session tokens currently fall through to an anonymous session instead of rejecting startup.
   - Verification: tests prove missing/invalid tokens are rejected before provider construction, while valid sessions still initialize.

3. **Critical — Link ingestion has incomplete SSRF and robots protections** — `done`
   - Redirect targets are not validated, DNS/public-address validation is incomplete at fetch time, Crawl4AI can fetch uncontrolled subresources, and the existing robots helper is not enforced.
   - Verification: tests cover private targets, redirects to private targets, robots denial, and the safe default extractor.

4. **High — RAG ingestion runs inside the latency-sensitive voice API process** — `done`
   - CPU-heavy parsing shares the voice event loop/process, creates FFmpeg/OpenCV collision risk, and uses an in-memory queue that loses jobs on restart.
   - Verification: API tests prove uploads become durable queued jobs; worker tests prove claiming/processing; migration/UI tests cover queued state; deployment config runs a separate worker.

5. **High — Local LLM capacity configuration admits more sessions than llama.cpp serves** — `done`
   - Checked-in launch commands use one llama.cpp slot (`-np 1`) while the backend allows two concurrent local-LLM sessions.
   - Verification: configuration tests prove the launch command and backend admission limit agree, and startup validation detects insufficient server slots.

6. **High — Local LLM warmup does not warm the real prompt prefix/cache** — `done`
   - Startup warms an unrelated tiny prompt, leaving the first real system prompt to pay a large prompt-evaluation cost.
   - Verification: tests prove every admitted slot warms the production system prompt and sends `cache_prompt=true`.

7. **High — Background memory work can contend with an active spoken response** — `done`
   - Deferred work is released on first synthesized audio rather than after TTS playback generation finishes, allowing local inference contention mid-response.
   - Verification: gate tests prove deferred work stays blocked through first audio and releases on TTS completion/interruption; provider tests prove memory work can be separated from the local voice LLM.

8. **High — Authentication hashing blocks the event loop and auth endpoints lack throttling** — `done`
   - bcrypt calls run synchronously in async routes, and login/registration can be brute-forced without an application-level rate limit.
   - Verification: tests prove hashing runs through a worker thread and repeated attempts receive HTTP 429.

9. **High — Live web searches are cancelled after one four-second Tavily attempt** — `done`
   - The filler is working, but it only masks the wait; the provider call is still terminated at exactly four seconds with no retry or useful attempt telemetry.
   - Verification: `tests/test_tool_timeouts.py` and `tests/test_tavily.py` pass (10 tests), including transient retry, explicit provider deadlines, and the outer-deadline invariant.

10. **High — Time-sensitive questions can bypass search and produce stale answers** — `done`
   - Search is merely exposed to the LLM, so it answered current/latest questions from model memory, missed the initial live-protest question, and invented an unsupported `2024` qualifier when later asked to “search for it.”
   - Verification: superseded by the universal semantic tool availability in issue 16; live Tavily and contextual-query probes still prove bounded result delivery and conversation-aware search input.

11. **High — A two-slot llama.cpp server loses the growing conversation cache between turns** — `done`
   - The log alternates between slot-local prompt caches, periodically re-evaluating hundreds of tokens and adding roughly one second to otherwise fast direct turns.
   - Verification: 31 admission/local-provider/readiness tests pass. They prove unique deterministic slot leases, explicit `id_slot` propagation with prompt caching, release, and reuse.

12. **High — Deterministic pre-LLM search forwards corrective follow-ups verbatim** — `done`
    - Code currently constructs Tavily input from the latest transcript, bypassing the LLM's conversation-aware query planning and producing searches such as `You are wrong with the camera specs`.
    - Verification: deterministic web-query construction was removed from context retrieval. The exact conversation produced the local-model tool query `Galaxy A30s camera specifications correct details`; tests prove full-history planning and a tool-disabled answer pass after results.

13. **High — Local LLM responses can be cut off mid-answer at the fixed completion limit** — `done`
   - `LOCAL_LLM_MAX_TOKENS=192` terminated a web-grounded response mid-sentence, and the provider streamed that incomplete answer directly to the UI/TTS without recovery.
   - Verification: the active configuration loads a 512-token response budget and 30-second stream deadline; provider tests prove automatic continuation only on `finish_reason=length`, while normal and tool-call completions remain single requests. A live llama.cpp probe completed naturally at 284 tokens with `finish_reason=stop`.

14. **High — Malformed unquoted `.env` text can leak subsequent secret-bearing lines in parser diagnostics** — `done`
   - Apostrophes and punctuation in unquoted voice messages caused `uv --env-file` to report malformed input and include following lines in its diagnostic output.
   - Verification: affected values are quoted and `uv run --env-file ../.env` now loads without parser warnings.

15. **High — The LLM lacks trusted current-date context and an exact-time capability** — `done`
   - Model knowledge can anchor answers to an obsolete year. The initial fix re-appended date/timezone metadata after every user message, unnecessarily invalidating part of the reusable prompt prefix.
   - Verification: each voice session receives date/timezone exactly once in its durable system instruction, per-turn routing does not mutate conversation messages, and local warmup uses the same dated prompt builder. The always-exposed clock tool still returns exact zoned time and rejects invalid IANA zones. The focused prompt/provider/routing suite passes (65 tests), the full backend suite passes (360 tests), and a runtime prompt probe contains exactly one date marker with no clock time.

16. **High — Regex-gated Tavily availability rejects valid semantic search decisions** — `done`
   - Current-year movie wording bypassed the regex gate, so Google correctly attempted `tavily_search` from conversational context but Pipecat rejected it because only the clock tool was advertised. The system prompt also retained conditional-availability language from that obsolete routing design and incorrectly treated user corrections as authoritative facts.
   - Verification: Tavily and the clock tool are advertised with automatic choice on every ordinary turn; no search regex, forced-search marker, named Tavily choice, obsolete conditional-search instruction, or instruction to accept an unverified user correction remains. Prompt tests also require complete spoken answers, contextual standalone queries, and verification of disputed claims. The focused tool/prompt suite passes (47 tests), and the full backend suite passes (360 tests).

17. **High — Live context is hard-trimmed instead of summarized into durable session state** — `done`
   - The live Pipecat context was capped at 12 messages/6,000 characters without native summarization. Stable facts and the prior database summary were inserted as developer messages, which Pipecat may summarize or drop, while a separate periodic database summarizer created a competing summary lifecycle.
   - Verification: stable facts and the canonical summary now live in the escaped session system-instruction data block; only recent primary dialogue seeds mutable context; a dedicated Groq model performs tool-safe native Pipecat summarization; applied summaries update the canonical database field; query/tool payloads are excluded; destructive mutations reject stale summaries; provider cancellation activates cooldown; emergency and disabled legacy trimming paths are bounded. A live Groq request produced a non-empty summary using `llama-3.1-8b-instant`, reasoning-model fallback controls are tested, and the full 388-test backend suite passes.

18. **Medium — FFmpeg/OpenCV load incompatible libavdevice copies in the voice process** — `done`
   - The duplicate-class warning still appears during WebRTC setup, so separating RAG ingestion did not remove the conflicting binary load path.
   - Verification: Pipecat's unconditional `smallwebrtc.transport` OpenCV import was identified. Four WebRTC/startup tests pass, and a fresh process loads Small WebRTC plus PyAV with the audio-only shim, no native `cv2`, and no duplicate Objective-C class warning.

19. **Medium — Voice latency telemetry depends entirely on a browser playback callback** — `done`
    - If the remote-audio-level event never arrives, otherwise valid server latency measurements are silently lost.
    - Verification: tests prove a server-side measurement is persisted independently and client playback telemetry remains distinguishable.

20. **Medium — Slow-processor diagnostics report expected streaming frames as stalls** — `done`
    - Normal TTS text/aggregation frames are not excluded, producing misleading warnings in healthy turns.
    - Verification: observer tests prove expected streaming frame classes do not trigger slow warnings.

21. **Medium — Kokoro's steady-state first-audio floor and adapter startup cost are not tuned** — `done`
    - The adapter import occurs during session construction and fixed chunking defaults are not exposed as a safely testable latency/quality trade-off.
    - Verification: startup tests prove the adapter is pre-imported; configuration tests cover bounded low-latency chunk settings.

22. **Low — Groq provider regression test asserts the obsolete request shape** — `done`
    - The implementation correctly nests provider options under `extra_body`, but one test still expects the old flattened arguments.
    - Verification: the updated provider test and the complete targeted suite pass.

## Final verification

- `uv run pytest -q`: 388 passed (2 third-party deprecation warnings).
- Live-context configuration probe: enabled with the balanced 3,000-token/20-message trigger, 900-token target, eight-message retention, and 40-message emergency cap.
- Live dedicated-Groq summary probe: passed with `llama-3.1-8b-instant`, a non-empty 276-character summary, and the expected safe summary boundary.
- Live Google semantic-routing probe: both read-only tools advertised with `auto`; current-year movie request called `tavily_search` with a contextual 2026 query.
- Live local clock-tool planning probe: selected `get_current_datetime` with `{"timezone":"America/New_York"}` and `finish_reason=tool_calls`.
- Live local-LLM completion probe: 284 tokens, 1,469 characters, `finish_reason=stop` (past the former 192-token cutoff).
- Exact Galaxy A30s correction probe: local LLM produced a contextual Tavily tool call rather than forwarding the correction text.
- Live Tavily probe: 3 results in about 2.05 seconds.
- Fresh-process audio-only WebRTC/PyAV import: passed without native OpenCV or duplicate-class warnings.
- `npm test`: 6 passed.
- `npm run build`: production build passed.
- `docker compose config --quiet`: passed.
- `uv lock --check`: passed.
- `uv run alembic heads`: one head, `20260810_durable_rag_queue`.
- `git diff --check`: passed.
