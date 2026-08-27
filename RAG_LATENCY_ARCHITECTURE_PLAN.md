# Universal low-latency conversational RAG plan

Status: living issue ledger, architecture, and implementation record.

## Objective

Build a source-agnostic conversational retrieval path that:

- keeps ordinary voice turns on the current direct path;
- understands corrections, ellipsis, pronouns, retries, and source changes across turns;
- works for arbitrary users, entities, document types, languages, and document contents;
- retrieves exact facts quickly without sacrificing semantic recall;
- avoids an extra LLM/tool round trip when routing can be decided before inference;
- injects only sufficient, grounded evidence into the local LLM;
- remains measurable and tunable on the target M4/16 GB machine.

The system must not encode knowledge of a particular person, filename, question,
or transcript. Rules may describe general dialogue acts, source capabilities, and
confidence signals, but entities and constraints must always be derived from the
live conversation and authenticated source metadata.

## Evidence from the 2026-08-24 trace

The trace established the following critical-path timings after final STT:

| Path | Retrieval | LLM first output | First TTS audio | Outcome |
|---|---:|---:|---:|---|
| Initial direct turn | none | 131 ms | 482 ms | successful |
| Contextualized short source correction | 703 ms | 367 ms | 1,398 ms | false no-match |
| Generic source-presence statement | 558 ms | 372 ms | 1,236 ms | no-match |
| Complete single-turn RAG request | 572 ms | 980 ms | 1,944 ms | successful |
| Direct gratitude after RAG | none | 911 ms | 1,317 ms | unnecessarily slow |

Additional evidence:

- Normal lexical database retrieval took 4.7-18.6 ms.
- Lexical retrieval during barge-in took 328.4 ms while event-loop lag reached
  271 ms.
- Query embedding took 546-561 ms in normal RAG turns and reached the 700 ms
  vector-fusion deadline in the interrupted turn.
- The successful result had vector similarity 0.602, below the configured 0.62
  vector threshold; lexical evidence was what made it acceptable.
- The failed contextual query had text rank 0.40 but was rejected because
  conversational filler diluted lexical-overlap coverage.
- The successful RAG prompt required 322 uncached tokens and 965 ms of llama.cpp
  prompt processing.
- A later direct gratitude turn required 297 uncached tokens and 891 ms of prompt
  processing.
- Moonshine reported roughly 655-721 ms from the VAD frame to the final callback.
  Current first-audio telemetry starts at final STT and therefore excludes much
  of the user-perceived endpoint latency.
- A barge-in produced a zero-duration extra turn boundary, making a strict
  one-turn conversational-state lifetime unsafe.
- The RAG turns in this trace exposed zero tools. The deterministic pre-LLM RAG
  path was active; the native two-pass tool path was not the cause of these
  particular timings.

## Issue ledger

### Correctness and conversational state

1. **Routing text, retrieval text, and scoring text are conflated.**
   The complete conversational utterance is useful for deciding intent, but it
   is often a poor lexical query and a poor denominator for relevance coverage.

2. **Contextualization preserves the subject but adds conversational noise.**
   Concatenating a previous request and a short follow-up correctly preserves
   missing information, but tokens such as polite framing and the literal
   `Follow-up` marker reduce lexical coverage and can prevent early exit.

3. **The stop-word approach is necessarily incomplete and language-dependent.**
   An ever-growing English list cannot reliably distinguish request framing from
   document-bearing terms across languages, accents, STT variants, or domains.

4. **Corrections are appended instead of represented as state updates.**
   A source correction can leave contradictory source words in one query. Source
   type, selected files, time range, and other constraints should be structured
   fields, not semantic query padding.

5. **A no-match does not retain the unresolved information need.**
   Timeout and failure create retry state, but no-match only installs a status
   message. A subsequent clarification can therefore search its own generic text
   instead of retrying the unresolved subject.

6. **Recent-query lifetime is coupled to raw turn boundaries.**
   Barge-in and duplicated/zero-duration boundaries can expire useful state even
   when the user is clearly continuing the same task.

7. **Source-management intent and content-retrieval intent are conflated.**
   Asking whether a file exists, is uploaded, or has finished processing should
   use cheap source metadata. It should not embed the sentence and search chunks.

8. **Grounded continuation evidence is installed too broadly.**
   Evidence useful for a referential follow-up is unnecessary for gratitude,
   closure, cancellation, or a new topic. Unconditional injection adds tokens
   and can bias the response toward stale evidence.

9. **Confidence thresholds are not calibrated as one coherent decision.**
   Text rank, lexical coverage, vector similarity, final score, and early-exit
   coverage use separate fixed thresholds. Their interaction can reject a good
   exact match or wait for a vector result that does not alter the decision.

10. **A generic no-match can become a misleading conversational answer.**
    The system distinguishes unavailable sources from no matching passage, but
    weak query construction still causes avoidable no-match responses.

### Retrieval latency and efficiency

11. **Remote query embedding dominates normal RAG latency.**
    In the observed successful turn, embedding consumed about 561 of 570 ms.
    Lexical search and vector database lookup together consumed about 12 ms.

12. **Vector work begins even for exact lexical requests.**
    Parallel launch minimizes ambiguous-query latency but spends network and
    deadline budget on names, identifiers, dates, and exact phrases that lexical
    retrieval can often resolve decisively.

13. **The lexical early-exit denominator includes non-evidence-bearing words.**
    A high text rank can still fail early exit because request framing is counted
    as if it must appear in the document.

14. **Generic source-status utterances waste the complete semantic path.**
    The observed source-presence query spent 558 ms retrieving content that the
    utterance did not actually identify.

15. **Exact-query caching cannot help varied conversational paraphrases enough.**
    Cache keys should remain safe and corpus-versioned, but the architecture also
    needs an intent/signature level at which equivalent resolved requests can be
    recognized without serving evidence from the wrong scope.

16. **Embedding model space is a deployment constraint.**
    Ingestion and query vectors must use the same model and dimension. Changing
    to a local embedding provider requires a controlled re-index, not a runtime
    fallback between incompatible providers.

### Prompt and generation latency

17. **RAG evidence is expensive for a 4B local model to prefill.**
    Hundreds of uncached prompt tokens add hundreds of milliseconds even when
    retrieval itself is fast.

18. **Adding and removing query-scoped messages reduces longest-prefix reuse.**
    llama.cpp caching is working and session requests are slot-pinned, but a
    changing prompt suffix must still be reprocessed. Large evidence blocks make
    this expected invalidation expensive.

19. **The evidence block is chunk-oriented rather than answer-oriented.**
    A full chunk may contain much more than the sentences needed to answer the
    resolved question.

20. **Voice response length is insufficiently controlled.**
    A gratitude turn generated 132 tokens and occupied the model for 5.3 seconds.
    Streaming hides some latency, but long low-value responses consume the local
    model, complicate barge-in, and delay other work.

21. **Native tool fallback can add a second model pass.**
    It is appropriate for ambiguous capabilities, but explicit document content
    requests should normally be resolved before the first LLM pass.

### Realtime scheduling, turn identity, and measurement

22. **User-perceived endpoint latency is not the primary displayed basis.**
    Final-STT-to-audio is useful, but it hides the observed 655-721 ms STT
    finalization interval. Both measurements are needed.

23. **Moonshine final callback is a major latency component.**
    Native final inference was much shorter than the total VAD-to-callback time,
    so callback scheduling, delivery gating, stream cadence, and the native VAD
    window require isolated measurement.

24. **Barge-in can create resource contention and unstable turn boundaries.**
    The trace shows 271 ms event-loop lag, a 328 ms lexical query, cancellation
    of an earlier LLM response, and an immediate zero-duration extra turn.

25. **Deferred-work gating has a potential speech-phase gap.**
    The realtime gate begins at final STT, while a new user start first releases
    the previous key. Deferred enrichment can therefore become eligible while
    the user is speaking. This is an architectural risk; the trace does not prove
    that enrichment caused the observed lag.

26. **Component deadlines are present but not governed by one turn budget.**
    A deadline prevents unlimited delay, but each branch should also know whether
    its expected information gain justifies spending the remaining voice budget.

27. **Filler speech can compete with the real answer.**
    Filler is a perceived-latency tool, not a compute optimization. On a serialized
    or contended TTS path it can delay answer audio. It must be delayed, cancellable,
    and driven by predicted remaining work.

28. **An active transactional workflow can fall back to unaudited free-form text.**
    An exposed tool with `tool_choice=auto` does not guarantee that a small local
    model will call it on short continuations such as a confirmation or progress
    question. The model can describe a transition that the backend never made.

29. **Transport completion is not business completion.**
    A tool invocation can complete successfully while its result remains
    `collecting_fields` or `awaiting_confirmation`. UI status must display the
    result state rather than relabeling every returned invocation as completed.

30. **Sensitive fields can span multiple finalized STT fragments.**
    Turn-completeness analysis and final transcription boundaries are fallible.
    A schema field such as an email address can be split across adjacent finals;
    treating each fragment independently loses valid information.

31. **Context summarization model availability is a runtime dependency.**
    A retired or unavailable provider model produces a degraded diagnostic and
    leaves the live call running, but repeated failure prevents context compaction.

32. **Idle-work admission alone does not prevent an idle-to-live race.**
    A summarization or embedding job can pass an idle check immediately before a
    new voice turn begins. Best-effort work must be registered and preemptible.

33. **Tool-only model outputs were absent from latency telemetry.**
    A function-call response may contain no text frame, causing later
    tool-generated speech to miss first-output and first-audio metrics.

34. **Tool-schema telemetry did not understand Pipecat's schema wrapper.**
    Counting only plain lists reported zero tools even when a `ToolsSchema`
    contained standard and custom tools.

35. **Diagnostic logs exposed raw private utterances and spoken answers.**
    Local deployment does not remove the need for privacy-safe logs, especially
    when complaint identifiers and contact details can appear in queries or TTS.

## Universal architecture

### 1. Separate the three decisions

Every completed user turn should produce three independent artifacts:

1. **Route decision** — direct response, source metadata operation, private
   content retrieval, public web retrieval, action workflow, or clarification.
2. **Resolved information need** — the source-agnostic subject, requested
   operation, constraints, and relationship to an unresolved prior turn.
3. **Retrieval plan** — cache, metadata filter, exact/lexical search, semantic
   fallback, reranking, and evidence budget.

The raw utterance may influence all three, but the same string must not be used
unchanged for all three.

### 2. Maintain a small dialogue retrieval state

Maintain state per authenticated live call, independent of LLM chat history:

| Field | Purpose |
|---|---|
| task id | Stable identity for one unresolved information need |
| status | idle, pending, retrieving, grounded, unresolved, or closed |
| operation | lookup, summarize, compare, list, quote, verify, or source status |
| subject signature | High-signal content terms, literals, and relationships |
| source scope | Source type, selected source IDs, collection, or all private sources |
| constraints | Dates, pages, sections, people, products, places, and other derived filters |
| query variants | Current, contextual, exact, and semantic forms |
| evidence anchor | Source IDs, chunk IDs, scores, and compact answer evidence |
| last outcome | success, no-match, timeout, failure, interrupted, or superseded |
| lifecycle | Completed-turn count, last activity time, and corpus version |

This is not a domain schema. Values are extracted from the user's own language
and source metadata. Unknown fields remain unknown.

State transitions should be based on completed semantic turns:

```text
idle
  -> pending       source-related request without enough information
  -> retrieving    complete source-related request

retrieving
  -> grounded      sufficient evidence found
  -> unresolved    no-match, timeout, failure, or interruption

grounded/unresolved
  -> retrieving    correction, refinement, retry, or referential follow-up
  -> closed        gratitude, cancellation, explicit completion, expiry, or topic change
```

No-match remains unresolved; it is not equivalent to a completed task.

Use both a short wall-clock expiry and a completed-semantic-turn limit. Do not
expire this state solely because VAD emitted an additional start/stop boundary.

### 3. Resolve dialogue without another generative LLM pass

The fast resolver should produce structured output, not polished prose:

- preserve literals such as names, identifiers, quoted phrases, dates, amounts,
  page references, and uncommon content terms;
- classify the dialogue act as new request, continuation, correction, refinement,
  retry, closure, or unrelated topic;
- update source scope and constraints separately from subject terms;
- never append implementation markers such as `Follow-up` to lexical terms;
- carry prior subject information only when continuation confidence is adequate;
- ask a concise clarification when safe resolution is not possible.

The initial resolver can be deterministic and data-driven. Later, a tiny local
classifier may replace or augment it if evaluation proves rules insufficient.
Using the 4B generation model for every rewrite would add exactly the extra pass
the low-latency design is trying to avoid.

When resolution is uncertain, build multiple bounded query variants rather than
pretending one rewrite is certainly correct:

- exact/literal variant;
- current-turn high-signal variant;
- contextual variant containing unresolved subject plus current refinements;
- semantic natural-language variant.

Candidate results can be fused while preserving which variant produced them.

### 4. Use source metadata as a first-class routing layer

Before content retrieval, distinguish:

- source existence/list/status operations;
- ingestion readiness or failure;
- source selection or source switching;
- questions about content.

Source references should resolve against authenticated metadata. Source type is
a filter or scope signal, not a term that must appear inside document content.
This supports PDFs, saved links, text documents, future media transcripts, and
collections through the same interface.

### 5. Use a confidence-driven retrieval cascade

Run the cheapest stage capable of answering the resolved need:

1. **Corpus and source-state cache** — return source status without content search.
2. **Resolved-result cache** — keyed by user, corpus version, source scope, operation,
   and normalized intent signature.
3. **Exact lookup** — identifiers, quoted phrases, dates, page/section metadata,
   filenames, and exact entity sequences.
4. **Lexical retrieval** — FTS/BM25-style candidates using only high-signal terms.
5. **Semantic vector fallback** — for paraphrase, conceptual, or low-coverage needs.
6. **Optional reranking** — only when candidates are ambiguous enough to justify it.

For typical 5-20 ms lexical queries, prefer a lexical-first or short hedged design:

- start lexical immediately;
- delay vector embedding by a small measured hedge interval, or start it only
  after lexical fails to become decisive;
- cancel vector work when exact/lexical confidence is already sufficient;
- do not run semantic retrieval for a pure source-status operation.

The hedge interval must be selected from measured p95 lexical latency on the
target machine, not from a fixed universal number.

### 6. Make evidence acceptance a calibrated decision

Candidate generation and evidence acceptance are different decisions. Acceptance
should combine:

- exact literal coverage, with identifiers weighted most strongly;
- coverage over high-signal query terms only;
- lexical rank;
- vector similarity for the active embedding model;
- source-scope match;
- top-result margin over competing candidates;
- agreement across query variants;
- document/chunk quality and boilerplate penalties;
- whether enough evidence exists for the requested operation.

Do not count politeness, routing language, source nouns, or internal query markers
in semantic coverage. Do not assume one vector threshold transfers across models.
Calibrate thresholds against an evaluation corpus and record the reason a chunk
was accepted or rejected.

High-confidence exact evidence may bypass vectors. Low-confidence evidence must
not be injected merely to meet a latency target.

### 7. Build answer-oriented evidence

After selecting chunks, construct the smallest evidence packet sufficient for
the requested operation:

- extract a sentence window around matched literals or semantic spans;
- retain source identity and page/section provenance;
- include adjacent context only when needed to avoid changing meaning;
- deduplicate overlapping passages;
- allocate more tokens for summarize/compare operations than factual lookup;
- enforce a total evidence budget derived from measured prompt-prefill speed.

The raw retrieved chunks remain available for audit, but the LLM prompt should
receive compact answer evidence rather than the complete audit payload.

### 8. Preserve llama.cpp cacheability

- Keep system instructions and stable session memory in an immutable prefix.
- Keep routing/tool schemas absent from direct turns unless required.
- Append compact query-scoped evidence at the prompt tail.
- Remove raw evidence after its answer, retaining only ordinary conversation and
  a small evidence anchor when a referential follow-up is plausible.
- Do not install the anchor for gratitude, closure, cancellation, or unrelated topics.
- Measure `cache_n`, `prompt_n`, and prompt milliseconds for every request.

Some suffix reprocessing after ephemeral RAG context is unavoidable. The goal is
to minimize the invalidated suffix, not to claim it can always be eliminated.

### 9. Protect realtime work on the M4

Define the realtime critical interval as speech start through answer audio release,
not merely final STT through LLM completion.

- Do not start local background LLM work during this interval.
- Make deferred work re-check or acquire the gate immediately before expensive work.
- Prefer cancellation/preemption for best-effort enrichment when a user starts speaking.
- Cancel superseded LLM, retrieval, filler, and TTS work promptly on barge-in.
- Bound concurrent CPU/GPU/ANE-heavy tasks using measured resource profiles.
- Keep persistence asynchronous, but ensure it cannot starve the event loop.
- Evaluate a small warmed local embedding model separately and concurrently with
  Moonshine, Qwen, and Kokoro before adopting it.

### 10. Treat filler as an adaptive UX branch

Emit spoken filler only when predicted remaining latency exceeds both:

- the configured silence tolerance; and
- the time/cost of opening a filler TTS context.

The filler must be delayed, cancellable, non-contextual, and limited to one per
turn. Visual retrieval state can appear immediately without occupying TTS.

### 11. Measure the complete latency chain

Record both user-perceived and component-relative timings:

```text
audio speech end
  -> VAD confirmation
  -> final STT callback
  -> resolved route/query
  -> first candidate / retrieval complete
  -> prompt submitted
  -> first model output
  -> first speakable text
  -> TTS request
  -> first generated audio
  -> first transported/playable audio
```

For every retrieval decision, record privacy-safe structured diagnostics:

- route and dialogue-act decision;
- whether prior state was used and why;
- query-variant fingerprints and high-signal term counts;
- source scope and corpus version;
- cache hit/miss;
- candidate counts and per-stage latency;
- acceptance signals and selected evidence size;
- embedding provider/model and whether vector work affected the decision;
- llama.cpp cache/prompt token counts;
- event-loop lag and competing workload classes;
- cancellation, supersession, and barge-in state.

Raw private queries and document passages should remain in the authenticated audit
surface only where required; aggregate telemetry should prefer IDs, counts, buckets,
and hashes.

## Latency policy

Budgets must be calibrated on the production hardware and reported at p50, p95,
and p99. The following are design targets, not hardcoded constants:

| Stage | Exact/source-scoped target | Semantic fallback target |
|---|---:|---:|
| routing + state resolution | <= 5 ms | <= 5 ms |
| source/cache lookup | <= 10 ms | <= 10 ms |
| lexical retrieval | <= 30 ms p95 | <= 30 ms p95 |
| embedding + vector fallback | skipped | <= 250 ms local goal; bounded remote fallback |
| evidence construction | <= 10 ms | <= 20 ms |
| LLM first speakable output after context | <= 500 ms | <= 800 ms |
| TTS first audio after text | <= 300 ms | <= 300 ms |
| post-final-STT first audio | <= 1.0 s | <= 1.5 s |

Separately target and report endpoint-to-final-STT latency. It cannot be hidden
inside the post-final metric.

Each optional branch must implement an information-gain rule: spend remaining
latency only when the branch is likely to change evidence acceptance or answer
quality. A timeout is a safety ceiling, not a reason to wait until the ceiling.

## Evaluation plan

Build a provider- and user-independent evaluation set with expected routes,
resolved needs, relevant chunks, and answerability labels.

### Conversation categories

- direct non-source chat;
- explicit source question in one turn;
- source mentioned only after the subject;
- subject mentioned only after the source;
- pronoun and elliptical follow-ups;
- corrections to subject, source, date, or requested operation;
- retry after timeout, failure, and no-match;
- topic change after grounded evidence;
- gratitude, closure, and cancellation after RAG;
- source existence, list, upload, and ingestion-status questions;
- exact names, identifiers, dates, amounts, pages, and quoted phrases;
- semantic paraphrases with little lexical overlap;
- summaries, comparisons, enumeration, and multi-source synthesis;
- multiple matching and conflicting documents;
- unavailable, processing, failed, empty, and newly updated corpora;
- STT substitutions, partial names, accents, and multilingual utterances;
- barge-in before retrieval, during LLM generation, and during TTS;
- document prompt injection and untrusted instructions.

### Quality metrics

- route precision/recall for each operation class;
- resolved-intent preservation and incorrect carry-over rate;
- retrieval Recall@k, MRR/nDCG, and exact-literal recall;
- evidence acceptance precision and false-injection rate;
- grounded-answer correctness and citation/provenance accuracy;
- truthful no-match/status accuracy;
- unnecessary vector-call and unnecessary tool-pass rates;
- evidence tokens per answered turn;
- barge-in cancellation correctness and state continuity.

### Latency metrics

- endpoint-to-final-STT p50/p95/p99;
- final-STT-to-first-audio p50/p95/p99 by route;
- lexical, embedding, vector DB, fusion, evidence, prompt-prefill, generation,
  and TTS distributions;
- llama.cpp cached versus uncached prompt tokens;
- event-loop lag and workload overlap;
- result/cache hit rates and saved milliseconds;
- percentage of exact requests answered without embeddings;
- percentage of turns that invoke a second LLM pass.

Every optimization must pass both the quality and latency gates. A faster system
that silently increases false grounding or cross-topic carry-over does not pass.

## Phased implementation plan

### Phase 0 — Freeze the baseline

- Preserve the supplied trace as a regression fixture with private content removed.
- Add the full end-to-end latency basis from audio endpoint to playable audio.
- Add diagnostics for resolved state, query variants, high-signal terms, acceptance
  reason, vector decision impact, and llama.cpp cache reuse.
- Establish p50/p95 baselines for direct, exact RAG, semantic RAG, no-match, and barge-in.

Exit criterion: every millisecond on the critical path and every retrieval outcome
can be attributed to a named stage and decision.

### Phase 1 — Correct universal dialogue state

- Introduce the source-agnostic retrieval task state.
- Separate route, operation, subject signature, source scope, and constraints.
- Treat correction/refinement/retry as state updates.
- Preserve unresolved state after no-match as well as timeout/failure.
- Base expiry on completed semantic turns plus time, not raw VAD starts.
- Gate continuation evidence on dialogue act.
- Route source status/list/readiness to metadata rather than chunk search.

Exit criterion: the multi-turn evaluation set matches expected resolved needs with
no scenario-specific names, filenames, or question templates in runtime logic.

### Phase 2 — Install the retrieval cascade

- Create exact, high-signal lexical, contextual, and semantic query variants.
- Add lexical-first or measured hedged vector execution.
- Make exact literals and source scope first-class scoring signals.
- Calibrate one coherent evidence-acceptance policy.
- Record when vector retrieval changes the selected evidence; eliminate it when it does not.

Exit criterion: high-precision factual queries normally avoid embeddings while
semantic-query recall does not regress.

### Phase 3 — Reduce prompt cost

- Build answer-oriented sentence windows with provenance.
- Use operation-specific evidence budgets.
- Avoid evidence anchors on closure/unrelated turns.
- Preserve a stable prompt prefix and measure the invalidated suffix.
- Add voice response-length policies appropriate to dialogue act and operation.

Exit criterion: RAG prompt-prefill and post-RAG direct-turn latency meet targets
without reducing grounded-answer quality.

### Phase 4 — Fix realtime scheduling and STT endpoint latency

- Audit the zero-duration barge-in turn and establish one canonical completed-turn ID.
- Protect speech start through answer audio in realtime scheduling.
- Make deferred enrichment re-check/cancel on renewed activity.
- Break down Moonshine force-update, callback, delivery-gate, and drain latency.
- Benchmark supported endpoint/VAD settings rather than changing them blindly.
- Verify cancellation under simultaneous Qwen, Kokoro, Moonshine, DB, and retrieval load.

Exit criterion: no background work enters the live critical path, barge-in preserves
state exactly once, and endpoint-to-final-STT meets its measured target.

### Phase 5 — Decide embedding deployment

- Benchmark the current remote provider under warm/cold and network-tail conditions.
- Benchmark one small local embedding model under realistic concurrent voice load.
- Compare latency, memory pressure, thermal behavior, retrieval quality, and index size.
- If changing model space, perform a versioned full re-index with no cross-provider fallback.
- Retain remote vectors only if their quality/operational advantage justifies their tail latency.

Exit criterion: one embedding model space is selected from evidence, documented,
versioned, and monitored.

### Phase 6 — Tune, canary, and guard against regression

- Tune thresholds on the evaluation set, never on a single transcript.
- Run shadow retrieval before changing production answers.
- Canary by session/user cohort with rollback controls.
- Alert on route drift, no-match spikes, vector-call rate, prompt-token growth,
  event-loop lag, and latency SLO regression.
- Keep a compact permanent regression suite for every failure class above.

Exit criterion: quality and latency SLOs hold over representative real conversations,
including adversarial and barge-in cases.

## Recommended order of attack

1. Fix state semantics: retain no-match tasks, distinguish source metadata from
   content retrieval, and gate evidence continuation.
2. Separate high-signal retrieval terms from conversational context.
3. Make exact/lexical evidence decisive before paying for vectors.
4. Compact the evidence prompt and control voice response length.
5. Correct turn identity and measure/tune Moonshine endpoint latency.
6. Evaluate local embeddings only after unnecessary vector calls are removed.

This order addresses correctness first, then removes work. It avoids adding a new
model or complicated router before proving that the existing fast lexical path can
handle the large class of exact, source-scoped questions already visible in the trace.

## Non-goals and invariants

- Do not run RAG on every turn merely because a user owns documents.
- Do not use the main generative model for routine query rewriting.
- Do not weaken grounding thresholds globally to hide poor query construction.
- Do not treat a no-match as proof that a source is absent.
- Do not carry prior evidence into unrelated or closing turns.
- Do not mix embeddings from different model spaces.
- Do not optimize only final-STT-to-audio while ignoring endpoint-to-final-STT.
- Do not accept latency gains that reduce evidence precision, authorization scope,
  provenance, or document prompt-injection defenses.

## Implementation status — first universal slice

Implemented on 2026-08-24:

- Unresolved timeout, failure, and no-match retrievals retain their resolved query
  for the next clarification turn.
- Source corrections replace obsolete source qualifiers while preserving the
  subject, operation, dates, and identifiers of the original information need.
- Retrieval no longer injects an internal `Follow-up:` label into search or
  embedding text.
- Lexical search uses ordered content-bearing terms rather than conversational
  scaffolding and source-routing words.
- A fully covered, high-signal lexical result can cancel the in-flight embedding
  branch and release the voice turn immediately.
- Source availability, readiness, and count questions use authenticated corpus
  metadata rather than semantic chunk retrieval.
- Grounded evidence is installed on plausible referential follow-ups, but not on
  acknowledgements, closures, or source-status questions.
- Context expiry advances on completed aggregated user messages, not duplicate
  speech-start/VAD events.

Verification for this slice:

- 82 RAG tests passed; one unrelated Docling/MLX ingestion test was excluded
  because importing `mlx_whisper` aborts the local Python process natively.
- 13 issue/evidence-authorization tests passed.
- 31 RAG-tool, tool-routing, and filler tests passed.
- Python compilation and `git diff --check` passed.

## Implementation status — workflow and realtime reliability slice

Implemented on 2026-08-24 from the 16:00 trace:

- Any active complaint workflow now forces the backend workflow controller when
  no unrelated tool is selected. The controller supports `status` and `defer`,
  and all of its spoken responses bypass a second LLM pass.
- Submission remains idempotent and is only reported after persistence returns a
  success result with an issue ID. Progress questions report the persisted draft
  state; they cannot be presented as completed transport work in the UI.
- Adjacent finalized fragments are retained in a short, bounded field buffer only
  while a schema field is being collected. Email recovery is structural and
  schema-validated; it contains no customer, provider, or scenario-specific value.
- A source-only correction with no evidence-bearing terms now replaces obsolete
  source scope while retaining the prior unresolved subject, even without a
  predefined correction phrase.
- Moonshine now finalizes each upstream-VAD-bounded utterance through the native
  stream flush path. Only the lightweight stream rotates; the shared 245M model
  remains loaded. This removes the later-silence dependency of a force-update on
  an open stream.
- Best-effort enrichment registers its running child task against the realtime
  gate. A new voice turn cancels it and the queue retries the same item only after
  the call becomes idle again.
- The configured Groq summary model now uses the available
  `openai/gpt-oss-20b` profile instead of the unavailable retired model seen in
  the trace.
- Function-call frames now count as first LLM output for tool-only turns, and
  tool counts include Pipecat standard and custom schema entries.
- Retrieval/router/TTS logs now emit length, word count, and a truncated SHA-256
  fingerprint rather than raw user or assistant content.

Verification for this slice:

- 411 backend tests passed with 19 environment-dependent skips outside the RAG
  module.
- 84 of 85 RAG tests passed; the existing native Docling/MLX parser test remains
  deselected because it aborts the Python process on this environment.
- 12 frontend tests passed; the production build and lint both passed.
- Python compilation and `git diff --check` passed.

Still deliberately deferred to measured follow-up phases: answer-oriented evidence
compaction, voice answer-length policy, embedding deployment selection, threshold
calibration on the representative evaluation matrix, and live hardware latency
measurement after restarting the backend with this stream-finalization path.
