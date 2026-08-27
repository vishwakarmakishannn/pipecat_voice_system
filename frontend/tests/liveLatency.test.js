import assert from 'node:assert/strict';
import test from 'node:test';

import {
  addLatencySample,
  addDirectLatencySample,
  summarizeDirectLatency,
  summarizeLatencyCohorts,
  summarizeLiveLatency,
} from '../src/utils/liveLatency.js';

test('live latency separates perceived playback from the server pipeline', () => {
  const summary = summarizeLiveLatency({
    stt_latency_ms: 253,
    llm_latency_ms: 228.2,
    llm_ttft_ms: 127.4,
    tts_latency_ms: 242.1,
    answer_audio_ms: 723.3,
    user_stop_to_playback_ms: 1037.2,
    endpointing_ms: 410.5,
    turn_release_ms: 31.2,
  });

  assert.deepEqual(summary, {
    perceivedLatencyMs: 1037.2,
    serverPipelineMs: 723.3,
    clientTransportGapMs: 313.9,
    sttMs: 253,
    llmMs: 228.2,
    responsePreparationMs: 228.2,
    modelTtftMs: 127.4,
    ttsMs: 242.1,
    endpointingMs: 410.5,
    serverEndpointingMs: null,
    turnReleaseMs: 31.2,
  });
});

test('server pipeline falls back to the complete stage sum', () => {
  const summary = summarizeLiveLatency({
    stt_latency_ms: 253,
    response_preparation_ms: 228,
    tts_provider_ms: 242,
  });

  assert.equal(summary.perceivedLatencyMs, null);
  assert.equal(summary.serverPipelineMs, 723);
  assert.equal(summary.clientTransportGapMs, null);
  assert.equal(summary.modelTtftMs, null);
});

test('text playback is preferred and invalid metrics are ignored', () => {
  const summary = summarizeLiveLatency({
    text_send_to_playback_ms: 800,
    user_stop_to_playback_ms: 900,
    answer_audio_ms: 500,
    stt_latency_ms: Number.NaN,
    llm_latency_ms: -1,
    tts_latency_ms: 200,
  });

  assert.equal(summary.perceivedLatencyMs, 800);
  assert.equal(summary.serverPipelineMs, 500);
  assert.equal(summary.clientTransportGapMs, 300);
  assert.equal(summary.sttMs, null);
  assert.equal(summary.llmMs, null);
  assert.equal(summary.ttsMs, 200);
  assert.equal(summary.responsePreparationMs, null);
});

test('an incompatible client/server timing pair does not report a false gap', () => {
  const summary = summarizeLiveLatency({
    user_stop_to_playback_ms: 450,
    answer_audio_ms: 500,
  });

  assert.equal(summary.clientTransportGapMs, null);
});

test('direct perceived percentiles use one completed browser sample per turn', () => {
  const base = {
    call_id: 'call-a',
    measurement_source: 'client',
    latency_complete: true,
    input_mode: 'voice',
    category: 'direct',
    interrupted: false,
    outcome: 'completed',
  };
  let samples = [];
  for (const [turnId, latencyMs] of [[1, 800], [2, 1000], [3, 1200]]) {
    samples = addDirectLatencySample(samples, {
      ...base,
      turn_id: turnId,
      user_stop_to_playback_ms: latencyMs,
    });
  }
  samples = addDirectLatencySample(samples, {
    ...base,
    turn_id: 3,
    user_stop_to_playback_ms: 1200,
  });

  assert.deepEqual(summarizeDirectLatency(samples), {
    count: 3,
    p50Ms: 1000,
    p90Ms: 1160,
    sttFinalizationCount: 0,
    fallbackCount: 0,
    fallbackRatePct: null,
    finalShorterCount: 0,
  });
});

test('latency cohorts separate direct, RAG, and tool turns with STT fallback rates', () => {
  const base = {
    call_id: 'call-a',
    measurement_source: 'client',
    latency_complete: true,
    input_mode: 'voice',
    interrupted: false,
    outcome: 'completed',
  };
  let samples = [];
  samples = addLatencySample(samples, {
    ...base,
    turn_id: 1,
    category: 'direct',
    user_stop_to_playback_ms: 800,
    stt_finalization_ms: { fallback_forced: 0 },
  });
  samples = addLatencySample(samples, {
    ...base,
    turn_id: 2,
    category: 'rag',
    user_stop_to_playback_ms: 1600,
    stt_finalization_ms: {
      fallback_forced: 1,
      final_shorter_than_interim: 1,
    },
  });
  samples = addLatencySample(samples, {
    ...base,
    turn_id: 3,
    category: 'tool',
    user_stop_to_playback_ms: 4000,
    stt_finalization_ms: { fallback_forced: 0 },
  });

  const cohorts = summarizeLatencyCohorts(samples);
  assert.equal(cohorts.direct.p50Ms, 800);
  assert.equal(cohorts.rag.p50Ms, 1600);
  assert.equal(cohorts.rag.fallbackRatePct, 100);
  assert.equal(cohorts.rag.finalShorterCount, 1);
  assert.equal(cohorts.tool.p90Ms, 4000);
});

test('direct percentiles exclude tools, text, cancelled, and interrupted turns', () => {
  const base = {
    call_id: 'call-a',
    turn_id: 1,
    measurement_source: 'client',
    latency_complete: true,
    input_mode: 'voice',
    category: 'direct',
    interrupted: false,
    outcome: 'completed',
    user_stop_to_playback_ms: 900,
  };

  for (const override of [
    { category: 'tool' },
    { category: 'rag' },
    { input_mode: 'text' },
    { outcome: 'cancelled' },
    { interrupted: true },
    { measurement_source: 'server' },
  ]) {
    assert.deepEqual(addDirectLatencySample([], { ...base, ...override }), []);
  }
});
