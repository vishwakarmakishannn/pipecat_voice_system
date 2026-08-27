const validLatency = (value) => (
  typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? value
    : null
);

function percentile(values, quantile) {
  if (!values.length) return null;
  const ordered = [...values].sort((left, right) => left - right);
  const index = (ordered.length - 1) * quantile;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const value = lower === upper
    ? ordered[lower]
    : ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower);
  return Math.round(value * 10) / 10;
}

const LATENCY_CATEGORIES = ['direct', 'rag', 'tool'];

const diagnosticFlag = (value) => (
  typeof value === 'number' && Number.isFinite(value)
    ? value >= 0.5
    : null
);

export function addLatencySample(samples, telemetry) {
  const latencyMs = validLatency(telemetry?.user_stop_to_playback_ms);
  const category = telemetry?.category;
  const isEligible = (
    latencyMs != null
    && telemetry?.measurement_source === 'client'
    && telemetry?.latency_complete === true
    && telemetry?.input_mode === 'voice'
    && LATENCY_CATEGORIES.includes(category)
    && telemetry?.interrupted !== true
    && telemetry?.outcome !== 'cancelled'
  );
  if (!isEligible) return samples;

  const sampleId = `${telemetry.call_id || telemetry.session_id || 'call'}:${telemetry.turn_id}`;
  const finalization = telemetry?.stt_finalization_ms;
  const sample = {
    id: sampleId,
    category,
    latencyMs,
    fallbackForced: diagnosticFlag(finalization?.fallback_forced),
    finalShorter: diagnosticFlag(finalization?.final_shorter_than_interim),
  };
  const existingIndex = samples.findIndex((item) => item.id === sampleId);
  if (existingIndex < 0) return [...samples, sample];
  if (
    samples[existingIndex].latencyMs === sample.latencyMs
    && samples[existingIndex].category === sample.category
    && samples[existingIndex].fallbackForced === sample.fallbackForced
    && samples[existingIndex].finalShorter === sample.finalShorter
  ) return samples;
  return samples.map((item, index) => (index === existingIndex ? sample : item));
}

export function addDirectLatencySample(samples, telemetry) {
  if (telemetry?.category !== 'direct') return samples;
  return addLatencySample(samples, telemetry);
}

function summarizeLatencySamples(samples = []) {
  const values = samples
    .map((sample) => validLatency(sample?.latencyMs))
    .filter((value) => value != null);
  const finalizationSamples = samples.filter(
    (sample) => typeof sample?.fallbackForced === 'boolean',
  );
  const fallbackCount = finalizationSamples.filter(
    (sample) => sample.fallbackForced,
  ).length;
  return {
    count: values.length,
    p50Ms: percentile(values, 0.5),
    p90Ms: percentile(values, 0.9),
    sttFinalizationCount: finalizationSamples.length,
    fallbackCount,
    fallbackRatePct: finalizationSamples.length
      ? Math.round((fallbackCount * 1000) / finalizationSamples.length) / 10
      : null,
    finalShorterCount: finalizationSamples.filter(
      (sample) => sample.finalShorter,
    ).length,
  };
}

export function summarizeDirectLatency(samples = []) {
  return summarizeLatencySamples(
    samples.filter((sample) => sample?.category == null || sample.category === 'direct'),
  );
}

export function summarizeLatencyCohorts(samples = []) {
  return Object.fromEntries(
    LATENCY_CATEGORIES.map((category) => [
      category,
      summarizeLatencySamples(
        samples.filter((sample) => sample?.category === category),
      ),
    ]),
  );
}

export function summarizeLiveLatency(liveLatency) {
  const sttMs = validLatency(liveLatency?.stt_latency_ms);
  const responsePreparationMs = validLatency(
    liveLatency?.response_preparation_ms ?? liveLatency?.llm_latency_ms,
  );
  const modelTtftMs = validLatency(liveLatency?.llm_ttft_ms);
  const ttsMs = validLatency(
    liveLatency?.tts_latency_ms ?? liveLatency?.tts_provider_ms,
  );
  const componentTotalMs = [sttMs, responsePreparationMs, ttsMs]
    .every((value) => value != null)
    ? sttMs + responsePreparationMs + ttsMs
    : null;
  const serverPipelineMs = validLatency(liveLatency?.answer_audio_ms)
    ?? componentTotalMs;
  const perceivedLatencyMs = validLatency(liveLatency?.text_send_to_playback_ms)
    ?? validLatency(liveLatency?.user_stop_to_playback_ms);
  const clientTransportGapMs = (
    perceivedLatencyMs == null
    || serverPipelineMs == null
    || perceivedLatencyMs < serverPipelineMs
  )
    ? null
    : Math.round((perceivedLatencyMs - serverPipelineMs) * 10) / 10;

  return {
    perceivedLatencyMs,
    serverPipelineMs,
    clientTransportGapMs,
    sttMs,
    // Compatibility alias for callers that still consume the old shape.
    llmMs: responsePreparationMs,
    responsePreparationMs,
    modelTtftMs,
    ttsMs,
    endpointingMs: validLatency(liveLatency?.endpointing_ms),
    serverEndpointingMs: validLatency(liveLatency?.server_endpointing_ms),
    turnReleaseMs: validLatency(liveLatency?.turn_release_ms),
  };
}
