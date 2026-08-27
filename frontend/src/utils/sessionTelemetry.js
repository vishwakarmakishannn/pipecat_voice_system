function elapsed(startedAt) {
  return Math.round((performance.now() - startedAt) * 10) / 10;
}

export function createSessionTelemetry(endpoint) {
  const startedAt = performance.now();
  const payload = {
    endpoint,
    started_unix_ms: Date.now(),
    stages_ms: { start_requested: 0 },
    ice: {},
    resource: null,
  };

  return {
    payload,
    mark(stage) {
      if (payload.stages_ms[stage] == null) payload.stages_ms[stage] = elapsed(startedAt);
    },
    markState(group, state) {
      const key = `${group}_${String(state).replace(/[^a-z0-9]+/gi, '_').toLowerCase()}`;
      if (payload.stages_ms[key] == null) payload.stages_ms[key] = elapsed(startedAt);
    },
    finishResourceTiming() {
      const entries = performance.getEntriesByName(endpoint, 'resource');
      const entry = entries.at(-1);
      if (!entry) return;
      const duration = (from, to) => Math.max(0, Math.round((to - from) * 10) / 10);
      payload.resource = {
        dns_ms: duration(entry.domainLookupStart, entry.domainLookupEnd),
        tcp_ms: duration(entry.connectStart, entry.connectEnd),
        tls_ms: entry.secureConnectionStart > 0
          ? duration(entry.secureConnectionStart, entry.connectEnd)
          : null,
        request_ms: duration(entry.requestStart, entry.responseStart),
        response_ms: duration(entry.responseStart, entry.responseEnd),
        total_ms: Math.round(entry.duration * 10) / 10,
        next_hop_protocol: entry.nextHopProtocol || null,
      };
    },
  };
}

export function monitorPeerConnection(transport, telemetry, timeoutMs = 5000) {
  let peerConnection = null;
  let pollTimer = null;
  let timeoutTimer = null;
  const listeners = [];

  const stop = () => {
    if (pollTimer) window.clearInterval(pollTimer);
    if (timeoutTimer) window.clearTimeout(timeoutTimer);
    pollTimer = null;
    timeoutTimer = null;
    for (const [target, event, handler] of listeners) {
      target.removeEventListener(event, handler);
    }
    listeners.length = 0;
  };

  const attach = () => {
    peerConnection = transport?.pc;
    if (!peerConnection) return false;
    telemetry.mark('peer_connection_created');
    const watch = (event, group, property) => {
      const handler = () => telemetry.markState(group, peerConnection[property]);
      peerConnection.addEventListener(event, handler);
      listeners.push([peerConnection, event, handler]);
      handler();
    };
    watch('icegatheringstatechange', 'ice_gathering', 'iceGatheringState');
    watch('iceconnectionstatechange', 'ice_connection', 'iceConnectionState');
    watch('signalingstatechange', 'signaling', 'signalingState');
    watch('connectionstatechange', 'peer_connection', 'connectionState');
    return true;
  };

  if (!attach()) {
    pollTimer = window.setInterval(() => {
      if (attach()) {
        window.clearInterval(pollTimer);
        pollTimer = null;
      }
    }, 10);
  }
  timeoutTimer = window.setTimeout(stop, timeoutMs);
  return stop;
}

export function publishSessionTelemetry(telemetry) {
  telemetry.finishResourceTiming();
  const detail = structuredClone(telemetry.payload);
  window.dispatchEvent(new CustomEvent('aura-voice-session-latency', { detail }));
  console.info('voice_session_latency', detail);
  return detail;
}

export function recordCaptureTrack(track, telemetry) {
  if (!track || !telemetry) return null;
  const settings = track.getSettings?.() || {};
  const capabilities = track.getCapabilities?.() || {};
  const latency = settings.latency;
  telemetry.payload.capture = {
    reported_latency_ms: latency == null ? null : Math.round(latency * 10000) / 10,
    sample_rate: settings.sampleRate ?? null,
    channel_count: settings.channelCount ?? null,
    echo_cancellation: settings.echoCancellation ?? null,
    noise_suppression: settings.noiseSuppression ?? null,
    auto_gain_control: settings.autoGainControl ?? null,
    latency_capability_ms: capabilities.latency
      ? {
          min: Math.round((capabilities.latency.min || 0) * 10000) / 10,
          max: Math.round((capabilities.latency.max || 0) * 10000) / 10,
        }
      : null,
  };
  telemetry.mark('capture_track_started');
  return telemetry.payload.capture;
}

export async function recordSelectedCandidatePair(transport, telemetry) {
  const peerConnection = transport?.pc;
  if (!peerConnection?.getStats) return null;
  const report = await peerConnection.getStats();
  let selectedPair = null;
  report.forEach((stat) => {
    if (stat.type === 'candidate-pair' && (stat.selected || stat.nominated && stat.state === 'succeeded')) {
      selectedPair = stat;
    }
  });
  if (!selectedPair) return null;
  const local = report.get(selectedPair.localCandidateId);
  const remote = report.get(selectedPair.remoteCandidateId);
  telemetry.payload.ice = {
    local_candidate_type: local?.candidateType || null,
    local_protocol: local?.protocol || null,
    remote_candidate_type: remote?.candidateType || null,
    remote_protocol: remote?.protocol || null,
    current_rtt_ms: selectedPair.currentRoundTripTime == null
      ? null
      : Math.round(selectedPair.currentRoundTripTime * 10000) / 10,
  };
  telemetry.mark('selected_candidate_pair_recorded');
  return telemetry.payload.ice;
}

export async function collectWebRTCAudioStats(transport) {
  const peerConnection = transport?.pc;
  if (!peerConnection?.getStats) return null;
  const report = await peerConnection.getStats();
  let inbound = null;
  let pair = null;
  report.forEach((stat) => {
    if (stat.type === 'inbound-rtp' && stat.kind === 'audio' && !stat.isRemote) inbound = stat;
    if (stat.type === 'candidate-pair' && (stat.selected || stat.nominated && stat.state === 'succeeded')) pair = stat;
  });
  if (!inbound) return null;
  const emitted = inbound.jitterBufferEmittedCount || 0;
  return {
    webrtc_jitter_ms: inbound.jitter == null ? null : Math.round(inbound.jitter * 10000) / 10,
    jitter_buffer_avg_ms: emitted > 0 && inbound.jitterBufferDelay != null
      ? Math.round((inbound.jitterBufferDelay / emitted) * 10000) / 10
      : null,
    packets_lost: inbound.packetsLost ?? null,
    packets_received: inbound.packetsReceived ?? null,
    concealed_samples: inbound.concealedSamples ?? null,
    concealment_events: inbound.concealmentEvents ?? null,
    rtt_ms: pair?.currentRoundTripTime == null
      ? null
      : Math.round(pair.currentRoundTripTime * 10000) / 10,
  };
}

export function monitorRemoteAudioTrack(track, onLevel, intervalMs = 16) {
  if (!track || track.kind !== 'audio') return () => {};
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass || typeof MediaStream === 'undefined') return () => {};

  const context = new AudioContextClass({ latencyHint: 'interactive' });
  const source = context.createMediaStreamSource(new MediaStream([track]));
  const analyser = context.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0;
  source.connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  let stopped = false;
  const timer = window.setInterval(() => {
    if (stopped || track.readyState === 'ended') return;
    analyser.getFloatTimeDomainData(samples);
    let energy = 0;
    for (const sample of samples) energy += sample * sample;
    onLevel(Math.sqrt(energy / samples.length));
  }, intervalMs);

  void context.resume?.().catch(() => undefined);
  const stop = () => {
    if (stopped) return;
    stopped = true;
    window.clearInterval(timer);
    track.removeEventListener?.('ended', stop);
    source.disconnect?.();
    analyser.disconnect?.();
    void context.close?.().catch(() => undefined);
  };
  track.addEventListener?.('ended', stop, { once: true });
  return stop;
}

export async function withConnectionDeadline(connectPromise, timeoutMs, onTimeout) {
  let timer;
  try {
    return await Promise.race([
      connectPromise,
      new Promise((_, reject) => {
        timer = window.setTimeout(() => reject(new Error(`WebRTC connection timed out after ${timeoutMs} ms`)), timeoutMs);
      }),
    ]);
  } catch (error) {
    if (String(error?.message || '').includes('WebRTC connection timed out')) await onTimeout?.();
    throw error;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

function audioTrackFrom(element) {
  return element?.srcObject?.getAudioTracks?.()[0] || null;
}

function nextAnimationFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(resolve));
}

async function waitForAudioTrack(element, expectedTrack, timeoutMs) {
  const deadline = performance.now() + timeoutMs;
  while (expectedTrack?.readyState !== 'ended') {
    if (audioTrackFrom(element)?.id === expectedTrack?.id) return true;
    if (performance.now() >= deadline) return false;
    await nextAnimationFrame();
  }
  return false;
}

export async function ensureBotAudioPlayback(expectedTrack, timeoutMs = 1000) {
  const element = document.querySelector('audio');
  if (!element) return false;

  if (expectedTrack) {
    const attached = await waitForAudioTrack(element, expectedTrack, timeoutMs);
    if (!attached) return false;
  } else if (!audioTrackFrom(element)) {
    return false;
  }

  element.muted = false;
  element.volume = 1;

  // React attaches the remote MediaStream in an effect after TrackStarted.
  // If that source changes while play() is pending, Chromium rejects with an
  // AbortError. Retry against the same live track; a replaced/ended track is a
  // stale playback request and should not be reported as an autoplay failure.
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      await element.play();
      return true;
    } catch (error) {
      if (
        expectedTrack
        && (
          expectedTrack.readyState === 'ended'
          || audioTrackFrom(element)?.id !== expectedTrack.id
        )
      ) {
        return false;
      }
      if (error?.name !== 'AbortError') throw error;
      await nextAnimationFrame();
    }
  }
  return false;
}
