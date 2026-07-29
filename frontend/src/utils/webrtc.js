function splitUrls(value) {
  return String(value || '')
    .split(',')
    .map((url) => url.trim())
    .filter(Boolean);
}

export function buildIceServers(env = import.meta.env) {
  const servers = [];
  const stunUrls = splitUrls(env?.VITE_STUN_URLS || 'stun:stun.l.google.com:19302');
  if (stunUrls.length) servers.push({ urls: stunUrls });

  const turnUrls = splitUrls(env?.VITE_TURN_URLS);
  if (turnUrls.length) {
    if (!env?.VITE_TURN_USERNAME || !env?.VITE_TURN_CREDENTIAL) {
      throw new Error('TURN URLs require VITE_TURN_USERNAME and VITE_TURN_CREDENTIAL');
    }
    servers.push({
      urls: turnUrls,
      username: env.VITE_TURN_USERNAME,
      credential: env.VITE_TURN_CREDENTIAL,
    });
  }
  return servers;
}

function envBoolean(value, fallback) {
  if (value == null || value === '') return fallback;
  return !['0', 'false', 'off', 'no'].includes(String(value).toLowerCase());
}

export function buildAudioConstraints(env = import.meta.env) {
  return {
    echoCancellation: envBoolean(env?.VITE_AUDIO_ECHO_CANCELLATION, true),
    noiseSuppression: envBoolean(env?.VITE_AUDIO_NOISE_SUPPRESSION, true),
    autoGainControl: envBoolean(env?.VITE_AUDIO_AUTO_GAIN_CONTROL, true),
    channelCount: { ideal: 1 },
    sampleRate: { ideal: 48000 },
  };
}

export function localSpeechLevelThreshold(env = import.meta.env) {
  const raw = env?.VITE_LOCAL_SPEECH_LEVEL_THRESHOLD ?? '0.01';
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error('VITE_LOCAL_SPEECH_LEVEL_THRESHOLD must be between 0 and 1');
  }
  return value;
}

export function webRTCConnectTimeoutMs(env = import.meta.env) {
  const raw = env?.VITE_WEBRTC_CONNECT_TIMEOUT_MS ?? '8000';
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 1000 || value > 30000) {
    throw new Error('VITE_WEBRTC_CONNECT_TIMEOUT_MS must be between 1000 and 30000');
  }
  return value;
}
