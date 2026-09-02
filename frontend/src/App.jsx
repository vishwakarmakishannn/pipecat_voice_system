import React, { lazy, Suspense, useCallback, useMemo, useState } from 'react';
import { RTVIEvent, PipecatClient } from '@pipecat-ai/client-js';
import {
  PipecatClientAudio,
  PipecatClientProvider,
  usePipecatClient,
  usePipecatClientMicControl,
  usePipecatClientTransportState,
  useRTVIClientEvent,
} from '@pipecat-ai/client-react';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import { ExternalLink, Mic, Moon, Send, Sun, Volume2, X } from 'lucide-react';
import { jwtDecode } from 'jwt-decode';
import { BrowserRouter, useLocation, useNavigate } from 'react-router-dom';
import { fetchWithAuth, API_BASE } from './utils/api';
import { buildAudioConstraints, buildIceServers, localSpeechLevelThreshold, webRTCConnectTimeoutMs } from './utils/webrtc';
import { collectWebRTCAudioStats, createSessionTelemetry, ensureBotAudioPlayback, monitorPeerConnection, monitorRemoteAudioTrack, publishSessionTelemetry, recordCaptureTrack, recordSelectedCandidatePair, withConnectionDeadline } from './utils/sessionTelemetry';
import { addLatencySample } from './utils/liveLatency';
import './App.css';
import { CallDetailPage, CallsPage } from './components/CallPages';
import { MemoriesPage } from './components/ResourcePages';

const START_ENDPOINT =
  import.meta.env.VITE_PIPECAT_START_URL ||
  `${API_BASE}/start`;

const MAX_TRANSCRIPT_ITEMS = (() => {
  const value = Number(import.meta.env.VITE_MAX_TRANSCRIPT_ITEMS || 250);
  return Number.isInteger(value) && value >= 50 && value <= 2000 ? value : 250;
})();

function capTranscriptItems(items) {
  return items.length > MAX_TRANSCRIPT_ITEMS
    ? items.slice(items.length - MAX_TRANSCRIPT_ITEMS)
    : items;
}

const Auth = lazy(() => import('./components/Auth'));
const Sidebar = lazy(() => import('./components/Sidebar'));
const TranscriptPanel = lazy(() => import('./components/TranscriptPanel'));

function createPipecatClient(iceServers = buildIceServers()) {
  return new PipecatClient({
    transport: new SmallWebRTCTransport({
      iceServers,
      waitForICEGathering: false,
    }),
    enableMic: true,
    enableCam: false,
  });
}

function isValidTokenValue(token) {
  if (!token) return false;
  try {
    const decoded = jwtDecode(token);
    return decoded.exp * 1000 > Date.now();
  } catch {
    return false;
  }
}

function VoiceApp({ onResetClient }) {
  const location = useLocation();
  const navigate = useNavigate();
  const pcClient = usePipecatClient();
  const transportState = usePipecatClientTransportState();
  const { enableMic, isMicEnabled } = usePipecatClientMicControl();
  
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState('');
  
  const [transcripts, setTranscripts] = useState([]);
  const [currentCallId, setCurrentCallId] = useState(null);
  const [currentCallStatus, setCurrentCallStatus] = useState(null);
  const [currentRecordingStatus, setCurrentRecordingStatus] = useState(null);
  const [currentProviders, setCurrentProviders] = useState(null);
  const [currentCallStartedAt, setCurrentCallStartedAt] = useState(null);
  const [currentCallElapsed, setCurrentCallElapsed] = useState(0);

  const currentCallIdRef = React.useRef(null);
  const transcriptAreaRef = React.useRef(null);
  const transcriptBottomRef = React.useRef(null);
  const shouldAutoScrollRef = React.useRef(true);
  
  const botTextRef = React.useRef('');
  const pendingBotTextRef = React.useRef('');
  const botTextFlushTimerRef = React.useRef(null);
  const activeBotMessageIdRef = React.useRef(null);
  const toolCallPayloadsRef = React.useRef({});
  const voiceTurnTimingRef = React.useRef({
    inputMode: null,
    textSubmittedAt: null,
    speechStartedAt: null,
    speechStoppedAt: null,
    turnStopSignalAt: null,
    lastLocalSpeechAt: null,
    localSpeechObserved: false,
    botTtsStartedAt: null,
    firstRemoteAudioAt: null,
    awaitingRemoteAudio: false,
  });
  const pendingLatencyRef = React.useRef(null);
  const sessionTelemetryRef = React.useRef(null);
  const stopPeerMonitorRef = React.useRef(null);
  const stopRemoteAudioMonitorRef = React.useRef(null);
  
  const [expandedToolCalls, setExpandedToolCalls] = useState({});
  const [liveLatency, setLiveLatency] = useState(null);
  const [latencySamples, setLatencySamples] = useState([]);
  const [isVoiceBusy, setIsVoiceBusy] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [isSendingText, setIsSendingText] = useState(false);
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('aura_theme');
    if (savedTheme === 'light' || savedTheme === 'dark') return savedTheme;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  const reportClientEvent = useCallback((code, message, details = {}) => {
    const callId = currentCallIdRef.current;
    if (!callId) return;
    void fetchWithAuth(`/api/calls/${callId}/client-events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, message, severity: 'error', details }),
      timeoutMs: 2500,
    }).catch((reportError) => console.warn('Could not persist client diagnostic', reportError));
  }, []);

  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    localStorage.setItem('aura_theme', theme);
  }, [theme]);

  const toggleToolCall = (id) => {
    setExpandedToolCalls(prev => ({...prev, [id]: !prev[id]}));
  };

  const isActive = ['connected', 'ready'].includes(transportState);
  const canStart = ['disconnected', 'initialized', 'error'].includes(transportState);

  const statusLabel = useMemo(() => {
    if (isConnecting) return 'Connecting';
    if (transportState === 'ready') return 'Connected';
    return transportState.charAt(0).toUpperCase() + transportState.slice(1);
  }, [isConnecting, transportState]);

  React.useEffect(() => {
    if (location.pathname === '/') navigate('/playground', { replace: true });
  }, [location.pathname, navigate]);

  React.useEffect(() => {
    currentCallIdRef.current = currentCallId;
  }, [currentCallId]);

  React.useEffect(() => {
    if (!currentCallStartedAt || currentCallStatus !== 'active') return undefined;
    const update = () => setCurrentCallElapsed(Math.max(0, Math.floor((Date.now() - Date.parse(currentCallStartedAt)) / 1000)));
    const timer = window.setInterval(update, 1000);
    const initial = window.setTimeout(update, 0);
    return () => { window.clearInterval(timer); window.clearTimeout(initial); };
  }, [currentCallStartedAt, currentCallStatus]);

  React.useEffect(() => {
    if (!currentCallId || !['disconnected', 'initialized', 'error'].includes(transportState)) {
      return undefined;
    }
    let cancelled = false;
    let timer;
    const refreshTerminalState = async () => {
      try {
        const response = await fetchWithAuth(`/api/calls/${currentCallId}`);
        if (!response.ok || cancelled) return;
        const call = await response.json();
        setCurrentCallStatus(call.status);
        if (call.duration_ms != null) {
          setCurrentCallElapsed(Math.max(0, Math.floor(call.duration_ms / 1000)));
        }
        setCurrentRecordingStatus(call.recording?.status || null);
        setCurrentProviders(call.providers || null);
        if (!['completed', 'failed', 'cancelled', 'abandoned'].includes(call.status)) {
          timer = window.setTimeout(refreshTerminalState, 750);
        }
      } catch {
        if (!cancelled) timer = window.setTimeout(refreshTerminalState, 1000);
      }
    };
    timer = window.setTimeout(refreshTerminalState, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [currentCallId, transportState]);

  React.useEffect(() => {
    if (!shouldAutoScrollRef.current) return;
    const frameId = window.requestAnimationFrame(() => {
      const area = transcriptAreaRef.current;
      if (area) area.scrollTo({ top: area.scrollHeight, behavior: 'auto' });
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [transcripts]);

  const handleTranscriptScroll = useCallback(() => {
    const area = transcriptAreaRef.current;
    if (!area) return;
    const distanceFromBottom = area.scrollHeight - area.scrollTop - area.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom < 80;
  }, []);

  const startNewCall = async () => {
    if (isActive) await disconnect();
    currentCallIdRef.current = null;
    setCurrentCallId(null);
    setCurrentCallStatus(null);
    setCurrentRecordingStatus(null);
    setCurrentProviders(null);
    setCurrentCallStartedAt(null);
    setCurrentCallElapsed(0);
    setTranscripts([]);
    setTextInput('');
    shouldAutoScrollRef.current = true;
    setLiveLatency(null);
    setLatencySamples([]);
  };

  const addTranscript = useCallback((role, text, isDelta = false, messageId = null) => {
    if (!text) return;
    if (role === 'You') shouldAutoScrollRef.current = true;
    
    setTranscripts((items) => {
      const existingIndex = messageId ? items.findIndex(item => item.id === messageId) : -1;
      if (existingIndex !== -1) {
        const updated = [...items];
        updated[existingIndex] = {
          ...updated[existingIndex],
          text: isDelta ? updated[existingIndex].text + text : text,
        };
        return updated;
      }
      
      return capTranscriptItems([
        ...items,
        {
          id: messageId || `${Date.now()}-${items.length}`,
          role,
          text,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        },
      ]);
    });
  }, []);

  const upsertToolCall = useCallback((data) => {
    const toolCallId = data.tool_call_id || `tool-${Date.now()}`;
    const previousPayload = toolCallPayloadsRef.current[toolCallId] || {};
    const payload = {
      ...previousPayload,
      tool_call_id: toolCallId,
      function_name: data.function_name || previousPayload.function_name,
      arguments: data.arguments || data.args || previousPayload.arguments,
      result: Object.prototype.hasOwnProperty.call(data, 'result')
        ? data.result
        : previousPayload.result,
      status: data.status || previousPayload.status,
    };
    toolCallPayloadsRef.current[toolCallId] = payload;

    setTranscripts((items) => {
      const existingIdx = items.findIndex((item) => item.id === toolCallId);
      if (existingIdx !== -1) {
        const updated = [...items];
        updated[existingIdx] = {
          ...updated[existingIdx],
          text: JSON.stringify(payload),
        };
        return updated;
      }
      return capTranscriptItems([
        ...items,
        {
          id: toolCallId,
          role: 'ToolCall',
          text: JSON.stringify(payload),
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        },
      ]);
    });
  }, []);

  const flushBotText = useCallback(() => {
    if (botTextFlushTimerRef.current) {
      window.clearTimeout(botTextFlushTimerRef.current);
      botTextFlushTimerRef.current = null;
    }
    const text = pendingBotTextRef.current;
    pendingBotTextRef.current = '';
    if (!text) return;
    if (!activeBotMessageIdRef.current) {
      activeBotMessageIdRef.current = `bot-${Date.now()}`;
    }
    addTranscript('Aura', text, true, activeBotMessageIdRef.current);
  }, [addTranscript]);

  const scheduleBotTextFlush = useCallback(() => {
    if (botTextFlushTimerRef.current) return;
    botTextFlushTimerRef.current = window.setTimeout(flushBotText, 40);
  }, [flushBotText]);

  React.useEffect(() => () => {
    if (botTextFlushTimerRef.current) {
      window.clearTimeout(botTextFlushTimerRef.current);
    }
  }, []);

  const commitClientLatency = useCallback(async () => {
    const pending = pendingLatencyRef.current;
    const timing = voiceTurnTimingRef.current;
    if (
      !pending
      || !pending.serverPayload?.latency_complete
      || timing.firstRemoteAudioAt == null
    ) return;

    const clientMetrics = {
      client_message_to_audio_ms: Math.max(
        0,
        Math.round(
          (timing.firstRemoteAudioAt - pending.receivedPerfAt) * 10,
        ) / 10,
      ),
      user_stop_to_playback_ms: timing.inputMode === 'text' || timing.speechStoppedAt == null
        ? null
        : Math.round((timing.firstRemoteAudioAt - timing.speechStoppedAt) * 10) / 10,
      text_send_to_playback_ms: timing.inputMode === 'text' && timing.textSubmittedAt != null
        ? Math.round((timing.firstRemoteAudioAt - timing.textSubmittedAt) * 10) / 10
        : null,
      turn_stop_signal_to_playback_ms: timing.turnStopSignalAt == null
        ? null
        : Math.round((timing.firstRemoteAudioAt - timing.turnStopSignalAt) * 10) / 10,
      endpointing_ms: timing.turnStopSignalAt == null || !timing.localSpeechObserved
        ? null
        : Math.max(0, Math.round((timing.turnStopSignalAt - timing.speechStoppedAt) * 10) / 10),
      client_speech_ms: timing.speechStartedAt == null || timing.speechStoppedAt == null
        ? null
        : Math.round((timing.speechStoppedAt - timing.speechStartedAt) * 10) / 10,
      tts_signal_to_playback_ms: timing.botTtsStartedAt == null
        ? null
        : Math.round((timing.firstRemoteAudioAt - timing.botTtsStartedAt) * 10) / 10,
      playback_detected_unix_ms: Date.now(),
      playback_signal: 'first_nonzero_remote_audio_level',
      speech_end_signal: timing.inputMode === 'text'
        ? 'text_submitted'
        : timing.localSpeechObserved
          ? 'last_nonzero_local_audio_level'
          : 'server_turn_stop_event',
    };

    // Capture the media timestamp and release the turn immediately. Browser
    // getStats() can be slow or hang during ICE changes, so it gets only a
    // small out-of-band enrichment window.
    if (pendingLatencyRef.current === pending) pendingLatencyRef.current = null;
    const webrtcMetrics = await Promise.race([
      collectWebRTCAudioStats(pcClient?.transport).catch(() => null),
      new Promise((resolve) => window.setTimeout(() => resolve(null), 150)),
    ]);
    const completeMetrics = {
      ...(pending.serverPayload || {}),
      measurement_source: 'client',
      ...clientMetrics,
      ...(webrtcMetrics || {}),
      capture_reported_latency_ms:
        sessionTelemetryRef.current?.payload.capture?.reported_latency_ms ?? null,
      capture_sample_rate:
        sessionTelemetryRef.current?.payload.capture?.sample_rate ?? null,
      capture_channel_count:
        sessionTelemetryRef.current?.payload.capture?.channel_count ?? null,
      capture_echo_cancellation:
        sessionTelemetryRef.current?.payload.capture?.echo_cancellation ?? null,
      capture_noise_suppression:
        sessionTelemetryRef.current?.payload.capture?.noise_suppression ?? null,
      capture_auto_gain_control:
        sessionTelemetryRef.current?.payload.capture?.auto_gain_control ?? null,
    };
    setLiveLatency((current) => (
      current?.turn_id === pending.turnId
        ? { ...current, ...completeMetrics }
        : current
    ));
    setLatencySamples((samples) => (
      addLatencySample(samples, completeMetrics)
    ));
    // Playback has already begun. Persist metrics out-of-band so telemetry
    // cannot delay media or the next user turn.
    void fetchWithAuth('/api/telemetry/voice-latency', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(completeMetrics),
      timeoutMs: 2500,
    }).catch((telemetryError) => {
      console.warn('Could not persist voice latency telemetry', telemetryError);
    });
  }, [pcClient]);

  const recordRemoteAudioLevel = useCallback((level) => {
    const timing = voiceTurnTimingRef.current;
    if (!timing.awaitingRemoteAudio || Number(level) <= 0.0001) return;
    timing.awaitingRemoteAudio = false;
    timing.firstRemoteAudioAt = performance.now();
    commitClientLatency();
  }, [commitClientLatency]);

  useRTVIClientEvent(
    RTVIEvent.TransportStateChanged,
    useCallback((state) => {
      sessionTelemetryRef.current?.markState('transport', state);
      if (state === 'ready' || state === 'connected' || state === 'error') {
        setIsConnecting(false);
      }
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.TrackStarted,
    useCallback((track, participant) => {
      if (track?.kind !== 'audio') return;
      if (participant?.local && track.applyConstraints) {
        track.applyConstraints(buildAudioConstraints()).catch((constraintError) => {
          console.warn('Could not apply microphone DSP constraints', constraintError);
          reportClientEvent(
            'transport.microphone_failed',
            'The browser could not apply the requested microphone constraints.',
            { error_name: constraintError?.name || 'ConstraintError' },
          );
        }).finally(() => {
          recordCaptureTrack(track, sessionTelemetryRef.current);
        });
        return;
      }
      if (!participant?.local) {
        stopRemoteAudioMonitorRef.current?.();
        stopRemoteAudioMonitorRef.current = monitorRemoteAudioTrack(
          track,
          recordRemoteAudioLevel,
        );
        window.setTimeout(() => {
          ensureBotAudioPlayback(track).catch((playbackError) => {
            const message = playbackError?.message || playbackError;
            setError(
              playbackError?.name === 'NotAllowedError'
                ? `Browser blocked bot audio playback: ${message}`
                : `Could not start bot audio playback: ${message}`,
            );
            reportClientEvent(
              'transport.audio_playback_failed',
              'The browser could not start agent audio playback.',
              { error_name: playbackError?.name || 'PlaybackError' },
            );
          });
        }, 0);
      }
    }, [recordRemoteAudioLevel, reportClientEvent]),
  );

  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback((data) => {
      if (data.final) addTranscript('You', data.text, false);
    }, [addTranscript]),
  );

  useRTVIClientEvent(
    RTVIEvent.UserStartedSpeaking,
    useCallback(() => {
      setIsVoiceBusy(true);
      const timing = voiceTurnTimingRef.current;
      if (timing.speechStartedAt == null || timing.firstRemoteAudioAt != null) {
        voiceTurnTimingRef.current = {
          inputMode: 'voice',
          textSubmittedAt: null,
          speechStartedAt: performance.now(),
          speechStoppedAt: null,
          turnStopSignalAt: null,
          lastLocalSpeechAt: null,
          localSpeechObserved: false,
          botTtsStartedAt: null,
          firstRemoteAudioAt: null,
          awaitingRemoteAudio: false,
        };
        pendingLatencyRef.current = null;
      }
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.LocalAudioLevel,
    useCallback((level) => {
      const timing = voiceTurnTimingRef.current;
      if (timing.speechStartedAt == null || timing.turnStopSignalAt != null) return;
      if (Number(level) <= localSpeechLevelThreshold()) return;
      timing.lastLocalSpeechAt = performance.now();
      timing.localSpeechObserved = true;
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.UserStoppedSpeaking,
    useCallback(() => {
      const timing = voiceTurnTimingRef.current;
      const receivedAt = performance.now();
      timing.turnStopSignalAt = receivedAt;
      timing.speechStoppedAt = timing.localSpeechObserved
        ? timing.lastLocalSpeechAt
        : receivedAt;
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.BotTtsStarted,
    useCallback(() => {
      setIsVoiceBusy(true);
      const timing = voiceTurnTimingRef.current;
      timing.botTtsStartedAt = performance.now();
      timing.awaitingRemoteAudio = true;
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.BotTtsStopped,
    useCallback(() => {
      setIsVoiceBusy(false);
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.RemoteAudioLevel,
    recordRemoteAudioLevel,
  );

  React.useEffect(() => () => {
    stopRemoteAudioMonitorRef.current?.();
    stopRemoteAudioMonitorRef.current = null;
  }, []);

  useRTVIClientEvent(
    RTVIEvent.BotLlmStarted,
    useCallback(() => {
      if (botTextFlushTimerRef.current) {
        window.clearTimeout(botTextFlushTimerRef.current);
        botTextFlushTimerRef.current = null;
      }
      botTextRef.current = '';
      pendingBotTextRef.current = '';
      activeBotMessageIdRef.current = `bot-${Date.now()}`;
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.LLMFunctionCallInProgress,
    useCallback((data) => upsertToolCall({ ...data, status: 'in_progress' }), [upsertToolCall]),
  );

  useRTVIClientEvent(
    RTVIEvent.LLMFunctionCallStopped,
    useCallback((data) => upsertToolCall({
      ...data,
      status: data.cancelled ? 'cancelled' : 'completed',
    }), [upsertToolCall]),
  );

  useRTVIClientEvent(
    RTVIEvent.BotLlmText,
    useCallback((data) => {
      botTextRef.current += data.text;
      pendingBotTextRef.current += data.text;
      scheduleBotTextFlush();
    }, [scheduleBotTextFlush]),
  );

  useRTVIClientEvent(
    RTVIEvent.BotLlmStopped,
    useCallback(() => {
      flushBotText();
      activeBotMessageIdRef.current = null;
    }, [flushBotText]),
  );

  useRTVIClientEvent(
    RTVIEvent.Error,
    useCallback((message) => {
      setError(message?.data?.error || message?.data?.message || 'Pipecat connection failed');
      setIsConnecting(false);
      reportClientEvent(
        'transport.connection_lost',
        'The realtime voice transport reported an error.',
        { transport_state: transportState },
      );
    }, [reportClientEvent, transportState]),
  );

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((data) => {
      const messageData = data?.data || data;
      if (messageData?.type === 'call_ready' && messageData.payload?.call_id) {
        const callId = messageData.payload.call_id;
        currentCallIdRef.current = callId;
        setCurrentCallId(callId);
        setCurrentCallStatus(messageData.payload.status || 'active');
        setCurrentProviders(messageData.payload.providers || null);
        setCurrentCallStartedAt(messageData.payload.started_at || new Date().toISOString());
        return;
      }
      if (messageData?.type === 'call_status' && messageData.payload) {
        setCurrentCallStatus(messageData.payload.status);
        return;
      }
      if (messageData?.type === 'recording_status' && messageData.payload) {
        setCurrentRecordingStatus(messageData.payload.status);
        return;
      }
      if (messageData?.type === 'timeline_item' && messageData.payload) {
        const item = messageData.payload;
        if (item.component === 'tool' && item.request_id) {
          upsertToolCall({
            tool_call_id: item.request_id,
            status: item.outcome === 'failed' ? 'failed' : 'error',
          });
          return;
        }
        setTranscripts((items) => capTranscriptItems([...items, {
          id: item.fingerprint || `event-${Date.now()}`,
          role: 'Error',
          text: JSON.stringify(item),
          timestamp: new Date(item.created_at || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        }]));
        return;
      }
      if (messageData?.type === 'turn_metrics' && messageData.payload) {
        const receivedPerfAt = performance.now();
        const timing = voiceTurnTimingRef.current;
        const payload = messageData.payload;
        const previousPending = pendingLatencyRef.current;
        const sameTurn = previousPending?.turnId === payload.turn_id;
        const serverPayload = sameTurn
          ? { ...(previousPending.serverPayload || {}), ...payload }
          : payload;
        // Only the completed server snapshot corresponds to first generated
        // audio. Earlier STT/LLM snapshots progressively fill the breakdown.
        const textSendToAudioMs = (
          payload.latency_complete
          && timing.inputMode === 'text'
          && timing.textSubmittedAt != null
        )
          ? Math.max(0, Math.round((receivedPerfAt - timing.textSubmittedAt) * 10) / 10)
          : null;
        pendingLatencyRef.current = {
          turnId: payload.turn_id,
          receivedPerfAt: payload.latency_complete
            ? receivedPerfAt
            : sameTurn
              ? previousPending.receivedPerfAt
              : receivedPerfAt,
          serverPayload,
        };
        const latencyUpdate = {
          ...serverPayload,
          ...(textSendToAudioMs == null ? {} : {
            text_send_to_playback_ms: textSendToAudioMs,
            speech_end_signal: 'text_submitted',
            playback_signal: 'first_audio_server_signal',
          }),
          client_received_unix_ms: Date.now(),
        };
        setLiveLatency((current) => (
          current?.turn_id === payload.turn_id
            ? { ...current, ...latencyUpdate }
            : latencyUpdate
        ));
        commitClientLatency();
        return;
      }
      if (messageData?.type === 'assistant_transcript' && messageData.payload?.text) {
        addTranscript(
          'Aura',
          messageData.payload.text,
          Boolean(messageData.payload.delta),
          messageData.payload.id || null,
        );
        return;
      }
      if (messageData?.type === 'tool_call' && messageData.payload) {
        upsertToolCall(messageData.payload);
        return;
      }
      return;
    }, [addTranscript, commitClientLatency, upsertToolCall]),
  );

  const startCall = async () => {
    if (!pcClient || isConnecting || !canStart) return false;

    setError('');
    setIsConnecting(true);
    setLiveLatency(null);
    setLatencySamples([]);
    const telemetry = createSessionTelemetry(START_ENDPOINT);
    sessionTelemetryRef.current = telemetry;
    stopPeerMonitorRef.current?.();
    stopPeerMonitorRef.current = monitorPeerConnection(pcClient.transport, telemetry);

    try {
      const connectPromise = pcClient.startBotAndConnect({
        endpoint: START_ENDPOINT,
        requestData: {
          transport: 'webrtc',
          body: {
            token: localStorage.getItem('aura_token'),
          },
        },
      });
      await withConnectionDeadline(
        connectPromise,
        webRTCConnectTimeoutMs(),
        () => pcClient.disconnect(),
      );
      telemetry.mark('connect_promise_resolved');
      await recordSelectedCandidatePair(pcClient.transport, telemetry).catch(() => null);
      publishSessionTelemetry(telemetry);
      return true;
    } catch (err) {
      telemetry.mark('connect_failed');
      publishSessionTelemetry(telemetry);
      setError(err?.message || 'Could not connect to bot.py');
      setIsConnecting(false);
      return false;
    }
  };

  const toggleMicrophone = async () => {
    if (!isActive) {
      await startCall();
      return;
    }
    try {
      await enableMic(!isMicEnabled);
    } catch (micError) {
      setError(micError?.message || 'Could not change microphone state');
      reportClientEvent(
        'transport.microphone_failed',
        'The browser could not change the microphone state.',
        { error_name: micError?.name || 'MicrophoneError' },
      );
    }
  };

  const sendTextMessage = async (event) => {
    event.preventDefault();
    const content = textInput.trim();
    if (!content || isSendingText || isVoiceBusy || !isActive) return;

    setError('');
    setIsSendingText(true);
    shouldAutoScrollRef.current = true;
    const messageId = `text-${Date.now()}`;
    voiceTurnTimingRef.current = {
      inputMode: 'text',
      textSubmittedAt: performance.now(),
      speechStartedAt: null,
      speechStoppedAt: null,
      turnStopSignalAt: null,
      lastLocalSpeechAt: null,
      localSpeechObserved: false,
      botTtsStartedAt: null,
      firstRemoteAudioAt: null,
      awaitingRemoteAudio: false,
    };
    pendingLatencyRef.current = null;
    setLiveLatency(null);
    addTranscript('You', content, false, messageId);
    setTextInput('');
    try {
      await pcClient.sendText(content, {
        run_immediately: true,
        audio_response: true,
      });
    } catch (err) {
      setTranscripts((items) => items.filter((item) => item.id !== messageId));
      setTextInput(content);
      setError(err?.message || 'Could not send message');
    } finally {
      setIsSendingText(false);
    }
  };

  const disconnect = useCallback(async () => {
    setError('');
    setIsConnecting(false);
    setIsVoiceBusy(false);
    if (currentCallStartedAt) {
      setCurrentCallElapsed(Math.max(
        0,
        Math.floor((Date.now() - Date.parse(currentCallStartedAt)) / 1000),
      ));
    }
    setCurrentCallStatus((status) => status === 'active' ? 'ending' : status);
    stopPeerMonitorRef.current?.();
    stopPeerMonitorRef.current = null;
    try {
      if (isActive) {
        await Promise.resolve(
          pcClient?.sendClientMessage('call.end', { call_id: currentCallIdRef.current }),
        ).catch(() => undefined);
        await Promise.resolve(pcClient?.disconnectBot()).catch(() => undefined);
        await new Promise((resolve) => window.setTimeout(resolve, 30));
      }
      await pcClient?.disconnect();
    } finally {
      activeBotMessageIdRef.current = null;
      botTextRef.current = '';
      onResetClient();
    }
  }, [currentCallStartedAt, isActive, onResetClient, pcClient]);

  React.useEffect(() => {
    const handleBeforeLogout = (event) => {
      event.detail?.respondWith?.(disconnect());
    };
    const handlePageHide = () => {
      if (!isActive) return;
      void Promise.resolve(
        pcClient?.sendClientMessage('call.end', {
          call_id: currentCallIdRef.current,
        }),
      ).catch(() => undefined);
      void Promise.resolve(pcClient?.disconnectBot()).catch(() => undefined);
      void Promise.resolve(pcClient?.disconnect()).catch(() => undefined);
    };
    window.addEventListener('aura-before-logout', handleBeforeLogout);
    window.addEventListener('pagehide', handlePageHide);
    return () => {
      window.removeEventListener('aura-before-logout', handleBeforeLogout);
      window.removeEventListener('pagehide', handlePageHide);
    };
  }, [disconnect, isActive, pcClient]);

  return (
    <div className="app-container">
      <Sidebar
        startNewCall={startNewCall}
        liveLatency={liveLatency}
        latencySamples={latencySamples}
      />

      {location.pathname === '/calls' ? <CallsPage /> : location.pathname.startsWith('/calls/') ? (
        <CallDetailPage callId={location.pathname.split('/')[2]} />
      ) : location.pathname === '/memories' ? <MemoriesPage /> : (
      <div className="main-stage page-stage">
        <div className="main-header">
          <div>
            <div className="sidebar-title" style={{ margin: 0 }}>{currentCallId ? 'Current call' : 'New call'}</div>
            {currentCallId ? <div className="active-call-strip">
              <span className={`status-pill ${currentCallStatus || 'initializing'}`}>{currentCallStatus || 'initializing'}</span>
              <code>{currentCallId}</code>
              <span className="active-call-elapsed">{Math.floor(currentCallElapsed / 60)}:{String(currentCallElapsed % 60).padStart(2, '0')}</span>
              {currentProviders ? <span className="active-provider-chips">
                {['stt', 'llm', 'tts'].map((kind) => <em key={kind}>{kind.toUpperCase()} {currentProviders[kind]?.model || currentProviders[kind]?.provider || '—'}</em>)}
              </span> : null}
            </div> : null}
          </div>
          <div className="header-actions">
            <button
              className="theme-toggle"
              onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <div className="status-indicator">
              <div className={`status-dot ${isActive ? 'active' : ''}`} style={isActive ? {backgroundColor: '#10b981'} : {}}></div>
              {statusLabel}
            </div>
          </div>
        </div>
        
        <TranscriptPanel
          transcripts={transcripts}
          transcriptAreaRef={transcriptAreaRef}
          handleTranscriptScroll={handleTranscriptScroll}
          transcriptBottomRef={transcriptBottomRef}
          toggleToolCall={toggleToolCall}
          expandedToolCalls={expandedToolCalls}
        />

        <div className="voice-controls-area">
          <div className="interaction-row">
            <form className="text-composer" onSubmit={sendTextMessage}>
              <textarea
                value={textInput}
                onChange={(event) => setTextInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder={isActive ? 'Message Aura…' : 'Start voice to enable messaging'}
                aria-label="Message Aura"
                rows={1}
                disabled={!isActive || isSendingText}
              />
              <button
                type="submit"
                className="text-send-button"
                disabled={!isActive || isVoiceBusy || isSendingText || !textInput.trim()}
                aria-label="Send message"
                title="Send message"
              >
                <Send size={18} strokeWidth={2.2} />
              </button>
            </form>
            <button
              className={`mic-button ${isActive && isMicEnabled ? 'active' : ''} ${isActive && !isMicEnabled ? 'muted' : ''}`}
              disabled={isConnecting || (!isActive && !canStart)}
              onClick={toggleMicrophone}
              aria-label={isActive ? (isMicEnabled ? 'Mute microphone' : 'Unmute microphone') : 'Start call'}
              aria-pressed={isActive && isMicEnabled}
              title={isActive ? (isMicEnabled ? 'Mute microphone' : 'Unmute microphone') : 'Start call'}
            >
              <Mic className="mic-icon" strokeWidth={2} />
            </button>
            <div className="inline-call-controls">
              <button className="control-btn" disabled={!isActive} aria-label="Bot audio is enabled" title="Bot audio is enabled">
                <Volume2 size={18} strokeWidth={2} />
              </button>
              <button
                className="control-btn"
                disabled={!isActive && !isConnecting}
                onClick={disconnect}
                style={{color: '#64748b'}}
                title="Disconnect"
              >
                <X size={18} strokeWidth={2} />
              </button>
            </div>
          </div>
          {error ? <div className="error-message">{error}</div> : null}
          {currentCallId && !isActive && !isConnecting ? <div className="post-call-actions">
            <span>Call {currentCallStatus || 'ending'}{currentRecordingStatus ? ` · recording ${currentRecordingStatus}` : ''}</span>
            <button className="secondary-button" onClick={() => navigate(`/calls/${currentCallId}`)}><ExternalLink size={15} /> View call details</button>
            <button className="secondary-button" onClick={startNewCall}>Start new call</button>
          </div> : null}
        </div>
      </div>
      )}
    </div>
  );
}

function App() {
  const [isTokenValid, setIsTokenValid] = useState(() => isValidTokenValue(localStorage.getItem('aura_token')));
  const [client, setClient] = useState(null);
  const [clientError, setClientError] = useState('');
  const iceServersRef = React.useRef(buildIceServers());
  const logoutInProgressRef = React.useRef(false);

  React.useEffect(() => {
    if (!isTokenValid) {
      return undefined;
    }
    let cancelled = false;
    const initializeClient = async () => {
      try {
        const response = await fetchWithAuth('/api/transport/ice-servers');
        const payload = await response.json();
        if (cancelled) return;
        iceServersRef.current = Array.isArray(payload.ice_servers)
          ? payload.ice_servers
          : buildIceServers();
        setClient(createPipecatClient(iceServersRef.current));
        setClientError('');
      } catch (initializationError) {
        if (!cancelled) {
          setClientError(initializationError?.message || 'Could not initialize voice transport');
        }
      }
    };
    void initializeClient();
    return () => { cancelled = true; };
  }, [isTokenValid]);

  React.useEffect(() => {
    const handleLogout = async () => {
      if (logoutInProgressRef.current) return;
      logoutInProgressRef.current = true;
      let teardown = Promise.resolve();
      window.dispatchEvent(new CustomEvent('aura-before-logout', {
        detail: {
          respondWith(value) {
            teardown = Promise.resolve(value);
          },
        },
      }));
      await Promise.race([
        teardown.catch(() => undefined),
        new Promise((resolve) => window.setTimeout(resolve, 1500)),
      ]);
      localStorage.removeItem('aura_token');
      setClient(null);
      setIsTokenValid(false);
      logoutInProgressRef.current = false;
    };
    window.addEventListener('logout', handleLogout);
    return () => window.removeEventListener('logout', handleLogout);
  }, []);

  const handleLogin = useCallback((newToken) => {
    setIsTokenValid(isValidTokenValue(newToken));
  }, []);

  const resetClient = useCallback(() => {
    setClient(createPipecatClient(iceServersRef.current));
  }, []);

  if (!isTokenValid) {
    return <Suspense fallback={null}><Auth onLogin={handleLogin} /></Suspense>;
  }
  if (clientError) {
    return <div className="app-container"><div className="error-message">{clientError}</div></div>;
  }
  if (!client) {
    return <div className="app-container"><div className="page-loading">Preparing secure voice transport…</div></div>;
  }

  return (
    <BrowserRouter>
      <PipecatClientProvider client={client}>
        <Suspense fallback={<div className="app-container" />}><VoiceApp onResetClient={resetClient} /></Suspense>
        <PipecatClientAudio />
      </PipecatClientProvider>
    </BrowserRouter>
  );
}

export default App;
