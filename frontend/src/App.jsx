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
import { Mic, Moon, Send, Sun, Volume2, X } from 'lucide-react';
import { jwtDecode } from 'jwt-decode';
import { fetchWithAuth, API_BASE } from './utils/api';
import { buildAudioConstraints, buildIceServers, localSpeechLevelThreshold, webRTCConnectTimeoutMs } from './utils/webrtc';
import { collectWebRTCAudioStats, createSessionTelemetry, ensureBotAudioPlayback, monitorPeerConnection, publishSessionTelemetry, recordCaptureTrack, recordSelectedCandidatePair, withConnectionDeadline } from './utils/sessionTelemetry';
import './App.css';

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
const ChunkInspector = lazy(() => import('./components/ChunkInspector'));

function createPipecatClient() {
  return new PipecatClient({
    transport: new SmallWebRTCTransport({
      iceServers: buildIceServers(),
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
  const pcClient = usePipecatClient();
  const transportState = usePipecatClientTransportState();
  const { enableMic, isMicEnabled } = usePipecatClientMicControl();
  
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState('');
  
  const [transcripts, setTranscripts] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  
  const [memories, setMemories] = useState([]);
  const [isMemoryPanelOpen, setIsMemoryPanelOpen] = useState(false);
  const [isMemoryLoading, setIsMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState('');
  
  const [sidebarTab, setSidebarTab] = useState('chats');
  const [ragFiles, setRagFiles] = useState([]);
  const [isFilesLoading, setIsFilesLoading] = useState(false);
  const [isUploadingFile, setIsUploadingFile] = useState(false);
  const [isAddingLink, setIsAddingLink] = useState(false);
  const [linkUrl, setLinkUrl] = useState('');
  const [fileError, setFileError] = useState('');
  const [inspectedFile, setInspectedFile] = useState(null);
  
  const currentConversationIdRef = React.useRef(null);
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
  
  const [expandedToolCalls, setExpandedToolCalls] = useState({});
  const [liveLatency, setLiveLatency] = useState(null);
  const [isVoiceBusy, setIsVoiceBusy] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [isSendingText, setIsSendingText] = useState(false);
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('aura_theme');
    if (savedTheme === 'light' || savedTheme === 'dark') return savedTheme;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

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

  const fetchConversations = useCallback(async () => {
    try {
      const res = await fetchWithAuth(`/api/conversations`);
      if (res.ok) setConversations(await res.json());
    } catch (err) {
      console.warn('Failed to fetch conversations', err);
    }
  }, []);

  const fetchMemories = useCallback(async () => {
    setIsMemoryLoading(true);
    setMemoryError('');
    try {
      const res = await fetchWithAuth(`/api/memories`);
      if (!res.ok) throw new Error('Could not load memories');
      setMemories(await res.json());
    } catch (err) {
      setMemoryError(err?.message || 'Could not load memories');
    } finally {
      setIsMemoryLoading(false);
    }
  }, []);

  const fetchFiles = useCallback(async () => {
    setIsFilesLoading(true);
    setFileError('');
    try {
      const res = await fetchWithAuth(`/api/files`);
      if (!res.ok) throw new Error('Could not load files');
      setRagFiles(await res.json());
    } catch (err) {
      setFileError(err?.message || 'Could not load files');
    } finally {
      setIsFilesLoading(false);
    }
  }, []);

  React.useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      fetchConversations();
      fetchFiles();
    }, 0);
    return () => window.clearTimeout(timeoutId);
  }, [fetchConversations, fetchFiles]);

  React.useEffect(() => {
    if (sidebarTab !== 'files') return undefined;
    if (isVoiceBusy) return undefined;
    if (!ragFiles.some((file) => file.status === 'processing')) return undefined;
    const intervalId = window.setInterval(fetchFiles, 2500);
    return () => window.clearInterval(intervalId);
  }, [fetchFiles, isVoiceBusy, ragFiles, sidebarTab]);

  React.useEffect(() => {
    currentConversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

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

  const loadConversation = async (id) => {
    if (isActive) await disconnect();
    currentConversationIdRef.current = id;
    setCurrentConversationId(id);
    shouldAutoScrollRef.current = true;
    try {
      const res = await fetchWithAuth(`/api/conversations/${id}/messages`);
      if (res.ok) {
        const msgs = await res.json();
        setTranscripts(msgs.map(m => ({
          id: m.id,
          role: m.role,
          text: m.content,
          timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        })));
      }
    } catch (err) {
      console.warn('Failed to load conversation', err);
    }
  };

  const deleteConversation = async (e, id) => {
    e.stopPropagation();
    try {
      const res = await fetchWithAuth(`/api/conversations/${id}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        if (currentConversationId === id) {
          currentConversationIdRef.current = null;
          setCurrentConversationId(null);
          setTranscripts([]);
        }
        fetchConversations();
      }
    } catch (err) {
      console.warn('Failed to delete conversation', err);
    }
  };

  const toggleMemoryPanel = () => {
    const nextOpen = !isMemoryPanelOpen;
    setIsMemoryPanelOpen(nextOpen);
    if (nextOpen) fetchMemories();
  };

  const deleteMemory = async (id) => {
    try {
      const res = await fetchWithAuth(`/api/memories/${id}`, {
        method: 'DELETE'
      });
      if (!res.ok) throw new Error('Could not delete memory');
      setMemories((items) => items.filter((memory) => memory.id !== id));
    } catch (err) {
      setMemoryError(err?.message || 'Could not delete memory');
    }
  };

  const deleteAllMemories = async () => {
    try {
      const res = await fetchWithAuth(`/api/memories`, {
        method: 'DELETE'
      });
      if (!res.ok) throw new Error('Could not delete memories');
      setMemories([]);
    } catch (err) {
      setMemoryError(err?.message || 'Could not delete memories');
    }
  };

  const uploadRagFile = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    setIsUploadingFile(true);
    setFileError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetchWithAuth(`/api/files`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Could not upload PDF');
      }
      await fetchFiles();
    } catch (err) {
      setFileError(err?.message || 'Could not upload PDF');
    } finally {
      setIsUploadingFile(false);
    }
  };

  const addRagLink = async (event) => {
    event.preventDefault();
    const url = linkUrl.trim();
    if (!url) return;

    setIsAddingLink(true);
    setFileError('');
    try {
      const res = await fetchWithAuth(`/api/files/link`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url })
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Could not add link');
      }
      setLinkUrl('');
      await fetchFiles();
    } catch (err) {
      setFileError(err?.message || 'Could not add link');
    } finally {
      setIsAddingLink(false);
    }
  };

  const deleteRagFile = async (id) => {
    try {
      const res = await fetchWithAuth(`/api/files/${id}`, {
        method: 'DELETE'
      });
      if (!res.ok) throw new Error('Could not delete file');
      setRagFiles((items) => items.filter((file) => file.id !== id));
    } catch (err) {
      setFileError(err?.message || 'Could not delete file');
    }
  };

  const startNewConversation = async () => {
    if (isActive) await disconnect();
    currentConversationIdRef.current = null;
    setCurrentConversationId(null);
    setTranscripts([]);
    setTextInput('');
    shouldAutoScrollRef.current = true;
    setLiveLatency(null);
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
    if (!pending || timing.firstRemoteAudioAt == null) return;

    const clientMetrics = {
      client_message_to_audio_ms: Math.round(
        (timing.firstRemoteAudioAt - pending.receivedPerfAt) * 10,
      ) / 10,
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

    const webrtcMetrics = await collectWebRTCAudioStats(pcClient?.transport).catch(() => null);
    const completeMetrics = {
      ...(pending.serverPayload || {}),
      ...clientMetrics,
      ...(webrtcMetrics || {}),
    };
    setLiveLatency((current) => (
      current?.turn_id === pending.turnId
        ? { ...current, ...completeMetrics }
        : current
    ));
    if (pendingLatencyRef.current === pending) pendingLatencyRef.current = null;
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
        }).finally(() => {
          recordCaptureTrack(track, sessionTelemetryRef.current);
        });
        return;
      }
      if (!participant?.local) {
        window.setTimeout(() => {
          ensureBotAudioPlayback().catch((playbackError) => {
            setError(`Browser blocked bot audio playback: ${playbackError?.message || playbackError}`);
          });
        }, 0);
      }
    }, []),
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
      fetchConversations();
    }, [fetchConversations]),
  );

  useRTVIClientEvent(
    RTVIEvent.RemoteAudioLevel,
    useCallback((level) => {
      const timing = voiceTurnTimingRef.current;
      if (!timing.awaitingRemoteAudio || Number(level) <= 0.0001) return;
      timing.awaitingRemoteAudio = false;
      timing.firstRemoteAudioAt = performance.now();
      commitClientLatency();
    }, [commitClientLatency]),
  );

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
    }, []),
  );

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((data) => {
      const messageData = data?.data || data;
      if (messageData?.type === 'conversation_ready' && messageData.payload?.conversation_id) {
        const conversationId = messageData.payload.conversation_id;
        currentConversationIdRef.current = conversationId;
        setCurrentConversationId(conversationId);
        fetchConversations();
        return;
      }
      if (messageData?.type === 'latency_stats' && messageData.payload) {
        const receivedPerfAt = performance.now();
        const timing = voiceTurnTimingRef.current;
        // RemoteAudioLevel can be unavailable or remain continuously non-zero
        // across adjacent TTS responses. The server emits this message beside
        // the first generated audio frame, so use its receipt as a bounded
        // text-turn fallback and refine it later if decoded audio is observed.
        const textSendToAudioMs = timing.inputMode === 'text' && timing.textSubmittedAt != null
          ? Math.max(0, Math.round((receivedPerfAt - timing.textSubmittedAt) * 10) / 10)
          : null;
        pendingLatencyRef.current = {
          turnId: messageData.payload.turn_id,
          receivedPerfAt,
          serverPayload: messageData.payload,
        };
        setLiveLatency({
          ...messageData.payload,
          text_send_to_playback_ms: textSendToAudioMs,
          speech_end_signal: textSendToAudioMs == null
            ? messageData.payload.speech_end_signal
            : 'text_submitted',
          playback_signal: textSendToAudioMs == null
            ? messageData.payload.playback_signal
            : 'first_audio_server_signal',
          client_received_unix_ms: Date.now(),
        });
        commitClientLatency();
        return;
      }
      if (messageData?.type === 'assistant_transcript' && messageData.payload?.text) {
        addTranscript(
          'Aura',
          messageData.payload.text,
          false,
          messageData.payload.id || null,
        );
        return;
      }
      if (messageData?.type === 'tool_call' && messageData.payload) {
        upsertToolCall(messageData.payload);
        return;
      }
      if (messageData?.type !== 'rag_call' || !messageData.payload) return;
      const payload = messageData.payload;
      const ragCallId = payload.rag_call_id || `rag-${Date.now()}`;

      setTranscripts((items) => {
        if (items.some((item) => item.id === ragCallId)) return items;
        return capTranscriptItems([
          ...items,
          {
            id: ragCallId,
            role: 'RagCall',
            text: JSON.stringify(payload),
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          }
        ]);
      });
    }, [addTranscript, commitClientLatency, fetchConversations, upsertToolCall]),
  );

  const startConversation = async () => {
    if (!pcClient || isConnecting || !canStart) return false;

    setError('');
    setIsConnecting(true);
    setLiveLatency(null);
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
            conversation_id: currentConversationId,
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

  const disconnect = async () => {
    setError('');
    setIsConnecting(false);
    setIsVoiceBusy(false);
    stopPeerMonitorRef.current?.();
    stopPeerMonitorRef.current = null;
    try {
      await pcClient?.disconnect();
    } finally {
      activeBotMessageIdRef.current = null;
      botTextRef.current = '';
      onResetClient();
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        isMemoryPanelOpen={isMemoryPanelOpen}
        toggleMemoryPanel={toggleMemoryPanel}
        memories={memories}
        deleteAllMemories={deleteAllMemories}
        deleteMemory={deleteMemory}
        isMemoryLoading={isMemoryLoading}
        memoryError={memoryError}
        sidebarTab={sidebarTab}
        setSidebarTab={setSidebarTab}
        startNewConversation={startNewConversation}
        conversations={conversations}
        currentConversationId={currentConversationId}
        loadConversation={loadConversation}
        deleteConversation={deleteConversation}
        fetchFiles={fetchFiles}
        ragFiles={ragFiles}
        isFilesLoading={isFilesLoading}
        isUploadingFile={isUploadingFile}
        uploadRagFile={uploadRagFile}
        isAddingLink={isAddingLink}
        linkUrl={linkUrl}
        setLinkUrl={setLinkUrl}
        addRagLink={addRagLink}
        fileError={fileError}
        deleteRagFile={deleteRagFile}
        inspectRagFile={setInspectedFile}
        liveLatency={liveLatency}
      />

      <div className="main-stage">
        <div className="main-header">
          <div className="sidebar-title" style={{ margin: 0 }}>New conversation</div>
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
              onClick={() => isActive ? enableMic(!isMicEnabled) : startConversation()}
              aria-label={isActive ? (isMicEnabled ? 'Mute microphone' : 'Unmute microphone') : 'Start conversation'}
              aria-pressed={isActive && isMicEnabled}
              title={isActive ? (isMicEnabled ? 'Mute microphone' : 'Unmute microphone') : 'Start conversation'}
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
        </div>
      </div>
      {inspectedFile ? (
        <ChunkInspector file={inspectedFile} onClose={() => setInspectedFile(null)} />
      ) : null}
    </div>
  );
}

function App() {
  const [isTokenValid, setIsTokenValid] = useState(() => isValidTokenValue(localStorage.getItem('aura_token')));
  const [client, setClient] = useState(() => createPipecatClient());

  React.useEffect(() => {
    const handleLogout = () => {
      localStorage.removeItem('aura_token');
      setIsTokenValid(false);
    };
    window.addEventListener('logout', handleLogout);
    return () => window.removeEventListener('logout', handleLogout);
  }, []);

  const handleLogin = useCallback((newToken) => {
    setIsTokenValid(isValidTokenValue(newToken));
  }, []);

  const resetClient = useCallback(() => {
    setClient(createPipecatClient());
  }, []);

  if (!isTokenValid) {
    return <Suspense fallback={null}><Auth onLogin={handleLogin} /></Suspense>;
  }

  return (
    <PipecatClientProvider client={client}>
      <Suspense fallback={<div className="app-container" />}><VoiceApp onResetClient={resetClient} /></Suspense>
      <PipecatClientAudio />
    </PipecatClientProvider>
  );
}

export default App;
