import React from 'react';
import { AlertTriangle, ArrowLeft, ChevronDown, ChevronRight, Copy, Download, RotateCcw, Trash2, Wrench } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { fetchWithAuth } from '../utils/api';

const formatDuration = (ms) => {
  if (ms == null) return '—';
  const seconds = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
};
const formatLatency = (ms) => ms == null ? '—' : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
const providerLabel = (item) => [item?.provider, item?.model].filter(Boolean).join(' · ') || 'Not captured';

export function CallsPage() {
  const navigate = useNavigate();
  const [items, setItems] = React.useState([]);
  const [nextCursor, setNextCursor] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [deleted, setDeleted] = React.useState(false);
  const [statusFilter, setStatusFilter] = React.useState('');
  const [errorsOnly, setErrorsOnly] = React.useState(false);
  const [provider, setProvider] = React.useState('');
  const [model, setModel] = React.useState('');
  const [recordingStatus, setRecordingStatus] = React.useState('');
  const [startedFrom, setStartedFrom] = React.useState('');
  const [startedTo, setStartedTo] = React.useState('');
  const requestGeneration = React.useRef(0);
  const requestController = React.useRef(null);

  const load = React.useCallback(async (cursor = null, append = false) => {
    const generation = requestGeneration.current + 1;
    requestGeneration.current = generation;
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setLoading(true); setError('');
    const params = new URLSearchParams({ deleted: String(deleted) });
    if (cursor) params.set('cursor', cursor);
    if (statusFilter) params.set('status', statusFilter);
    if (errorsOnly) params.set('has_errors', 'true');
    if (provider.trim()) params.set('provider', provider.trim());
    if (model.trim()) params.set('model', model.trim());
    if (recordingStatus) params.set('recording_status', recordingStatus);
    if (startedFrom) params.set('started_from', new Date(`${startedFrom}T00:00:00`).toISOString());
    if (startedTo) params.set('started_to', new Date(`${startedTo}T23:59:59.999`).toISOString());
    try {
      const response = await fetchWithAuth(`/api/calls?${params}`, {
        signal: controller.signal,
      });
      if (!response.ok) throw new Error('Could not load calls');
      const data = await response.json();
      if (generation !== requestGeneration.current) return;
      setItems((current) => append ? [...current, ...data.items] : data.items);
      setNextCursor(data.next_cursor);
    } catch (loadError) {
      if (generation === requestGeneration.current && !controller.signal.aborted) {
        setError(loadError.message);
      }
    } finally {
      if (generation === requestGeneration.current) setLoading(false);
    }
  }, [deleted, errorsOnly, model, provider, recordingStatus, startedFrom, startedTo, statusFilter]);

  React.useEffect(() => {
    const timer = window.setTimeout(() => load(), 250);
    return () => {
      window.clearTimeout(timer);
      requestController.current?.abort();
    };
  }, [load]);

  const mutate = async (call, action) => {
    try {
      const response = await fetchWithAuth(`/api/calls/${call.id}${action === 'restore' ? '/restore' : ''}`, { method: action === 'restore' ? 'POST' : 'DELETE' });
      if (!response.ok) throw new Error(action === 'restore' ? 'Could not restore call' : 'Could not delete call');
      setItems((current) => current.filter((item) => item.id !== call.id));
    } catch (mutationError) {
      setError(mutationError.message);
    }
  };

  return (
    <main className="page-stage scroll-page">
      <div className="page-heading"><div><div className="eyebrow">Analysis</div><h1>Calls</h1><p>Immutable call records, recordings, latency, tools, and diagnostics.</p></div></div>
      <div className="filter-bar">
        <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} aria-label="Filter by status">
          <option value="">All statuses</option><option value="completed">Completed</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option><option value="abandoned">Abandoned</option>
        </select>
        <label><input type="checkbox" checked={errorsOnly} onChange={(event) => setErrorsOnly(event.target.checked)} /> Has errors</label>
        <label><input type="checkbox" checked={deleted} onChange={(event) => setDeleted(event.target.checked)} /> Deleted calls</label>
        <input value={provider} onChange={(event) => setProvider(event.target.value)} placeholder="Provider" aria-label="Filter by provider" />
        <input value={model} onChange={(event) => setModel(event.target.value)} placeholder="Model" aria-label="Filter by model" />
        <select value={recordingStatus} onChange={(event) => setRecordingStatus(event.target.value)} aria-label="Filter by recording status">
          <option value="">All recordings</option><option value="available">Available</option><option value="processing">Processing</option><option value="failed">Failed</option>
        </select>
        <label>From <input type="date" value={startedFrom} onChange={(event) => setStartedFrom(event.target.value)} /></label>
        <label>To <input type="date" value={startedTo} onChange={(event) => setStartedTo(event.target.value)} /></label>
      </div>
      {error ? <div className="error-message">{error}</div> : null}
      <div className="calls-table-wrap">
        <table className="calls-table"><thead><tr><th>Status</th><th>Call</th><th>Started</th><th>Duration</th><th>Providers</th><th>Turns</th><th>Errors</th><th>Recording</th><th /></tr></thead>
          <tbody>{items.map((call) => <tr key={call.id} onClick={() => navigate(`/calls/${call.id}`)}>
            <td><span className={`status-pill ${call.status}`}>{call.status}</span></td>
            <td><strong>{call.title}</strong><small>{call.summary || 'No summary yet'}</small><small>{call.end_reason || 'active'}</small></td>
            <td>{new Date(call.started_at).toLocaleString()}</td><td>{formatDuration(call.duration_ms)}</td>
            <td><small>STT · {providerLabel(call.providers.stt)}</small><small>LLM · {providerLabel(call.providers.llm)}</small><small>TTS · {providerLabel(call.providers.tts)}</small></td><td>{call.counts.turns}</td>
            <td className={call.counts.errors ? 'danger-text' : ''}>{call.counts.errors}</td><td>{call.recording?.status || '—'}</td>
            <td><button className="icon-btn" onClick={(event) => { event.stopPropagation(); mutate(call, deleted ? 'restore' : 'delete'); }} title={deleted ? 'Restore' : 'Delete'}>{deleted ? <RotateCcw size={15} /> : <Trash2 size={15} />}</button></td>
          </tr>)}</tbody>
        </table>
        {!loading && !items.length ? <div className="empty-page">No calls match these filters.</div> : null}
      </div>
      {loading ? <div className="page-loading">Loading calls…</div> : null}
      {nextCursor ? <button className="secondary-button" onClick={() => load(nextCursor, true)}>Load more</button> : null}
    </main>
  );
}

function RecordingPlayer({ call, audioRef }) {
  const [url, setUrl] = React.useState('');
  const [error, setError] = React.useState('');
  const [speed, setSpeed] = React.useState(1);
  const refreshAttempted = React.useRef(false);
  const loadAccess = React.useCallback(async () => {
    const response = await fetchWithAuth(`/api/calls/${call.id}/recording-access`, { method: 'POST' });
    if (!response.ok) throw new Error('Recording access failed');
    const data = await response.json();
    setUrl(data.url);
    setError('');
    return data.url;
  }, [call.id]);
  React.useEffect(() => {
    if (call.recording?.status !== 'available') return;
    refreshAttempted.current = false;
    const timer = window.setTimeout(() => {
      loadAccess().catch((accessError) => setError(accessError.message));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [call.recording?.status, loadAccess]);
  if (call.recording?.status !== 'available') return <div className="recording-placeholder">Recording: {call.recording?.status || 'not available'}{call.recording?.failure_message ? ` — ${call.recording.failure_message}` : ''}</div>;
  if (error) return <div className="recording-placeholder danger-text">{error}</div>;
  if (!url) return <div className="recording-placeholder">Preparing secure playback…</div>;
  const changeSpeed = (value) => {
    const next = Number(value);
    setSpeed(next);
    if (audioRef.current) audioRef.current.playbackRate = next;
  };
  const refreshPlayback = () => {
    if (refreshAttempted.current) {
      setError('Secure recording access expired or playback failed. Reload this call to retry.');
      return;
    }
    refreshAttempted.current = true;
    loadAccess().catch((accessError) => setError(accessError.message));
  };
  const download = async () => {
    try {
      const freshUrl = await loadAccess();
      const anchor = document.createElement('a');
      anchor.href = freshUrl;
      anchor.download = `call-${call.id}.mp3`;
      anchor.rel = 'noopener';
      anchor.click();
    } catch (downloadError) {
      setError(downloadError.message);
    }
  };
  return <div className="recording-player"><audio ref={audioRef} controls preload="metadata" src={url} onCanPlay={() => { refreshAttempted.current = false; }} onError={refreshPlayback} /><label>Speed <select value={speed} onChange={(event) => changeSpeed(event.target.value)}><option value="0.75">0.75×</option><option value="1">1×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><button className="icon-btn" onClick={download} title="Download MP3"><Download size={16} /></button></div>;
}

function TimelineItem({ item, seek, linkedEvent }) {
  const [expanded, setExpanded] = React.useState(false);
  if (item.item_type === 'transcript') return <article className="timeline-item transcript-card" onClick={() => item.audio_offset_ms != null && seek(item.audio_offset_ms)}>
    <div className={`timeline-icon ${item.speaker === 'You' ? 'user-avatar' : 'bot-avatar'}`}>{item.speaker === 'You' ? 'Y' : 'A'}</div><div><div className="timeline-title">{item.speaker === 'Aura' ? 'Aura AI' : 'You'} <span>{new Date(item.created_at).toLocaleTimeString()}</span><em>{item.source.replaceAll('_', ' ')}</em>{linkedEvent ? <em className="recovery-link">recovery for {linkedEvent.code}</em> : null}</div><p>{item.text}</p></div>
  </article>;
  const isError = item.item_type === 'event';
  const payload = item.item_type === 'operation' ? { arguments: item.arguments, result: item.result } : { technical_detail: item.technical_detail, details: item.details };
  return <article className={`timeline-item structured-card ${isError ? item.severity : ''}`}>
    <div className="timeline-icon">{isError ? <AlertTriangle size={16} /> : <Wrench size={16} />}</div><div className="structured-main">
      <button className="structured-header" onClick={() => setExpanded((value) => !value)}>{expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}<strong>{isError ? item.code : item.name}</strong><span className={`status-pill ${isError ? item.outcome : item.status}`}>{isError ? item.outcome : item.status}</span><small>{formatLatency(item.duration_ms)}</small></button>
      {isError ? <p>{item.message}</p> : null}{expanded ? <pre>{JSON.stringify(payload, null, 2)}</pre> : null}
    </div>
  </article>;
}

export function CallDetailPage({ callId }) {
  const navigate = useNavigate();
  const audioRef = React.useRef(null);
  const [call, setCall] = React.useState(null);
  const [timeline, setTimeline] = React.useState([]);
  const [turns, setTurns] = React.useState([]);
  const [timelineCursor, setTimelineCursor] = React.useState(null);
  const [turnCursor, setTurnCursor] = React.useState(null);
  const [tab, setTab] = React.useState('timeline');
  const [filter, setFilter] = React.useState('all');
  const [error, setError] = React.useState('');

  React.useEffect(() => {
    const controller = new AbortController();
    let current = true;
    const parseResponse = async (response, message) => {
      if (!response.ok) throw new Error(message);
      return response.json();
    };
    Promise.all([
      fetchWithAuth(`/api/calls/${callId}`, { signal: controller.signal }).then((r) => parseResponse(r, 'Call not found')),
      fetchWithAuth(`/api/calls/${callId}/timeline`, { signal: controller.signal }).then((r) => parseResponse(r, 'Could not load timeline')),
      fetchWithAuth(`/api/calls/${callId}/turns`, { signal: controller.signal }).then((r) => parseResponse(r, 'Could not load metrics')),
    ]).then(([callData, timelineData, turnsData]) => {
      if (!current) return;
      setCall(callData); setTimeline(timelineData.items); setTimelineCursor(timelineData.next_cursor); setTurns(turnsData.items); setTurnCursor(turnsData.next_cursor);
    }).catch((loadError) => {
      if (current && !controller.signal.aborted) setError(loadError.message);
    });
    return () => {
      current = false;
      controller.abort();
    };
  }, [callId]);
  if (error) return <main className="page-stage"><div className="error-message">{error}</div></main>;
  if (!call) return <main className="page-stage"><div className="page-loading">Loading call…</div></main>;
  const visible = timeline.filter((item) => filter === 'all' || (filter === 'conversation' ? item.item_type === 'transcript' : filter === 'operations' ? item.item_type === 'operation' : item.item_type === 'event'));
  const recoveredByTurn = new Map(
    timeline
      .filter((item) => item.item_type === 'event' && item.recovered && item.turn_id != null)
      .map((item) => [item.turn_id, item]),
  );
  const seek = (offset) => { if (audioRef.current) audioRef.current.currentTime = offset / 1000; };
  const loadMoreTimeline = async () => {
    const data = await fetchWithAuth(`/api/calls/${callId}/timeline?after=${timelineCursor}`).then((response) => response.json());
    setTimeline((items) => [...items, ...data.items]); setTimelineCursor(data.next_cursor);
  };
  const loadMoreTurns = async () => {
    const data = await fetchWithAuth(`/api/calls/${callId}/turns?after=${turnCursor}`).then((response) => response.json());
    setTurns((items) => [...items, ...data.items]); setTurnCursor(data.next_cursor);
  };
  const mutateCall = async () => {
    const restoring = Boolean(call.deleted_at);
    const response = await fetchWithAuth(`/api/calls/${call.id}${restoring ? '/restore' : ''}`, { method: restoring ? 'POST' : 'DELETE' });
    if (response.ok) navigate('/calls');
    else setError(restoring ? 'Could not restore call' : 'Could not delete call');
  };
  return <main className="page-stage call-detail scroll-page">
    <header className="call-sticky-header"><button className="icon-btn" onClick={() => navigate('/calls')}><ArrowLeft size={17} /></button><div className="call-identity"><div><h1>{call.title}</h1><span className={`status-pill ${call.status}`}>{call.status}</span></div><p>{new Date(call.started_at).toLocaleString()} · {formatDuration(call.duration_ms)} · {call.end_reason || 'active'}</p><button className="copy-id" onClick={() => navigator.clipboard.writeText(call.id)}><Copy size={13} /> {call.id}</button></div><button className={call.deleted_at ? 'secondary-button' : 'danger-button'} onClick={mutateCall}>{call.deleted_at ? <RotateCcw size={15} /> : <Trash2 size={15} />}{call.deleted_at ? 'Restore' : 'Delete'}</button></header>
    <RecordingPlayer call={call} audioRef={audioRef} />
    <div className="provider-grid">{['stt', 'llm', 'tts'].map((kind) => <div className="provider-card" key={kind}><span>{kind.toUpperCase()}</span><strong>{call.providers[kind].provider || '—'}</strong><small>{call.providers[kind].model || 'model unavailable'}</small></div>)}</div>
    <div className="detail-tabs">{['timeline', 'metrics', 'configuration'].map((name) => <button key={name} className={tab === name ? 'active' : ''} onClick={() => setTab(name)}>{name}</button>)}</div>
    {tab === 'timeline' ? <section><div className="filter-bar compact">{['all', 'conversation', 'operations', 'errors'].map((name) => <button className={filter === name ? 'active' : ''} key={name} onClick={() => setFilter(name)}>{name}</button>)}</div><div className="timeline-list">{visible.map((item) => <TimelineItem key={`${item.item_type}-${item.id}`} item={item} seek={seek} linkedEvent={item.source === 'spoken_recovery' ? recoveredByTurn.get(item.turn_id) : null} />)}</div>{timelineCursor ? <button className="secondary-button" onClick={loadMoreTimeline}>Load more timeline</button> : null}</section> : null}
    {tab === 'metrics' ? <section>
      <div className="metric-grid">
        <div><span>Duration</span><strong>{formatDuration(call.duration_ms)}</strong></div>
        <div><span>Turns</span><strong>{call.counts.turns}</strong></div>
        <div><span>Interruptions</span><strong>{call.counts.interruptions}</strong></div>
        <div><span>Tools</span><strong>{call.counts.tools}</strong></div>
        <div><span>Warnings</span><strong>{call.counts.warnings}</strong></div>
        <div><span>Errors</span><strong>{call.counts.errors}</strong></div>
        <div><span>Direct perceived p50 / p90</span><strong>{formatLatency(call.latency.p50_ms)} / {formatLatency(call.latency.p90_ms)}</strong><small>Browser end-to-end · n={call.latency.sample_count ?? 0}</small></div>
        {[['stt', 'STT'], ['turn_release', 'Turn release'], ['model_ttft', 'Model TTFT'], ['response_preparation', 'Response preparation'], ['tts', 'TTS']].map(([kind, label]) => <div key={kind}><span>{label} p50 / p90</span><strong>{formatLatency(call.latency.components?.[kind]?.p50_ms)} / {formatLatency(call.latency.components?.[kind]?.p90_ms)}</strong></div>)}
        <div><span>Tool / RAG time</span><strong>{formatLatency(call.latency.components?.tool_total_ms)} / {formatLatency(call.latency.components?.rag_total_ms)}</strong></div>
        <div><span>LLM input / output</span><strong>{call.usage?.llm_input_tokens || 0} / {call.usage?.llm_output_tokens || 0} tokens</strong></div>
        <div><span>STT audio</span><strong>{formatDuration(call.usage?.stt_audio_ms || 0)}</strong></div>
        <div><span>TTS usage</span><strong>{call.usage?.tts_characters || 0} characters</strong></div>
      </div>
      <table className="calls-table"><thead><tr><th>Turn</th><th>Category</th><th>STT</th><th>Turn release</th><th>Model TTFT</th><th>Response prep</th><th>TTS</th><th>Tool</th><th>RAG</th><th>End-to-end</th><th>Usage</th></tr></thead><tbody>{turns.map((turn) => <tr key={turn.id}><td>{turn.sequence}</td><td>{turn.metrics?.category || '—'}</td><td>{formatLatency(turn.stt_latency_ms)}</td><td>{formatLatency(turn.metrics?.turn_release_ms)}</td><td>{formatLatency(turn.metrics?.llm_ttft_ms)}</td><td>{formatLatency(turn.metrics?.response_preparation_ms ?? turn.llm_latency_ms)}</td><td>{formatLatency(turn.tts_latency_ms)}</td><td>{formatLatency(turn.tool_latency_ms)}</td><td>{formatLatency(turn.rag_latency_ms)}</td><td>{formatLatency(turn.end_to_end_latency_ms)}</td><td><small>{turn.llm_input_tokens || 0}/{turn.llm_output_tokens || 0} tokens · {turn.tts_characters || 0} chars</small></td></tr>)}</tbody></table>
      {turnCursor ? <button className="secondary-button" onClick={loadMoreTurns}>Load more turns</button> : null}
    </section> : null}
    {tab === 'configuration' ? <section className="config-grid">
      <div className="provider-grid">{['stt', 'llm', 'tts'].map((kind) => <div className="provider-card" key={kind}><span>{kind.toUpperCase()}</span><strong>{providerLabel(call.providers[kind])}</strong>{call.providers[kind].language ? <small>Language · {call.providers[kind].language}</small> : null}{call.providers[kind].voice ? <small>Voice · {call.providers[kind].voice}</small> : null}</div>)}</div>
      <pre>{JSON.stringify({ transport: call.transport, direction: call.direction, ...call.configuration }, null, 2)}</pre>
    </section> : null}
  </main>;
}
