import { Activity, Brain, LogOut, PhoneCall, Plus } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { summarizeLatencyCohorts, summarizeLiveLatency } from '../utils/liveLatency.js';

const NAV_ITEMS = [
  { path: '/playground', label: 'Playground', icon: Activity },
  { path: '/calls', label: 'Calls', icon: PhoneCall },
  { path: '/memories', label: 'Memories', icon: Brain },
];

export default function Sidebar({ startNewCall, liveLatency, latencySamples }) {
  const location = useLocation();
  const navigate = useNavigate();
  const formatLatency = (value) => value == null
    ? '—'
    : value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
  const latency = summarizeLiveLatency(liveLatency);
  const latencyCohorts = summarizeLatencyCohorts(latencySamples);
  const activeCohort = latencyCohorts[liveLatency?.category] || null;
  const cohortLabels = { direct: 'Direct', rag: 'RAG', tool: 'Tool' };

  return (
    <aside className="sidebar app-nav">
      <div className="sidebar-header">
        <div className="brand"><div className="brand-icon">A</div>Aura Voice</div>
        <button className="icon-btn logout-btn" onClick={() => window.dispatchEvent(new Event('logout'))} title="Log out">
          <LogOut size={16} />
        </button>
      </div>
      <button
        className="new-call-button"
        onClick={() => { startNewCall(); navigate('/playground'); }}
      >
        <Plus size={17} /> New call
      </button>
      <nav className="primary-nav" aria-label="Primary">
        {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
          <button
            key={path}
            className={location.pathname === path || (path === '/calls' && location.pathname.startsWith('/calls/')) ? 'active' : ''}
            onClick={() => navigate(path)}
          >
            <Icon size={17} /> {label}
          </button>
        ))}
      </nav>
      {liveLatency ? (
        <div className="nav-latency-card">
          <div className="eyebrow">Live latency</div>
          <div className="latency-summary">
            <div className="latency-summary-row latency-summary-primary">
              <span>Perceived latency</span>
              <strong>{formatLatency(latency.perceivedLatencyMs)}</strong>
            </div>
            <div className="latency-summary-row">
              <span>Server pipeline</span>
              <strong>{formatLatency(latency.serverPipelineMs)}</strong>
            </div>
            <div className="latency-summary-row">
              <span>Client/transport gap</span>
              <strong>{formatLatency(latency.clientTransportGapMs)}</strong>
            </div>
            <div className="latency-summary-row">
              <span>Endpointing</span>
              <strong>{formatLatency(latency.endpointingMs ?? latency.serverEndpointingMs)}</strong>
            </div>
            <div className="latency-summary-row">
              <span>Response preparation</span>
              <strong>{formatLatency(latency.responsePreparationMs)}</strong>
            </div>
            <div className="latency-summary-row">
              <span>STT fallback rate</span>
              <strong>
                {activeCohort?.fallbackRatePct == null
                  ? '—'
                  : `${activeCohort.fallbackRatePct.toFixed(1)}%`}
              </strong>
            </div>
          </div>
          <div className="latency-mini-grid">
            <span>STT <b>{formatLatency(latency.sttMs)}</b></span>
            <span>Turn release <b>{formatLatency(latency.turnReleaseMs)}</b></span>
            <span>Model TTFT <b>{formatLatency(latency.modelTtftMs)}</b></span>
            <span>TTS <b>{formatLatency(latency.ttsMs)}</b></span>
          </div>
          <div className="latency-percentiles">
            {Object.entries(cohortLabels).map(([category, label]) => {
              const cohort = latencyCohorts[category];
              if (!cohort?.count) return null;
              return (
                <div className="latency-cohort" key={category}>
                  <span>{label}</span>
                  <b>P50 {formatLatency(cohort.p50Ms)}</b>
                  <b>P90 {formatLatency(cohort.p90Ms)}</b>
                  <small>
                    n={cohort.count}
                    {cohort.fallbackRatePct == null
                      ? ''
                      : ` · STT fallback ${cohort.fallbackRatePct.toFixed(1)}%`}
                  </small>
                </div>
              );
            })}
            <small>Browser speech-end → first audible playback</small>
          </div>
        </div>
      ) : null}
    </aside>
  );
}
