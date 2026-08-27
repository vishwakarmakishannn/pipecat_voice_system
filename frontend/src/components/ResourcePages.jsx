import React from 'react';
import { Database, FileText, Link as LinkIcon, Trash2, Upload } from 'lucide-react';
import { fetchWithAuth } from '../utils/api';
import { hasPendingRagFiles } from '../utils/ragFiles';
import ChunkInspector from './ChunkInspector';

export function FilesPage() {
  const [files, setFiles] = React.useState([]);
  const [error, setError] = React.useState('');
  const [url, setUrl] = React.useState('');
  const [loading, setLoading] = React.useState(true);
  const [mutating, setMutating] = React.useState(false);
  const [inspectedFile, setInspectedFile] = React.useState(null);

  const load = React.useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true);
    try {
      const response = await fetchWithAuth('/api/files');
      if (!response.ok) throw new Error('Could not load files');
      setFiles(await response.json());
      setError('');
    } catch (loadError) {
      setError(loadError.message || 'Could not load files');
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    const timer = window.setTimeout(() => load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);
  React.useEffect(() => {
    if (!hasPendingRagFiles(files)) return undefined;
    const timer = window.setInterval(() => load({ quiet: true }), 2500);
    return () => window.clearInterval(timer);
  }, [files, load]);

  const upload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    const body = new FormData();
    body.append('file', file);
    setMutating(true); setError('');
    try {
      const response = await fetchWithAuth('/api/files', { method: 'POST', body });
      if (!response.ok) throw new Error('Could not upload PDF');
      await load({ quiet: true });
    } catch (uploadError) { setError(uploadError.message); }
    finally { setMutating(false); }
  };

  const addLink = async (event) => {
    event.preventDefault();
    if (!url.trim()) return;
    setMutating(true); setError('');
    try {
      const response = await fetchWithAuth('/api/files/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      });
      if (!response.ok) throw new Error('Could not add link');
      setUrl('');
      await load({ quiet: true });
    } catch (linkError) { setError(linkError.message); }
    finally { setMutating(false); }
  };

  const remove = async (id) => {
    setError('');
    const response = await fetchWithAuth(`/api/files/${id}`, { method: 'DELETE' });
    if (!response.ok) { setError('Could not delete file'); return; }
    setFiles((items) => items.filter((item) => item.id !== id));
  };

  return <main className="page-stage scroll-page">
    <div className="page-heading"><div><div className="eyebrow">Knowledge</div><h1>Files</h1><p>Documents and links available to call-time retrieval.</p></div></div>
    <div className="resource-actions"><form onSubmit={addLink}><LinkIcon size={16} /><input value={url} onChange={(event) => setUrl(event.target.value)} placeholder="Add a web link" disabled={mutating} /><button disabled={mutating}>Add</button></form><label className="secondary-button"><Upload size={16} /> Upload PDF<input hidden type="file" accept="application/pdf,.pdf" onChange={upload} disabled={mutating} /></label></div>
    {error ? <div className="error-message">{error}</div> : null}
    {loading ? <div className="page-loading">Loading files…</div> : null}
    {!loading && !files.length ? <div className="empty-page">No knowledge sources yet.</div> : null}
    <div className="resource-grid">{files.map((file) => <article key={file.id}><FileText size={20} /><div><strong>{file.title || file.filename}</strong><small>{file.status} · {file.chunk_count || 0} chunks</small></div>{file.status === 'ready' ? <button className="icon-btn" onClick={() => setInspectedFile(file)} title="Inspect stored chunks"><Database size={15} /></button> : null}<button className="icon-btn" onClick={() => remove(file.id)} title="Delete source"><Trash2 size={15} /></button></article>)}</div>
    {inspectedFile ? <ChunkInspector file={inspectedFile} onClose={() => setInspectedFile(null)} /> : null}
  </main>;
}

export function MemoriesPage() {
  const [items, setItems] = React.useState([]);
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    fetchWithAuth('/api/memories')
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Could not load memories')))
      .then(setItems)
      .catch((loadError) => setError(loadError.message))
      .finally(() => setLoading(false));
  }, []);

  const remove = async (id) => {
    const response = await fetchWithAuth(`/api/memories/${id}`, { method: 'DELETE' });
    if (!response.ok) { setError('Could not delete memory'); return; }
    setItems((current) => current.filter((item) => item.id !== id));
  };
  const removeAll = async () => {
    const response = await fetchWithAuth('/api/memories', { method: 'DELETE' });
    if (!response.ok) { setError('Could not delete memories'); return; }
    setItems([]);
  };

  return <main className="page-stage scroll-page">
    <div className="page-heading"><div><div className="eyebrow">Cross-call recall</div><h1>Memories</h1><p>Explicit saved facts that may be selectively supplied to new calls.</p></div><button className="danger-button" disabled={!items.length} onClick={removeAll}>Delete all</button></div>
    {error ? <div className="error-message">{error}</div> : null}
    {loading ? <div className="page-loading">Loading memories…</div> : null}
    {!loading && !items.length ? <div className="empty-page">No saved user facts.</div> : null}
    <div className="resource-grid memories">{items.map((item) => <article key={item.id}><div><strong>{item.key.replaceAll('_', ' ')}</strong><small>{item.fact_type}</small><p>{item.value}</p></div><button className="icon-btn" onClick={() => remove(item.id)} title="Delete memory"><Trash2 size={15} /></button></article>)}</div>
  </main>;
}
