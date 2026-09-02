import React from 'react';
import { Trash2 } from 'lucide-react';
import { fetchWithAuth } from '../utils/api';

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
