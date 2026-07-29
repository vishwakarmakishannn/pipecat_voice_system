import { useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight, Database, X } from 'lucide-react';
import { fetchWithAuth } from '../utils/api';

const PAGE_SIZE = 20;

export default function ChunkInspector({ file, onClose }) {
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    fetchWithAuth(`/api/files/${file.id}/chunks?offset=${offset}&limit=${PAGE_SIZE}`)
      .then((response) => response.json())
      .then((data) => { if (active) setPage(data); })
      .catch((err) => {
        if (!active) return;
        setError(err.status === 404
          ? 'Chunk inspector endpoint was not found. Restart the backend, then reopen this inspector.'
          : (err.message || 'Could not load stored chunks'));
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [file.id, offset]);

  useEffect(() => {
    const closeOnEscape = (event) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  const total = page?.total ?? (error ? 0 : (file.chunk_count ?? 0));
  const end = Math.min(offset + (page?.items.length ?? 0), total);
  const goToOffset = (nextOffset) => {
    setLoading(true);
    setError('');
    setOffset(nextOffset);
  };

  return (
    <div className="chunk-inspector-overlay" role="dialog" aria-modal="true" aria-label="Stored chunk inspector">
      <section className="chunk-inspector">
        <header className="chunk-inspector-header">
          <div>
            <div className="chunk-inspector-title"><Database size={18} /> Stored chunks</div>
            <div className="chunk-inspector-source" title={file.title || file.filename}>{file.title || file.filename}</div>
          </div>
          <button className="chunk-close" onClick={onClose} title="Close"><X size={19} /></button>
        </header>

        <div className="chunk-summary">
          <span>{total} chunks in vector DB</span>
          <span>Exact stored text</span>
          <span>Vector preview shows first 8 values</span>
        </div>

        <main className="chunk-list">
          {loading ? <div className="chunk-state">Loading stored chunks…</div> : null}
          {error ? <div className="chunk-state error">{error}</div> : null}
          {!loading && !error && !page?.items.length ? <div className="chunk-state">No chunks are stored for this source.</div> : null}
          {!loading && !error ? page?.items.map((chunk) => (
            <article className="chunk-card" key={chunk.id}>
              <div className="chunk-card-header">
                <strong>Chunk {chunk.chunk_index + 1}</strong>
                <div className="chunk-badges">
                  {chunk.page_start ? <span>Page {chunk.page_start}{chunk.page_end && chunk.page_end !== chunk.page_start ? `–${chunk.page_end}` : ''}</span> : null}
                  <span className={chunk.embedding_stored ? 'ok' : 'missing'}>{chunk.embedding_stored ? `${chunk.embedding_dimension}D vector stored` : 'Vector missing'}</span>
                  <span className={chunk.search_indexed ? 'ok' : 'missing'}>{chunk.search_indexed ? 'Keyword indexed' : 'Keyword index missing'}</span>
                </div>
              </div>
              {chunk.heading_path ? <div className="chunk-heading">{chunk.heading_path}</div> : null}
              <pre className="chunk-content">{chunk.content}</pre>
              <div className="chunk-vector">
                <span>{chunk.content_chars.toLocaleString()} characters</span>
                {chunk.embedding_preview.length ? <code>[{chunk.embedding_preview.map((value) => value.toFixed(5)).join(', ')}, …]</code> : null}
              </div>
            </article>
          )) : null}
        </main>

        <footer className="chunk-pagination">
          <span>{total ? `Showing ${offset + 1}–${end} of ${total}` : '0 chunks'}</span>
          <div>
            <button disabled={loading || offset === 0} onClick={() => goToOffset(Math.max(0, offset - PAGE_SIZE))}><ChevronLeft size={16} /> Previous</button>
            <button disabled={loading || offset + PAGE_SIZE >= total} onClick={() => goToOffset(offset + PAGE_SIZE)}>Next <ChevronRight size={16} /></button>
          </div>
        </footer>
      </section>
    </div>
  );
}
