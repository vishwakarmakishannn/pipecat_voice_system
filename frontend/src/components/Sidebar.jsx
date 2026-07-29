import { Plus, LogOut, Brain, MessageSquare, Trash2, Link as LinkIcon, Upload, RefreshCw, FileText, Activity, Database } from 'lucide-react';

export default function Sidebar({
  // Memory props
  isMemoryPanelOpen,
  toggleMemoryPanel,
  memories,
  deleteAllMemories,
  deleteMemory,
  isMemoryLoading,
  memoryError,
  
  // Tab props
  sidebarTab,
  setSidebarTab,
  
  // Conversation props
  startNewConversation,
  conversations,
  currentConversationId,
  loadConversation,
  deleteConversation,
  
  // File props
  fetchFiles,
  ragFiles,
  isFilesLoading,
  isUploadingFile,
  uploadRagFile,
  isAddingLink,
  linkUrl,
  setLinkUrl,
  addRagLink,
  fileError,
  deleteRagFile,
  inspectRagFile,
  liveLatency
}) {

  const formatLatency = (value) => {
    if (value == null) return '—';
    return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
  };

  const primaryLatency = (item) => (
    item.text_send_to_playback_ms ?? item.user_stop_to_playback_ms ?? item.answer_audio_ms
  );

  const formatFileSize = (sizeBytes) => {
    if (!sizeBytes) return '0 KB';
    if (sizeBytes < 1024 * 1024) return `${Math.max(1, Math.round(sizeBytes / 1024))} KB`;
    return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const sourceDisplayName = (file) => {
    if ((file.source_type || 'pdf') === 'link') {
      return file.title || file.site_name || file.final_url || file.url || file.filename;
    }
    return file.filename;
  };

  const sourceMeta = (file) => {
    const sourceType = file.source_type || 'pdf';
    if (sourceType === 'link') {
      const url = file.final_url || file.url || '';
      return `${formatFileSize(file.size_bytes)} · ${file.chunk_count} chunks${url ? ` · ${url}` : ''}`;
    }
    return `${formatFileSize(file.size_bytes)} · ${file.chunk_count} chunks`;
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-icon">A</div>
          Aura Voice
        </div>
        <div className="sidebar-controls">
          <button className={`icon-btn ${isMemoryPanelOpen ? 'active' : ''}`} onClick={toggleMemoryPanel} title="Saved Memories">
            <Brain size={16} strokeWidth={2.5} />
          </button>
          <button className="icon-btn" onClick={startNewConversation} title="New Conversation">
            <Plus size={16} strokeWidth={2.5} />
          </button>
          <button className="icon-btn logout-btn" onClick={() => window.dispatchEvent(new Event('logout'))} title="Logout">
            <LogOut size={16} strokeWidth={2.5} />
          </button>
        </div>
      </div>
      {isMemoryPanelOpen ? (
        <div className="memory-panel">
          <div className="memory-popover-header">
            <div>
              <div className="memory-popover-title">Saved memories</div>
              <div className="memory-popover-subtitle">{memories.length} active</div>
            </div>
            <button
              className="memory-delete-all"
              onClick={deleteAllMemories}
              disabled={!memories.length || isMemoryLoading}
            >
              Delete all
            </button>
          </div>

          {memoryError ? <div className="memory-error">{memoryError}</div> : null}

          <div className="memory-list">
            {isMemoryLoading ? (
              <div className="memory-empty">Loading...</div>
            ) : memories.length ? (
              memories.map((memory) => (
                <div className="memory-item" key={memory.id}>
                  <div className="memory-item-text">
                    <div className="memory-label">
                      {memory.key.replaceAll('_', ' ')}
                      <span>{memory.fact_type}</span>
                    </div>
                    <div className="memory-value">{memory.value}</div>
                  </div>
                  <button
                    className="memory-delete-btn"
                    onClick={() => deleteMemory(memory.id)}
                    title="Delete memory"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            ) : (
              <div className="memory-empty">No saved memories yet</div>
            )}
          </div>
        </div>
      ) : null}
      <div className="sidebar-tabs">
        <button
          className={`sidebar-tab ${sidebarTab === 'chats' ? 'active' : ''}`}
          onClick={() => setSidebarTab('chats')}
        >
          Chats
        </button>
        <button
          className={`sidebar-tab ${sidebarTab === 'files' ? 'active' : ''}`}
          onClick={() => {
            setSidebarTab('files');
            fetchFiles();
          }}
        >
          Files
        </button>
      </div>

      <div className="sidebar-content" style={{ overflowY: 'auto' }}>
        {sidebarTab === 'chats' ? (
          conversations.length === 0 ? (
            <>
              <div className="sidebar-title">No conversations</div>
              <div className="sidebar-subtitle">Your history will appear here</div>
            </>
          ) : (
            <div className="history-list">
              {conversations.map(conv => (
                <div 
                  key={conv.id} 
                  className={`history-item ${currentConversationId === conv.id ? 'active' : ''}`}
                  onClick={() => loadConversation(conv.id)}
                >
                  <MessageSquare size={16} className="history-icon" />
                  <div className="history-text">
                    <div className="history-title">{conv.title}</div>
                    <div className="history-time">{new Date(conv.created_at).toLocaleDateString()}</div>
                  </div>
                  <button className="delete-btn" onClick={(e) => deleteConversation(e, conv.id)} title="Delete">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )
        ) : (
          <div className="files-panel">
            <form className="link-form" onSubmit={addRagLink}>
              <div className="link-input-wrap">
                <LinkIcon size={15} />
                <input
                  value={linkUrl}
                  onChange={(event) => setLinkUrl(event.target.value)}
                  placeholder="Paste a link"
                  disabled={isAddingLink}
                  inputMode="url"
                />
              </div>
              <button className="link-add-btn" type="submit" disabled={isAddingLink || !linkUrl.trim()}>
                {isAddingLink ? 'Adding...' : 'Add'}
              </button>
            </form>
            <label className={`file-upload ${isUploadingFile ? 'disabled' : ''}`}>
              <Upload size={16} />
              <span>{isUploadingFile ? 'Uploading...' : 'Upload PDF'}</span>
              <input
                type="file"
                accept="application/pdf,.pdf"
                disabled={isUploadingFile}
                onChange={uploadRagFile}
              />
            </label>
            <button className="file-refresh-btn" onClick={fetchFiles} disabled={isFilesLoading} title="Refresh files">
              <RefreshCw size={15} className={isFilesLoading ? 'spinning' : ''} />
            </button>

            {fileError ? <div className="file-error">{fileError}</div> : null}

            <div className="file-list">
              {isFilesLoading && !ragFiles.length ? (
                <div className="file-empty">Loading files...</div>
              ) : ragFiles.length ? (
                ragFiles.map((file) => (
                  <div className="file-item" key={file.id}>
                    {(file.source_type || 'pdf') === 'link' ? (
                      <LinkIcon size={17} className="file-icon" />
                    ) : (
                      <FileText size={17} className="file-icon" />
                    )}
                    <div className="file-details">
                      <div className="file-name" title={sourceDisplayName(file)}>{sourceDisplayName(file)}</div>
                      <div className="file-meta">
                        {sourceMeta(file)}
                      </div>
                      {file.error ? <div className="file-error-text">{file.error}</div> : null}
                    </div>
                    <div className="file-actions">
                      <span className="source-type">{(file.source_type || 'pdf') === 'link' ? 'Link' : 'PDF'}</span>
                      <span className={`file-status ${file.status}`}>{file.status}</span>
                      <button
                        className="file-action-btn file-chunks"
                        onClick={() => inspectRagFile(file)}
                        disabled={file.status !== 'ready' || !file.chunk_count}
                        title={file.chunk_count ? 'Inspect stored chunks' : 'No stored chunks'}
                      >
                        <Database size={14} />
                      </button>
                      <button className="delete-btn file-delete" onClick={() => deleteRagFile(file.id)} title="Delete file">
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="file-empty">No PDFs uploaded yet</div>
              )}
            </div>
          </div>
        )}
      </div>
      {sidebarTab === 'chats' ? (
        <div className="latency-panel">
          <div className="latency-title">
            <span><Activity size={14} /> Live latency</span>
          </div>
          <div className="latency-latest">
            <span>{liveLatency ? (liveLatency.with_tools ? 'With tool' : 'Without tool') : 'Waiting for a turn'}</span>
            <strong>{formatLatency(liveLatency ? primaryLatency(liveLatency) : null)}</strong>
          </div>
          <div className="latency-footnote">
            {liveLatency?.text_send_to_playback_ms != null
              ? 'Text sent → first decoded playback audio'
              : liveLatency?.user_stop_to_playback_ms != null
              ? liveLatency.speech_end_signal === 'last_nonzero_local_audio_level'
                ? `Last local speech → decoded audio${liveLatency.endpointing_ms != null ? ` · endpoint ${formatLatency(liveLatency.endpointing_ms)}` : ''}`
                : 'Turn-stop signal → first decoded playback audio'
              : 'Final transcript → first generated answer audio'}
          </div>
        </div>
      ) : null}
    </div>
  );
}
