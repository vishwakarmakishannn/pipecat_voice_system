import { User, FileText, Wrench, ChevronDown, ChevronRight, Mic } from 'lucide-react';

export default function TranscriptPanel({
  transcripts,
  transcriptAreaRef,
  handleTranscriptScroll,
  transcriptBottomRef,
  toggleToolCall,
  expandedToolCalls
}) {
  return (
    <div
      className="transcription-area"
      ref={transcriptAreaRef}
      onScroll={handleTranscriptScroll}
    >
      {transcripts.length ? (
        <div className="transcript-list">
          {transcripts.map((item) => (
            <div className="transcript-item" key={item.id}>
              <div className={`transcript-avatar ${item.role === 'You' ? 'user-avatar' : item.role === 'ToolCall' || item.role === 'RagCall' ? 'tool-avatar' : 'bot-avatar'}`}>
                {item.role === 'You' ? <User size={18} strokeWidth={2.5} /> : item.role === 'ToolCall' ? <Wrench size={16} /> : item.role === 'RagCall' ? <FileText size={16} /> : 'A'}
              </div>
              <div className="transcript-message">
                <div className="transcript-role">
                  {item.role === 'Aura' ? 'Aura AI' : item.role === 'ToolCall' ? 'Tool Call' : item.role === 'RagCall' ? 'RAG Call' : item.role}
                  {item.timestamp && <span className="transcript-time">{item.timestamp}</span>}
                </div>
                {item.role === 'ToolCall' || item.role === 'RagCall' ? (() => {
                  let parsed;
                  try { parsed = JSON.parse(item.text); } catch { parsed = { function_name: 'Unknown', arguments: item.text }; }
                  return (
                    <div className="tool-call-block">
                      <div className="tool-call-header" onClick={() => toggleToolCall(item.id)}>
                        {expandedToolCalls[item.id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        <span style={{fontWeight: 500, color: '#475569'}}>{parsed.function_name}</span>
                      </div>
                      {expandedToolCalls[item.id] && (
                        <div className="tool-call-body">
                          <div className="tool-call-section">
                            <strong style={{fontSize: '0.75rem', color: '#64748b'}}>Arguments</strong>
                            <pre>{JSON.stringify(parsed.arguments, null, 2)}</pre>
                          </div>
                          {parsed.result && (
                            <div className="tool-call-section" style={{marginTop: '8px', borderTop: '1px solid #e2e8f0', paddingTop: '8px'}}>
                              <strong style={{fontSize: '0.75rem', color: '#64748b'}}>Result</strong>
                              <pre style={{maxHeight: '200px', overflowY: 'auto'}}>{JSON.stringify(parsed.result, null, 2)}</pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })() : (
                  <div className="transcript-text">{item.text}</div>
                )}
              </div>
            </div>
          ))}
          <div ref={transcriptBottomRef} aria-hidden="true" />
        </div>
      ) : (
        <div className="empty-state">
           <Mic size={48} color="#cbd5e1" strokeWidth={1} />
           <div className="empty-text">Start talking to Aura Voice</div>
        </div>
      )}
    </div>
  );
}
