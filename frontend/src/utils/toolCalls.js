export function hasToolCallResult(payload) {
  return Object.prototype.hasOwnProperty.call(payload, 'result');
}

export function formatToolCallStatus(status) {
  if (!status) return '';
  return String(status).replaceAll('_', ' ');
}

export function getToolCallDisplayStatus(payload) {
  const businessStatus = payload?.result && typeof payload.result === 'object'
    ? payload.result.status
    : null;
  return businessStatus || payload?.status || '';
}
