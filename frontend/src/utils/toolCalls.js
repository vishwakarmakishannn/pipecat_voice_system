export function hasToolCallResult(payload) {
  return Object.prototype.hasOwnProperty.call(payload, 'result');
}

export function formatToolCallStatus(status) {
  if (!status) return '';
  return String(status).replaceAll('_', ' ');
}
