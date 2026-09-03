const ERROR_SEVERITIES = new Set(['error', 'critical']);

export function normalizeDiagnosticSeverity(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

export function isLiveErrorSeverity(value) {
  return ERROR_SEVERITIES.has(normalizeDiagnosticSeverity(value));
}

export function isHiddenDiagnostic(item) {
  return item?.role === 'Diagnostic';
}

export function visibleTranscriptItems(items, showDiagnostics = false) {
  if (showDiagnostics) return items;
  return items.filter((item) => !isHiddenDiagnostic(item));
}
