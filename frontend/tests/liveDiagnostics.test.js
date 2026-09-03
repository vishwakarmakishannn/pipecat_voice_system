import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isHiddenDiagnostic,
  isLiveErrorSeverity,
  normalizeDiagnosticSeverity,
  visibleTranscriptItems,
} from '../src/utils/liveDiagnostics.js';

test('diagnostic severity is normalized before classification', () => {
  assert.equal(normalizeDiagnosticSeverity(' CRITICAL '), 'critical');
  assert.equal(normalizeDiagnosticSeverity(null), '');
});

test('only backend error severities are immediately visible as errors', () => {
  assert.equal(isLiveErrorSeverity('error'), true);
  assert.equal(isLiveErrorSeverity('critical'), true);
  assert.equal(isLiveErrorSeverity('warning'), false);
  assert.equal(isLiveErrorSeverity('info'), false);
  assert.equal(isLiveErrorSeverity(undefined), false);
});

test('ordinary diagnostics are hidden unless the operator enables them', () => {
  const items = [
    { id: 'user', role: 'You' },
    { id: 'info', role: 'Diagnostic' },
    { id: 'error', role: 'Error' },
  ];

  assert.equal(isHiddenDiagnostic(items[1]), true);
  assert.deepEqual(visibleTranscriptItems(items), [items[0], items[2]]);
  assert.deepEqual(visibleTranscriptItems(items, true), items);
});
