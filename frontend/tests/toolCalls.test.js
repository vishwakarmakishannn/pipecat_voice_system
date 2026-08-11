import assert from 'node:assert/strict';
import test from 'node:test';

import { formatToolCallStatus, hasToolCallResult } from '../src/utils/toolCalls.js';

test('tool result visibility is based on field presence, including null', () => {
  assert.equal(hasToolCallResult({ result: null }), true);
  assert.equal(hasToolCallResult({ result: { status: 'timeout' } }), true);
  assert.equal(hasToolCallResult({ status: 'in_progress' }), false);
});

test('tool status is formatted for transcript display', () => {
  assert.equal(formatToolCallStatus('in_progress'), 'in progress');
  assert.equal(formatToolCallStatus('completed'), 'completed');
  assert.equal(formatToolCallStatus(undefined), '');
});
