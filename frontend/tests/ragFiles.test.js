import assert from 'node:assert/strict';
import test from 'node:test';

import { hasPendingRagFiles } from '../src/utils/ragFiles.js';


test('RAG polling remains active for durable queued and claimed jobs', () => {
  assert.equal(hasPendingRagFiles([{ status: 'queued' }]), true);
  assert.equal(hasPendingRagFiles([{ status: 'processing' }]), true);
  assert.equal(hasPendingRagFiles([{ status: 'ready' }, { status: 'failed' }]), false);
});
