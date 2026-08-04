import assert from 'node:assert/strict';
import test from 'node:test';

import { ensureBotAudioPlayback } from '../src/utils/sessionTelemetry.js';

function stream(track) {
  return { getAudioTracks: () => track ? [track] : [] };
}

test('bot playback waits for the current track and retries a load race', async () => {
  const expectedTrack = { id: 'remote-new', readyState: 'live' };
  const oldTrack = { id: 'remote-old', readyState: 'live' };
  let animationFrames = 0;
  let playCalls = 0;
  const element = {
    srcObject: stream(oldTrack),
    muted: true,
    volume: 0,
    async play() {
      playCalls += 1;
    },
  };

  globalThis.document = { querySelector: () => element };
  globalThis.window = {
    requestAnimationFrame(callback) {
      animationFrames += 1;
      if (animationFrames === 1) element.srcObject = stream(expectedTrack);
      queueMicrotask(() => callback(performance.now()));
    },
  };

  assert.equal(await ensureBotAudioPlayback(expectedTrack), true);
  assert.equal(playCalls, 1);
  assert.equal(element.muted, false);
  assert.equal(element.volume, 1);

  playCalls = 0;
  element.play = async () => {
    playCalls += 1;
    if (playCalls === 1) {
      throw new DOMException('interrupted by a new load', 'AbortError');
    }
  };

  assert.equal(await ensureBotAudioPlayback(expectedTrack), true);
  assert.equal(playCalls, 2);
});

test('bot playback ignores a track that ended before attachment', async () => {
  const endedTrack = { id: 'remote-ended', readyState: 'ended' };
  let playCalls = 0;
  const element = {
    srcObject: stream(null),
    async play() {
      playCalls += 1;
    },
  };

  globalThis.document = { querySelector: () => element };

  assert.equal(await ensureBotAudioPlayback(endedTrack), false);
  assert.equal(playCalls, 0);
});

test('bot playback preserves a genuine browser autoplay rejection', async () => {
  const expectedTrack = { id: 'remote-new', readyState: 'live' };
  const element = {
    srcObject: stream(expectedTrack),
    async play() {
      throw new DOMException('playback blocked', 'NotAllowedError');
    },
  };

  globalThis.document = { querySelector: () => element };

  await assert.rejects(
    ensureBotAudioPlayback(expectedTrack),
    (error) => error.name === 'NotAllowedError',
  );
});
