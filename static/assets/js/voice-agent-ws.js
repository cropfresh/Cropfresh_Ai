/**
 * voice-agent-ws.js
 * WebSocket tab: binary PCM streaming to /api/v1/voice/ws.
 * Handles all server MessageType events: VAD, transcript, response_audio, barge-in, language.
 */

// ── Module state ───────────────────────────────────────────────────────────
let wsSocket = null;
let wsAudioCtx = null;
let wsScriptProcessor = null;
let wsMicStream = null;
const wsPlayer = new AudioPlayer();
let wsResponseChunks = []; // accumulate response audio chunks
let wsSessionActive = false;

// ── Event timeline ─────────────────────────────────────────────────────────

/**
 * Appends a row to the WS event timeline.
 * @param {'vad'|'transcript'|'response'|'bargein'|'lang'|'system'} type
 * @param {string} text
 */
function wsAddEvent(type, text) {
  const timeline = document.getElementById('wsTimeline');
  if (!timeline) return;

  const now = new Date().toLocaleTimeString('en-US', { hour12: false });
  const row = document.createElement('div');
  row.className = 'event-row';
  row.innerHTML = `
    <span class="event-dot ${type}"></span>
    <span class="event-time">${now}</span>
    <span class="event-text">${text}</span>
  `;
  timeline.appendChild(row);
  timeline.scrollTop = timeline.scrollHeight;
}

// ── WebSocket connection ───────────────────────────────────────────────────

function wsConnect() {
  if (wsSocket && wsSocket.readyState === WebSocket.OPEN) return;

  const userId = document.getElementById('wsUserId')?.value || 'ws-test-user';
  const lang = document.getElementById('wsLang')?.value || 'hi';
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${protocol}://${window.location.host}/api/v1/voice/ws?user_id=${encodeURIComponent(userId)}&language=${lang}`;

  wsSocket = new WebSocket(url);
  wsSocket.binaryType = 'arraybuffer';

  wsSocket.onopen = () => {
    wsSessionActive = true;
    setStatusPill('wsStatusPill', true, 'Connected');
    wsAddEvent('system', `✔ Connected · user=${userId} · lang=${lang.toUpperCase()}`);
    document.getElementById('btnWsConnect')?.setAttribute('disabled', '');
    document.getElementById('btnWsDisconnect')?.removeAttribute('disabled');
    document.getElementById('btnWsMic')?.removeAttribute('disabled');
  };

  wsSocket.onclose = () => {
    wsSessionActive = false;
    setStatusPill('wsStatusPill', false, 'Disconnected');
    wsAddEvent('system', '✘ Disconnected');
    wsStopMic();
    document.getElementById('btnWsConnect')?.removeAttribute('disabled');
    document.getElementById('btnWsDisconnect')?.setAttribute('disabled', '');
    document.getElementById('btnWsMic')?.setAttribute('disabled', '');
  };

  wsSocket.onerror = () => {
    wsAddEvent('system', '⚠ WebSocket error — check server is running');
    setStatusPill('wsStatusPill', false, 'Error');
  };

  wsSocket.onmessage = wsHandleMessage;
}

function wsDisconnect() {
  wsStopMic();
  wsSocket?.close();
}

// ── Message handler ────────────────────────────────────────────────────────

function wsHandleMessage(event) {
  // Server sends JSON text frames
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    wsAddEvent('system', `⚠ Non-JSON frame: ${typeof event.data}`);
    return;
  }

  const type = msg.type || '';

  switch (type) {
    case 'ready':
      wsAddEvent('system', '▶ Pipeline ready');
      break;
    case 'vad_start':
      wsAddEvent('vad', `🎙 Speech start at ${msg.timestamp_ms ?? '?'}ms`);
      break;
    case 'vad_speech':
      // * Intentionally silent — too frequent to show
      break;
    case 'vad_end':
      wsAddEvent('vad', `🔇 Speech end at ${msg.timestamp_ms ?? '?'}ms`);
      break;
    case 'language_detected':
      wsAddEvent('lang', `🌐 Language: ${(msg.language || '?').toUpperCase()} · ${((msg.confidence || 0) * 100).toFixed(0)}%`);
      break;
    case 'transcript_partial':
      wsAddEvent('transcript', `◌ Partial: ${msg.text}`);
      break;
    case 'transcript_final':
      wsAddEvent('transcript', `✎ "${msg.text}" (${(msg.provider || '?')})`);
      wsResponseChunks = []; // reset for next response
      break;
    case 'response_text':
      wsAddEvent('response', `💬 ${msg.text}`);
      break;
    case 'response_audio':
      // * Accumulate base64 audio chunks; play when complete
      if (msg.audio_base64) wsPlayer.enqueue(msg.audio_base64, `audio/${msg.format || 'wav'}`);
      break;
    case 'response_end':
      wsAddEvent('response', `✔ Response done (${formatDuration(msg.duration_seconds)})`);
      break;
    case 'bargein':
      wsAddEvent('bargein', '⚡ Barge-in — response cancelled');
      wsPlayer.clearQueue();
      break;
    case 'error':
      wsAddEvent('system', `⚠ Error: ${msg.error}`);
      break;
    default:
      wsAddEvent('system', `? Unknown: ${JSON.stringify(msg)}`);
  }
}

// ── Mic streaming (PCM 16kHz mono) ────────────────────────────────────────

async function wsStartMic() {
  if (!wsSocket || wsSocket.readyState !== WebSocket.OPEN) {
    wsAddEvent('system', '⚠ Connect to WebSocket first');
    return;
  }
  try {
    wsMicStream = await navigator.mediaDevices.getUserMedia({ audio: {
      sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true,
    }});
    wsAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = wsAudioCtx.createMediaStreamSource(wsMicStream);
    wsScriptProcessor = wsAudioCtx.createScriptProcessor(4096, 1, 1);

    source.connect(wsScriptProcessor);
    wsScriptProcessor.connect(wsAudioCtx.destination);

    wsScriptProcessor.onaudioprocess = (event) => {
      if (!wsSocket || wsSocket.readyState !== WebSocket.OPEN) return;
      const floatData = event.inputBuffer.getChannelData(0);
      // Convert Float32 → Int16
      const int16 = new Int16Array(floatData.length);
      for (let i = 0; i < floatData.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, floatData[i] * 32768));
      }
      // Send as base64 JSON matching the server's audio_chunk format
      const b64 = btoa(String.fromCharCode(...new Uint8Array(int16.buffer)));
      wsSocket.send(JSON.stringify({ type: 'audio_chunk', audio_base64: b64 }));
    };

    wsAddEvent('vad', '🎙 Mic streaming started (16kHz PCM)');
    document.getElementById('btnWsMic')?.setAttribute('disabled', '');
    document.getElementById('btnWsStopMic')?.removeAttribute('disabled');
  } catch (err) {
    wsAddEvent('system', `⚠ Mic error: ${err.message}`);
  }
}

function wsStopMic() {
  wsScriptProcessor?.disconnect();
  wsAudioCtx?.close();
  wsMicStream?.getTracks().forEach((t) => t.stop());
  wsScriptProcessor = null;
  wsAudioCtx = null;
  wsMicStream = null;
  document.getElementById('btnWsMic')?.removeAttribute('disabled');
  document.getElementById('btnWsStopMic')?.setAttribute('disabled', '');
  wsAddEvent('vad', '🔇 Mic streaming stopped');
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnWsConnect')?.addEventListener('click', wsConnect);
  document.getElementById('btnWsDisconnect')?.addEventListener('click', wsDisconnect);
  document.getElementById('btnWsMic')?.addEventListener('click', () =>
    wsStartMic().catch((e) => wsAddEvent('system', `⚠ ${e.message}`)));
  document.getElementById('btnWsStopMic')?.addEventListener('click', wsStopMic);
  document.getElementById('btnWsClearTimeline')?.addEventListener('click', () => {
    const tl = document.getElementById('wsTimeline');
    if (tl) tl.innerHTML = '';
  });
});
