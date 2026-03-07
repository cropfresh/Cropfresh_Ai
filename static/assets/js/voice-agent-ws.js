/**
 * voice-agent-ws.js
 * WebSocket tab: Push-to-talk streaming to /api/v1/voice/ws.
 *
 * Architecture:
 *   Hold PTT button → capture 250ms PCM chunks → send as JSON audio_chunk frames
 *   Release PTT    → send audio_end → server STT → VoiceAgent → TTS response audio
 *
 * Server message types handled:
 *   ready | vad_start | vad_end | transcript_partial | transcript_final |
 *   response_text | response_audio | response_end | bargein | error | language_detected
 */

// ── State ─────────────────────────────────────────────────────────────────
let wsSocket       = null;
let wsAudioCtx     = null;
let wsProcessor    = null;
let wsMicStream    = null;
let wsPttActive    = false;
let wsSessionActive= false;
let wsAgentMsgEl   = null;   // current agent bubble being filled
const wsPlayer     = new AudioPlayer();

// ── Chat UI helpers ────────────────────────────────────────────────────────

function wsGetChat() { return document.getElementById('wsChat'); }

/**
 * Append a chat bubble.
 * @param {'user'|'agent'|'system'} role
 * @param {string} text
 * @returns {HTMLElement} the bubble element (for live updates)
 */
function wsAppendBubble(role, text) {
  const chat = wsGetChat();
  if (!chat) return null;

  const wrap = document.createElement('div');
  wrap.className = `chat-bubble-wrap ${role}`;

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;

  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function wsUpdateBubble(el, text) {
  if (el) el.textContent = text;
  const chat = wsGetChat();
  if (chat) chat.scrollTop = chat.scrollHeight;
}

/** Set the live status banner */
function wsSetStatus(text, cls = '') {
  const el = document.getElementById('wsLiveStatus');
  if (!el) return;
  el.textContent = text;
  el.className = `ws-status-bar ${cls}`;
}

/** Append to event timeline */
function wsAddEvent(type, text) {
  const tl = document.getElementById('wsTimeline');
  if (!tl) return;
  const now = new Date().toLocaleTimeString('en-US', { hour12: false });
  const row = document.createElement('div');
  row.className = 'event-row';
  row.innerHTML = `<span class="event-dot ${type}"></span><span class="event-time">${now}</span><span class="event-text">${text}</span>`;
  tl.appendChild(row);
  tl.scrollTop = tl.scrollHeight;
}

// ── WebSocket connection ───────────────────────────────────────────────────

function wsConnect() {
  if (wsSocket && wsSocket.readyState === WebSocket.OPEN) return;

  const userId = document.getElementById('wsUserId')?.value || 'ws-test-user';
  const lang   = document.getElementById('wsLang')?.value || 'hi';
  const proto  = location.protocol === 'https:' ? 'wss' : 'ws';
  
  let wsSessionId = sessionStorage.getItem('voice_ws_session_id') || crypto.randomUUID();
  sessionStorage.setItem('voice_ws_session_id', wsSessionId);
  
  const url    = `${proto}://${location.host}/api/v1/voice/ws?user_id=${encodeURIComponent(userId)}&language=${lang}&session_id=${wsSessionId}`;

  wsSetStatus('🔄 Connecting…', 'connecting');
  wsSocket = new WebSocket(url);
  wsSocket.binaryType = 'arraybuffer';

  wsSocket.onopen = () => {
    wsSessionActive = true;
    setStatusPill('wsStatusPill', true, 'Connected');
    wsAddEvent('system', `✔ Connected · user=${userId} · lang=${lang.toUpperCase()}`);
    wsSetStatus('🟢 Connected — hold PTT to speak', 'connected');
    document.getElementById('btnWsConnect')?.setAttribute('disabled', '');
    document.getElementById('btnWsDisconnect')?.removeAttribute('disabled');
    document.getElementById('btnWsPtt')?.removeAttribute('disabled');
  };

  wsSocket.onclose = () => {
    wsSessionActive = false;
    setStatusPill('wsStatusPill', false, 'Disconnected');
    wsAddEvent('system', '✘ Disconnected');
    wsSetStatus('⚫ Disconnected', '');
    wsStopMic(false);
    document.getElementById('btnWsConnect')?.removeAttribute('disabled');
    document.getElementById('btnWsDisconnect')?.setAttribute('disabled', '');
    document.getElementById('btnWsPtt')?.setAttribute('disabled', '');
  };

  wsSocket.onerror = () => {
    wsAddEvent('system', '⚠ WebSocket error — is the server running?');
    wsSetStatus('🔴 Error', 'error');
    setStatusPill('wsStatusPill', false, 'Error');
  };

  wsSocket.onmessage = wsHandleMessage;
}

function wsDisconnect() {
  wsStopMic(false);
  wsSocket?.close();
}

// ── Message handler ────────────────────────────────────────────────────────

function wsHandleMessage(event) {
  let msg;
  try { msg = JSON.parse(event.data); }
  catch { return; }

  switch (msg.type) {
    case 'ready':
      if (msg.session_id) {
          sessionStorage.setItem('voice_ws_session_id', msg.session_id);
      }
      wsAddEvent('system', '▶ Voice session ready');
      wsSetStatus('🟢 Ready — hold PTT to speak', 'connected');
      break;

    case 'vad_start':
      wsAddEvent('vad', '🎙 Speech detected');
      wsSetStatus('🎙 Listening…', 'listening');
      break;

    case 'vad_end':
      wsAddEvent('vad', '🔇 Speech end — processing…');
      wsSetStatus('⏳ Transcribing…', 'thinking');
      break;

    case 'language_detected':
      wsAddEvent('lang', `🌐 ${(msg.language || '?').toUpperCase()} · ${((msg.confidence || 0) * 100).toFixed(0)}%`);
      break;

    case 'transcript_partial':
      wsSetStatus(`🎙 "${msg.text}"`, 'listening');
      break;

    case 'transcript_final': {
      const text = msg.text || '';
      wsAddEvent('transcript', `✎ "${text}" (${msg.provider || '?'})`);
      // Show user bubble
      wsAppendBubble('user', text || '(no speech detected)');
      wsSetStatus('⏳ Agent thinking…', 'thinking');
      // Prepare agent bubble
      wsAgentMsgEl = null;
      break;
    }

    case 'response_text': {
      const text = msg.text || '';
      wsAddEvent('response', `💬 ${text}`);
      // Create or update agent bubble
      if (!wsAgentMsgEl) {
        wsAgentMsgEl = wsAppendBubble('agent', text);
      } else {
        wsUpdateBubble(wsAgentMsgEl, text);
      }
      wsSetStatus('🔊 Speaking…', 'speaking');
      break;
    }

    case 'response_audio':
      if (msg.audio_base64) {
        // EdgeTTS returns MP3
        wsPlayer.enqueue(msg.audio_base64, `audio/${msg.format || 'mpeg'}`);
      }
      break;

    case 'response_end':
      wsAddEvent('response', `✔ Done (${formatDuration(msg.duration_seconds)})`);
      wsSetStatus('🟢 Ready — hold PTT to speak', 'connected');
      wsAgentMsgEl = null;
      break;

    case 'bargein':
      wsAddEvent('bargein', '⚡ Barge-in — response cancelled');
      wsPlayer.clearQueue();
      wsSetStatus('🎙 Listening…', 'listening');
      break;

    case 'error':
      wsAddEvent('system', `⚠ ${msg.error}`);
      wsAppendBubble('system', `Error: ${msg.error}`);
      wsSetStatus('🟢 Ready — hold PTT to speak', 'connected');
      break;

    default:
      wsAddEvent('system', `? ${JSON.stringify(msg)}`);
  }
}

// ── Push-to-Talk mic (PCM 16kHz mono) ─────────────────────────────────────

async function wsStartMic() {
  if (!wsSocket || wsSocket.readyState !== WebSocket.OPEN) return;
  if (wsPttActive) return;

  wsPttActive = true;
  wsSetStatus('🎙 Recording… (release to send)', 'listening');
  wsAddEvent('vad', '🎙 PTT start');

  try {
    wsMicStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    wsAudioCtx  = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = wsAudioCtx.createMediaStreamSource(wsMicStream);
    wsProcessor  = wsAudioCtx.createScriptProcessor(4096, 1, 1);

    source.connect(wsProcessor);
    wsProcessor.connect(wsAudioCtx.destination);

    wsProcessor.onaudioprocess = (e) => {
      if (!wsSocket || wsSocket.readyState !== WebSocket.OPEN || !wsPttActive) return;
      const f32   = e.inputBuffer.getChannelData(0);
      const i16   = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        i16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32768));
      }
      // base64 encode safely (handles large buffers)
      const bytes = new Uint8Array(i16.buffer);
      let bin = '';
      for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
      const b64 = btoa(bin);
      wsSocket.send(JSON.stringify({ type: 'audio_chunk', audio_base64: b64 }));
    };

    // Update PTT button style
    const btn = document.getElementById('btnWsPtt');
    if (btn) { btn.textContent = '🔴 Release to Send'; btn.classList.add('recording'); }

  } catch (err) {
    wsAddEvent('system', `⚠ Mic: ${err.message}`);
    wsPttActive = false;
    wsSetStatus('🟢 Ready — hold PTT to speak', 'connected');
  }
}

function wsStopMic(sendFlush = true) {
  if (!wsPttActive && !wsProcessor) return;

  wsPttActive = false;
  wsProcessor?.disconnect();
  wsAudioCtx?.close();
  wsMicStream?.getTracks().forEach(t => t.stop());
  wsProcessor = null;
  wsAudioCtx  = null;
  wsMicStream = null;

  const btn = document.getElementById('btnWsPtt');
  if (btn) { btn.textContent = '🎤 Hold to Speak (PTT)'; btn.classList.remove('recording'); }

  if (sendFlush && wsSocket?.readyState === WebSocket.OPEN) {
    wsSocket.send(JSON.stringify({ type: 'audio_end' }));
    wsAddEvent('vad', '🔇 PTT released → processing');
    wsSetStatus('⏳ Transcribing…', 'thinking');
  }
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnWsConnect')?.addEventListener('click', wsConnect);
  document.getElementById('btnWsDisconnect')?.addEventListener('click', wsDisconnect);

  // PTT: mousedown/touchstart = start, mouseup/touchend = stop
  const pttBtn = document.getElementById('btnWsPtt');
  if (pttBtn) {
    pttBtn.addEventListener('mousedown', () => wsStartMic().catch(e => wsAddEvent('system', `⚠ ${e.message}`)));
    pttBtn.addEventListener('mouseup',   () => wsStopMic(true));
    pttBtn.addEventListener('mouseleave',() => { if (wsPttActive) wsStopMic(true); });
    pttBtn.addEventListener('touchstart', e => { e.preventDefault(); wsStartMic().catch(e => wsAddEvent('system', `⚠ ${e.message}`)); });
    pttBtn.addEventListener('touchend',   e => { e.preventDefault(); wsStopMic(true); });
  }

  // Space bar PTT
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body && wsSessionActive && !wsPttActive) {
      e.preventDefault();
      wsStartMic().catch(() => {});
    }
  });
  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && wsPttActive) {
      e.preventDefault();
      wsStopMic(true);
    }
  });

  document.getElementById('btnWsClearTimeline')?.addEventListener('click', () => {
    const tl = document.getElementById('wsTimeline');
    if (tl) tl.innerHTML = '';
    const chat = wsGetChat();
    if (chat) chat.innerHTML = '';
  });
});
