/**
 * voice-agent-rest.js
 * REST tab: record mic via MicRecorder → POST /api/v1/voice/process → chat bubble UI.
 *
 * Conversation flow:
 *   User speaks → stop → show user bubble → API call → show agent bubble + play MP3 audio
 */

// ── State ─────────────────────────────────────────────────────────────────
const restRecorder  = new MicRecorder();
const restVisualizer = (() => {
  const c = document.getElementById('restWaveform');
  return c ? new MicVisualizer(c) : null;
})();
const restPlayer = new AudioPlayer();
let lastVoiceResult = null; // shared with tools inspector

// ── Chat helpers ───────────────────────────────────────────────────────────

function restGetChat() { return document.getElementById('restChat'); }

/**
 * Append a chat bubble to the REST conversation thread.
 * @param {'user'|'agent'|'system'} role
 * @param {string} text
 */
function restAppendBubble(role, text) {
  const chat = restGetChat();
  if (!chat) return;
  const wrap = document.createElement('div');
  wrap.className = `chat-bubble-wrap ${role}`;
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function restSetStatus(msg, cls = '') {
  const s = document.getElementById('restStatus');
  if (!s) return;
  s.textContent = msg;
  s.className = `rest-status ${cls}`;
}

// ── Legacy text elements (keep for badge strip compat) ─────────────────────
function setEl(id, text, className) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (className) el.className = className;
}
function clearBadgeStrip(id)  { const el = document.getElementById(id); if (el) el.innerHTML = ''; }
function renderBadgeStrip(id, badges) { clearBadgeStrip(id); appendBadgeStrip(id, badges); }
function appendBadgeStrip(id, badges) {
  const el = document.getElementById(id);
  if (!el) return;
  badges.forEach(({ cls, label }) => {
    const chip = document.createElement('span');
    chip.className = `badge ${cls}`;
    chip.textContent = label;
    el.appendChild(chip);
  });
}

// ── Mic orb ────────────────────────────────────────────────────────────────

function setMicOrbState(recording) {
  const orb   = document.getElementById('restMicOrb');
  const label = document.getElementById('restMicLabel');
  if (!orb) return;
  orb.classList.toggle('recording', recording);
  if (label) {
    label.textContent = recording ? 'Recording… (tap to stop)' : 'Tap to record';
    label.classList.toggle('recording', recording);
  }
}

// ── REST voice process ─────────────────────────────────────────────────────

function renderRestResult(data) {
  lastVoiceResult = data;

  // Chat bubble — user transcript
  const transcript = data.transcription?.trim() || '(no speech detected)';
  restAppendBubble('user', transcript);

  // Chat bubble — agent response
  const response = data.response_text?.trim() || '(no response)';
  restAppendBubble('agent', response);

  // Badges
  const badges = [];
  if (data.language) badges.push({ cls: 'badge-lang',   label: `🌐 ${data.language.toUpperCase()}` });
  if (data.intent)   badges.push({ cls: 'badge-intent', label: `⚡ ${data.intent}` });
  if (data.confidence != null) badges.push({ cls: 'badge-conf', label: `✓ ${(data.confidence * 100).toFixed(0)}%` });
  renderBadgeStrip('restBadges', badges);

  const entityChips = Object.entries(data.entities || {}).map(([k, v]) => ({
    cls: 'badge-entity',
    label: `${k}: ${Array.isArray(v) ? v.join(', ') : v}`,
  }));
  appendBadgeStrip('restBadges', entityChips);

  // Play MP3 response audio
  if (data.response_audio_base64) {
    restPlayer.enqueue(data.response_audio_base64, 'audio/mpeg');
    restSetStatus('🔊 Playing response…', 'speaking');
  } else {
    restSetStatus('✅ Done', '');
  }

  // Update tools inspector
  if (typeof updateToolsInspector === 'function') updateToolsInspector(data);
}

function renderRestError(msg) {
  restAppendBubble('system', `⚠ Error: ${msg}`);
  restSetStatus('🔴 Error', 'error');
}

// ── Mic toggle ─────────────────────────────────────────────────────────────

async function toggleRestRecording() {
  if (restRecorder.isRecording) {
    restRecorder.stop();
    restVisualizer?.stop();
    setMicOrbState(false);
    return;
  }
  try {
    await restRecorder.start();
    setMicOrbState(true);
    restSetStatus('🎙 Recording… tap orb to stop', 'listening');
    restVisualizer?.start(restRecorder.getStream());
  } catch (err) {
    renderRestError(err.message);
    setMicOrbState(false);
  }
}

restRecorder.onStop = async (blob) => {
  restSetStatus('⏳ Processing voice…', 'thinking');
  const lang = document.getElementById('restLang')?.value || 'auto';
  const form = new FormData();
  form.append('audio', blob, 'voice.webm');
  form.append('language', lang);
  form.append('user_id', 'voice-hub-user');

  try {
    const t0  = performance.now();
    const res = await fetch(`${BASE_API_URL}/api/v1/voice/process`, { method: 'POST', body: form });
    const ms  = Math.round(performance.now() - t0);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    renderRestResult(data);
    setEl('restLatency', `↯ ${ms}ms`, '');
  } catch (err) {
    renderRestError(err.message);
  }
};

// ── Transcribe-only mode ──────────────────────────────────────────────────

async function transcribeOnly() {
  const btn = document.getElementById('btnTranscribeToggle');
  if (!btn) return;

  if (restRecorder.isRecording) {
    restRecorder.stop();
    restVisualizer?.stop();
    setMicOrbState(false);
    btn.textContent = '🎙 Transcribe Only';
    return;
  }

  const originalOnStop = restRecorder.onStop;
  restRecorder.onStop = async (blob) => {
    restRecorder.onStop = originalOnStop;
    const lang = document.getElementById('restLang')?.value || 'auto';
    const form = new FormData();
    form.append('audio', blob, 'voice.webm');
    form.append('language', lang);

    restSetStatus('⏳ Transcribing…', 'thinking');
    try {
      const res = await fetch(`${BASE_API_URL}/api/v1/voice/transcribe`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      restAppendBubble('user', data.text || '(empty)');
      renderBadgeStrip('restBadges', [
        { cls: 'badge-lang',     label: `🌐 ${(data.language || '?').toUpperCase()}` },
        { cls: 'badge-conf',     label: `✓ ${((data.confidence || 0) * 100).toFixed(0)}%` },
        { cls: 'badge-provider', label: `⚙ ${data.provider || '?'}` },
      ]);
      restSetStatus('✅ Transcription done', '');
    } catch (err) {
      renderRestError(err.message);
    }
  };

  try {
    await restRecorder.start();
    setMicOrbState(true);
    restSetStatus('🎙 Recording (transcribe only)…', 'listening');
    restVisualizer?.start(restRecorder.getStream());
    btn.textContent = '⏹ Stop (Transcribe)';
  } catch (err) {
    renderRestError(err.message);
  }
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('restMicOrb')?.addEventListener('click', () =>
    toggleRestRecording().catch(e => renderRestError(e.message)));

  document.getElementById('btnTranscribeToggle')?.addEventListener('click', () =>
    transcribeOnly().catch(e => renderRestError(e.message)));
});
