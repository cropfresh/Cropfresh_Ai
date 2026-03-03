/**
 * voice-agent-rest.js
 * REST tab: record mic → POST /api/v1/voice/process → show results + play audio.
 * Also handles lone transcription and language selector.
 */

// ── Module state ───────────────────────────────────────────────────────────
const restRecorder = new MicRecorder();
const restVisualizer = (() => {
  const canvas = document.getElementById('restWaveform');
  return canvas ? new MicVisualizer(canvas) : null;
})();
const restPlayer = new AudioPlayer();

let lastVoiceResult = null; // shared with tools inspector

// ── UI helpers ─────────────────────────────────────────────────────────────

function setMicOrbState(recording) {
  const orb = document.getElementById('restMicOrb');
  const label = document.getElementById('restMicLabel');
  if (!orb) return;
  orb.classList.toggle('recording', recording);
  if (label) {
    label.textContent = recording ? 'Recording…' : 'Tap to record';
    label.classList.toggle('recording', recording);
  }
}

function showRestLoading(msg = 'Processing…') {
  setEl('restTranscript', msg, 'transcript-card');
  setEl('restResponse', '', 'response-card');
  clearBadgeStrip('restBadges');
}

function renderRestResult(data) {
  lastVoiceResult = data;
  setEl('restTranscript', data.transcription || '—', 'transcript-card');
  setEl('restResponse', data.response_text || '—', 'response-card');

  const badges = [];
  if (data.language) badges.push({ cls: 'badge-lang', label: `🌐 ${data.language.toUpperCase()}` });
  if (data.intent) badges.push({ cls: 'badge-intent', label: `⚡ ${data.intent}` });
  if (data.confidence != null) badges.push({ cls: 'badge-conf', label: `✓ ${(data.confidence * 100).toFixed(0)}%` });
  renderBadgeStrip('restBadges', badges);

  const entityChips = Object.entries(data.entities || {}).map(([k, v]) => ({
    cls: 'badge-entity',
    label: `${k}: ${Array.isArray(v) ? v.join(', ') : v}`,
  }));
  appendBadgeStrip('restBadges', entityChips);

  // Play response audio
  if (data.response_audio_base64) {
    restPlayer.enqueue(data.response_audio_base64, 'audio/wav');
  }

  // Update tools inspector if open
  if (typeof updateToolsInspector === 'function') updateToolsInspector(data);
}

function renderRestError(msg) {
  setEl('restTranscript', `Error: ${msg}`, 'transcript-card');
}

// ── Mic toggle ─────────────────────────────────────────────────────────────

async function toggleRestRecording() {
  if (restRecorder.isRecording) {
    restRecorder.stop();
    if (restVisualizer) restVisualizer.stop();
    setMicOrbState(false);
    return;
  }
  try {
    await restRecorder.start();
    setMicOrbState(true);
    if (restVisualizer) restVisualizer.start(restRecorder.getStream());
  } catch (err) {
    renderRestError(err.message);
    setMicOrbState(false);
  }
}

restRecorder.onStop = async (blob) => {
  showRestLoading();
  const lang = document.getElementById('restLang')?.value || 'auto';
  const form = new FormData();
  form.append('audio', blob, 'voice.webm');
  form.append('language', lang);
  form.append('user_id', 'voice-hub-user');
  try {
    const t0 = performance.now();
    const res = await fetch(`${BASE_API_URL}/api/v1/voice/process`, { method: 'POST', body: form });
    const ms = Math.round(performance.now() - t0);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderRestResult(data);
    setEl('restLatency', `↯ ${ms}ms`, '');
  } catch (err) {
    renderRestError(err.message);
  }
};

// ── Transcribe-only ────────────────────────────────────────────────────────

async function transcribeOnly() {
  const btn = document.getElementById('btnTranscribeToggle');
  if (!btn) return;

  if (restRecorder.isRecording) {
    restRecorder.stop();
    if (restVisualizer) restVisualizer.stop();
    setMicOrbState(false);
    btn.textContent = '🎙 Transcribe Only';
    return;
  }

  // Override onStop for transcribe mode
  const originalOnStop = restRecorder.onStop;
  restRecorder.onStop = async (blob) => {
    restRecorder.onStop = originalOnStop; // restore
    const lang = document.getElementById('restLang')?.value || 'auto';
    const form = new FormData();
    form.append('audio', blob, 'voice.webm');
    form.append('language', lang);
    try {
      const res = await fetch(`${BASE_API_URL}/api/v1/voice/transcribe`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setEl('restTranscript', data.text || '—', 'transcript-card');
      renderBadgeStrip('restBadges', [
        { cls: 'badge-lang', label: `🌐 ${(data.language || '?').toUpperCase()}` },
        { cls: 'badge-conf', label: `✓ ${((data.confidence || 0) * 100).toFixed(0)}%` },
        { cls: 'badge-provider', label: `⚙ ${data.provider || '?'}` },
      ]);
    } catch (err) {
      renderRestError(err.message);
    }
  };

  try {
    await restRecorder.start();
    setMicOrbState(true);
    if (restVisualizer) restVisualizer.start(restRecorder.getStream());
    btn.textContent = '⏹ Stop (Transcribe)';
  } catch (err) {
    renderRestError(err.message);
  }
}

// ── Shared DOM helpers ─────────────────────────────────────────────────────

function setEl(id, text, className) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (className) el.className = className;
}

function clearBadgeStrip(id) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = '';
}

function renderBadgeStrip(id, badges) {
  clearBadgeStrip(id);
  appendBadgeStrip(id, badges);
}

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

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('restMicOrb')?.addEventListener('click', () =>
    toggleRestRecording().catch((e) => renderRestError(e.message)));

  document.getElementById('btnTranscribeToggle')?.addEventListener('click', () =>
    transcribeOnly().catch((e) => renderRestError(e.message)));
});
