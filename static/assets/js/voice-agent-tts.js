/**
 * voice-agent-tts.js
 * TTS Lab tab: text-to-speech via POST /api/v1/voice/synthesize.
 * Includes waveform playback visualizer and a history list of last 5 items.
 */

// ── Module state ───────────────────────────────────────────────────────────
const TTS_HISTORY_MAX = 5;
const ttsHistory = [];
let ttsVisualizer = null;

// ── Synthesize ─────────────────────────────────────────────────────────────

async function ttsSynthesize() {
  const textEl = document.getElementById('ttsTextInput');
  const langEl = document.getElementById('ttsLang');
  const voiceEl = document.getElementById('ttsVoice');
  const emotionEl = document.getElementById('ttsEmotion');
  const btn = document.getElementById('btnTtsSynthesize');

  const text = textEl?.value.trim();
  if (!text) {
    ttsShowError('Please enter text to synthesize.');
    return;
  }

  btn?.setAttribute('disabled', '');
  ttsShowStatus('Synthesizing…');

  const payload = {
    text,
    language: langEl?.value || 'hi',
    voice: voiceEl?.value || 'default',
    emotion: emotionEl?.value || 'neutral',
  };

  try {
    const t0 = performance.now();
    const res = await requestJson('POST', '/api/v1/voice/synthesize', payload);
    const ms = Math.round(performance.now() - t0);

    if (!res.ok) throw new Error(res.data?.detail || `HTTP ${res.status}`);

    const { audio_base64, format, duration_seconds } = res.data;
    ttsShowStatus(`✔ ${formatDuration(duration_seconds)} · ${(format || 'wav').toUpperCase()} · ${ms}ms`);

    // Render native audio player
    const audioEl = document.getElementById('ttsAudioPlayer');
    if (audioEl && audio_base64) {
      const url = base64ToBlob(audio_base64, `audio/${format || 'wav'}`);
      audioEl.src = url;
      audioEl.parentElement?.classList.remove('audio-player--hidden');
      audioEl.style.display = 'block';
      audioEl.play().catch(() => {/* user interaction may block autoplay */});

      // Start waveform visualizer during playback
      if (ttsVisualizer) ttsVisualizer.stop();
      const canvas = document.getElementById('ttsWaveform');
      if (canvas) {
        ttsVisualizer = new MicVisualizer(canvas);
        // ? We can't easily pass the audio element stream; draw idle bars instead
        ttsDrawIdleBars(canvas);
        audioEl.onended = () => { if (ttsVisualizer) ttsVisualizer.stop(); };
      }
    }

    // Add to history
    ttsAddHistory({
      text: text.length > 60 ? text.slice(0, 57) + '…' : text,
      lang: payload.language.toUpperCase(),
      duration: formatDuration(duration_seconds),
      format: (format || 'wav').toUpperCase(),
    });

  } catch (err) {
    ttsShowError(err.message);
  } finally {
    btn?.removeAttribute('disabled');
  }
}

// ── Idle waveform animation ────────────────────────────────────────────────

/**
 * Draws a gentle idle sine-wave pulse on the canvas when audio plays.
 * @param {HTMLCanvasElement} canvas
 */
function ttsDrawIdleBars(canvas) {
  const ctx = canvas.getContext('2d');
  const barCount = 24;
  const color = getComputedStyle(document.documentElement)
    .getPropertyValue('--wave-color').trim() || '#22c55e';
  let frame = 0;
  let rafId;

  function draw() {
    const W = canvas.width = canvas.offsetWidth;
    const H = canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, W, H);
    const barW = Math.floor(W / barCount) - 2;
    for (let i = 0; i < barCount; i++) {
      const phase = (frame * 0.07) + (i * 0.45);
      const barH = ((Math.sin(phase) + 1) / 2) * H * 0.75 + H * 0.08;
      const x = i * (barW + 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.roundRect(x, H - barH, barW, barH, 3);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    frame++;
    rafId = requestAnimationFrame(draw);
  }
  draw();

  // Expose stop handle
  ttsVisualizer = { stop: () => cancelAnimationFrame(rafId) };
}

// ── History management ─────────────────────────────────────────────────────

function ttsAddHistory(item) {
  ttsHistory.unshift(item);
  if (ttsHistory.length > TTS_HISTORY_MAX) ttsHistory.pop();
  renderTtsHistory();
}

function renderTtsHistory() {
  const container = document.getElementById('ttsHistoryList');
  if (!container) return;
  container.innerHTML = '';
  ttsHistory.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'tts-history-item';
    row.innerHTML = `
      <span class="tts-history-text" title="${item.text}">${item.text}</span>
      <span class="tts-history-meta">${item.lang} · ${item.duration} · ${item.format}</span>
    `;
    container.appendChild(row);
  });
}

// ── Status / error ─────────────────────────────────────────────────────────

function ttsShowStatus(msg) {
  const el = document.getElementById('ttsStatus');
  if (!el) return;
  el.className = 'result';
  el.textContent = msg;
}

function ttsShowError(msg) {
  const el = document.getElementById('ttsStatus');
  if (!el) return;
  el.className = 'result error';
  el.textContent = `Error: ${msg}`;
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnTtsSynthesize')?.addEventListener('click', () =>
    ttsSynthesize().catch((e) => ttsShowError(e.message)));

  // Allow Ctrl+Enter in the textarea
  document.getElementById('ttsTextInput')?.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      ttsSynthesize().catch((ex) => ttsShowError(ex.message));
    }
  });
});
