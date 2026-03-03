/**
 * voice-agent-core.js
 * Shared utilities: tab switching, mic visualizer, mic recorder, audio player queue.
 */

// ── Tab switching ──────────────────────────────────────────────────────────

/**
 * Activates a tab panel by id and updates tab buttons.
 * @param {string} panelId
 */
function activateTab(panelId) {
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === panelId);
  });
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === panelId);
  });
}

// ── Mic Waveform Visualizer ────────────────────────────────────────────────

/**
 * Draws animated frequency bars on a canvas from a mic stream.
 * Call stop() to cancel animation when done.
 */
class MicVisualizer {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this._canvas = canvas;
    this._ctx = canvas.getContext('2d');
    this._rafId = null;
    this._analyser = null;
    this._dataArray = null;
  }

  /**
   * @param {MediaStream} stream
   */
  start(stream) {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    this._analyser = audioCtx.createAnalyser();
    this._analyser.fftSize = 64;
    this._dataArray = new Uint8Array(this._analyser.frequencyBinCount);
    source.connect(this._analyser);
    this._draw();
  }

  _draw() {
    const { _canvas: canvas, _ctx: ctx, _analyser: analyser, _dataArray: data } = this;
    analyser.getByteFrequencyData(data);

    const W = canvas.width = canvas.offsetWidth;
    const H = canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, W, H);

    const barCount = data.length;
    const barW = Math.floor(W / barCount) - 1;
    const activeColor = getComputedStyle(document.documentElement)
      .getPropertyValue('--wave-color').trim() || '#22c55e';

    for (let i = 0; i < barCount; i++) {
      const barH = (data[i] / 255) * H * 0.9;
      const x = i * (barW + 1);
      ctx.fillStyle = activeColor;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.roundRect(x, H - barH, barW, barH, 3);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    this._rafId = requestAnimationFrame(() => this._draw());
  }

  stop() {
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    if (this._canvas) {
      const ctx = this._canvas.getContext('2d');
      ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    }
  }
}

// ── Mic Recorder ──────────────────────────────────────────────────────────

/**
 * Wraps MediaRecorder with start/stop control.
 * Calls onStop(blob) when recording ends.
 */
class MicRecorder {
  constructor() {
    this._recorder = null;
    this._stream = null;
    this._chunks = [];
    /** @type {((blob: Blob) => void) | null} */
    this.onStop = null;
  }

  get isRecording() {
    return this._recorder?.state === 'recording';
  }

  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._chunks = [];
    this._recorder = new MediaRecorder(this._stream, { mimeType: 'audio/webm' });
    this._recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this._chunks.push(e.data);
    };
    this._recorder.onstop = () => {
      this._stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(this._chunks, { type: 'audio/webm' });
      if (this.onStop) this.onStop(blob);
    };
    this._recorder.start();
  }

  stop() {
    if (this._recorder && this._recorder.state !== 'inactive') {
      this._recorder.stop();
    }
  }

  /** Returns raw mic stream for visualization (call after start()) */
  getStream() {
    return this._stream;
  }
}

// ── Audio Player (queue-based base64 audio) ────────────────────────────────

/**
 * Queue-based audio player; enqueues base64 chunks and plays them in order.
 */
class AudioPlayer {
  constructor() {
    this._queue = [];
    this._playing = false;
  }

  /** @param {string} b64  @param {string} [mime] */
  enqueue(b64, mime = 'audio/wav') {
    this._queue.push({ b64, mime });
    if (!this._playing) this._next();
  }

  _next() {
    if (this._queue.length === 0) { this._playing = false; return; }
    this._playing = true;
    const { b64, mime } = this._queue.shift();
    const url = base64ToBlob(b64, mime);
    playAudioBlob(url)
      .catch(() => {/* swallow playback errors silently */})
      .finally(() => this._next());
  }

  clearQueue() {
    this._queue = [];
  }
}

// ── Boot: wire up tab buttons ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
  });
});
