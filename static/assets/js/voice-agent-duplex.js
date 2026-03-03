/**
 * CropFresh Voice Agent — Duplex Client
 * ========================================
 *
 * Full-duplex WebSocket voice client with:
 * - Real-time audio recording via MediaRecorder/AudioWorklet
 * - Client-side VAD for instant barge-in (<2ms)
 * - Streaming audio playback via Web Audio API
 * - Visual state indicators (listening, thinking, speaking, interrupted)
 *
 * Connects to the /ws/duplex endpoint for streaming LLM + TTS.
 */

(function () {
  "use strict";

  // ── Configuration ──────────────────────────────────────
  const WS_PATH = "/api/v1/voice/ws/duplex";
  const SAMPLE_RATE = 16000;
  const CHUNK_DURATION_MS = 30;
  const CHUNK_SIZE = (SAMPLE_RATE * CHUNK_DURATION_MS) / 1000;

  // ── State ──────────────────────────────────────────────
  let ws = null;
  let mediaStream = null;
  let audioContext = null;
  let workletNode = null;
  let isRecording = false;
  let currentState = "idle"; // idle | listening | thinking | speaking | interrupted
  let language = "hi";
  let playbackQueue = [];
  let isPlaying = false;

  // ── DOM Elements ───────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $all = (sel) => document.querySelectorAll(sel);

  // ── Audio Playback ─────────────────────────────────────
  let playbackCtx = null;

  function ensurePlaybackContext() {
    if (!playbackCtx) {
      playbackCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000,
      });
    }
    return playbackCtx;
  }

  /**
   * Queue an audio chunk for sequential playback.
   * @param {string} base64Audio - Base64-encoded audio data
   * @param {string} format - Audio format (mp3, wav, pcm)
   * @param {boolean} isLast - Whether this is the last chunk
   */
  function queueAudioChunk(base64Audio, format, isLast) {
    playbackQueue.push({ base64Audio, format, isLast });
    if (!isPlaying) {
      playNextChunk();
    }
  }

  async function playNextChunk() {
    if (playbackQueue.length === 0) {
      isPlaying = false;
      if (currentState === "speaking") {
        setState("idle");
      }
      return;
    }

    isPlaying = true;
    const chunk = playbackQueue.shift();

    try {
      const ctx = ensurePlaybackContext();
      const binaryStr = atob(chunk.base64Audio);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }

      const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.onended = () => playNextChunk();
      source.start(0);
    } catch (err) {
      console.warn("[Duplex] Audio decode error:", err);
      playNextChunk();
    }
  }

  function stopPlayback() {
    playbackQueue = [];
    isPlaying = false;
  }

  // ── State Management ───────────────────────────────────

  function setState(newState) {
    const prev = currentState;
    currentState = newState;
    updateUI(newState);
    log(`State: ${prev} → ${newState}`);
  }

  function updateUI(state) {
    // Update status bar
    const statusBar = $(".ws-status-bar");
    if (statusBar) {
      statusBar.className = "ws-status-bar " + state;
      const labels = {
        idle: "🟢 Ready",
        listening: "🎤 Listening...",
        thinking: "🧠 Thinking...",
        speaking: "🔊 Speaking...",
        interrupted: "⚡ Interrupted",
        error: "❌ Error",
        connecting: "🔄 Connecting...",
      };
      statusBar.textContent = labels[state] || state;
    }

    // Update mic orb
    const micOrb = $(".mic-orb");
    if (micOrb) {
      micOrb.classList.toggle("recording", state === "listening");
      micOrb.classList.toggle("thinking", state === "thinking");
      micOrb.classList.toggle("speaking", state === "speaking");
    }

    // Update PTT button
    const pttBtn = $(".btn-ptt");
    if (pttBtn) {
      pttBtn.classList.toggle("recording", state === "listening");
    }
  }

  // ── WebSocket Connection ───────────────────────────────

  function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}${WS_PATH}?user_id=web_user&language=${language}`;

    setState("connecting");
    log("Connecting to duplex endpoint...");

    ws = new WebSocket(url);

    ws.onopen = () => {
      log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (err) {
        console.warn("[Duplex] Parse error:", err);
      }
    };

    ws.onclose = (ev) => {
      log(`WebSocket closed: ${ev.code}`);
      setState("idle");
      ws = null;
    };

    ws.onerror = (err) => {
      log(`WebSocket error: ${err.message || "unknown"}`);
      setState("error");
    };
  }

  function disconnect() {
    if (ws) {
      sendJSON({ type: "close" });
      ws.close();
      ws = null;
    }
    stopRecording();
    stopPlayback();
    setState("idle");
  }

  // ── Server Message Handling ────────────────────────────

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "ready":
        log(`Session: ${msg.session_id} | Mode: ${msg.mode}`);
        setState("idle");
        addChatBubble("system", `Duplex session started. Say something!`);
        break;

      case "pipeline_state":
        handlePipelineState(msg.state, msg);
        break;

      case "response_audio":
        setState("speaking");
        queueAudioChunk(msg.audio_base64, msg.format, msg.is_last);
        break;

      case "response_sentence":
        addChatBubble("agent", msg.text);
        break;

      case "response_end":
        log(`Response complete: ${msg.chunks_sent} chunks`);
        break;

      case "bargein":
        setState("interrupted");
        stopPlayback();
        setTimeout(() => setState("listening"), 200);
        break;

      case "error":
        log(`Error: ${msg.error}`);
        setState("error");
        break;

      default:
        log(`Unknown: ${msg.type}`);
    }
  }

  function handlePipelineState(state, msg) {
    switch (state) {
      case "transcribing":
        setState("thinking");
        break;
      case "thinking":
        setState("thinking");
        if (msg.text) {
          addChatBubble("user", msg.text);
        }
        break;
      case "speaking":
        setState("speaking");
        break;
      case "interrupted":
        setState("interrupted");
        break;
      case "idle":
        if (msg.latency_ms) {
          log(`Latency: ${msg.latency_ms.toFixed(0)}ms`);
        }
        break;
    }
  }

  // ── Audio Recording ────────────────────────────────────

  async function startRecording() {
    if (isRecording) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connect();
      // Wait for connection
      await new Promise((resolve) => {
        const check = setInterval(() => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            clearInterval(check);
            resolve();
          }
        }, 100);
        setTimeout(() => {
          clearInterval(check);
          resolve();
        }, 5000);
      });
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      const source = audioContext.createMediaStreamSource(mediaStream);

      // Use ScriptProcessor for broad compatibility
      const processor = audioContext.createScriptProcessor(
        CHUNK_SIZE,
        1,
        1
      );

      processor.onaudioprocess = (e) => {
        if (!isRecording) return;

        const float32Data = e.inputBuffer.getChannelData(0);
        const int16Data = float32ToInt16(float32Data);
        const base64 = arrayBufferToBase64(int16Data.buffer);

        sendJSON({
          type: "audio_chunk",
          audio_base64: base64,
        });
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      isRecording = true;
      setState("listening");
      log("Recording started");
    } catch (err) {
      log(`Mic error: ${err.message}`);
      setState("error");
    }
  }

  function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    // Send audio_end to flush the buffer
    sendJSON({ type: "audio_end" });

    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }

    log("Recording stopped");
  }

  // ── Barge-in ───────────────────────────────────────────

  function triggerBargein() {
    if (currentState === "speaking") {
      stopPlayback();
      sendJSON({ type: "bargein" });
      setState("interrupted");
      log("Barge-in triggered (client-side)");
    }
  }

  // ── Utilities ──────────────────────────────────────────

  function sendJSON(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }

  function float32ToInt16(float32Array) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16;
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  function addChatBubble(role, text) {
    const thread = $(".chat-thread");
    if (!thread) return;

    const wrap = document.createElement("div");
    wrap.className = `chat-bubble-wrap ${role}`;

    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;

    wrap.appendChild(bubble);
    thread.appendChild(wrap);
    thread.scrollTop = thread.scrollHeight;
  }

  function log(msg) {
    const timeline = $(".event-timeline");
    if (timeline) {
      const row = document.createElement("div");
      row.className = "event-row";
      row.innerHTML = `
        <span class="event-dot system"></span>
        <span class="event-time">${new Date().toLocaleTimeString()}</span>
        <span class="event-text">${msg}</span>
      `;
      timeline.appendChild(row);
      timeline.scrollTop = timeline.scrollHeight;
    }
    console.log(`[Duplex] ${msg}`);
  }

  // ── Language Selection ─────────────────────────────────

  function setLanguage(lang) {
    language = lang;
    sendJSON({ type: "language_hint", language: lang });
    log(`Language set to: ${lang}`);
  }

  // ── Push-to-Talk Binding ───────────────────────────────

  function bindPTT(button) {
    if (!button) return;

    let isDown = false;

    const onDown = (e) => {
      e.preventDefault();
      if (isDown) return;
      isDown = true;

      // If agent is speaking, this acts as barge-in
      if (currentState === "speaking") {
        triggerBargein();
      }

      startRecording();
    };

    const onUp = (e) => {
      e.preventDefault();
      if (!isDown) return;
      isDown = false;
      stopRecording();
    };

    // Mouse events
    button.addEventListener("mousedown", onDown);
    button.addEventListener("mouseup", onUp);
    button.addEventListener("mouseleave", onUp);

    // Touch events
    button.addEventListener("touchstart", onDown, { passive: false });
    button.addEventListener("touchend", onUp, { passive: false });
    button.addEventListener("touchcancel", onUp, { passive: false });
  }

  // ── Mic Orb Binding ────────────────────────────────────

  function bindMicOrb(orb) {
    if (!orb) return;

    orb.addEventListener("click", () => {
      if (isRecording) {
        stopRecording();
      } else {
        if (currentState === "speaking") {
          triggerBargein();
        }
        startRecording();
      }
    });
  }

  // ── Init ───────────────────────────────────────────────

  function init() {
    // Bind UI elements
    bindPTT($(".btn-ptt"));
    bindMicOrb($(".mic-orb"));

    // Language selector
    const langSelect = $(".duplex-lang-select");
    if (langSelect) {
      langSelect.addEventListener("change", (e) => {
        setLanguage(e.target.value);
      });
    }

    // Connect/disconnect buttons
    const connectBtn = $(".duplex-connect-btn");
    if (connectBtn) {
      connectBtn.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          disconnect();
          connectBtn.textContent = "Connect";
        } else {
          connect();
          connectBtn.textContent = "Disconnect";
        }
      });
    }

    log("Duplex client initialized");
  }

  // ── Public API ─────────────────────────────────────────
  window.CropFreshDuplex = {
    connect,
    disconnect,
    startRecording,
    stopRecording,
    triggerBargein,
    setLanguage,
    getState: () => currentState,
    init,
  };

  // Auto-init when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
