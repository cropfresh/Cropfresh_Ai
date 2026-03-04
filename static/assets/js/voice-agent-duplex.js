/**
 * CropFresh Voice Agent — Duplex Client v2
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

  const WS_PATH = "/api/v1/voice/ws/duplex";
  const SAMPLE_RATE = 16000;
  const CHUNK_DURATION_MS = 30;
  const CHUNK_SIZE = (SAMPLE_RATE * CHUNK_DURATION_MS) / 1000;

  let ws = null;
  let mediaStream = null;
  let audioContext = null;
  let workletNode = null;
  let fallbackProcessor = null;
  let isRecording = false;
  let currentState = "idle";
  let language = "hi";
  let playbackQueue = [];
  let isPlaying = false;
  let lastVadEnd = 0;
  let sessionLatency = 0;
  let currentSessionId = "";

  const $ = (sel) => document.querySelector(sel);
  const $all = (sel) => document.querySelectorAll(sel);

  let playbackCtx = null;

  function ensurePlaybackContext() {
    if (!playbackCtx) {
      playbackCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000,
      });
    }
    return playbackCtx;
  }

  function queueAudioChunk(base64Audio, format, isLast) {
    playbackQueue.push({ base64Audio, format, isLast });
    if (!isPlaying) playNextChunk();
  }

  async function playNextChunk() {
    if (playbackQueue.length === 0) {
      isPlaying = false;
      if (currentState === "speaking") setState("idle");
      return;
    }
    isPlaying = true;
    const chunk = playbackQueue.shift();

    try {
      const ctx = ensurePlaybackContext();
      const binaryStr = atob(chunk.base64Audio);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);

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

  function setState(newState) {
    const prev = currentState;
    if (prev === newState) return;
    currentState = newState;
    updateUI(newState, prev);
    log(`State: ${prev} → ${newState}`);
  }

  function updateLatencyUI(ms) {
    sessionLatency = ms;
    const el = $("#metricLatency");
    if (el) el.textContent = `⏱ ${ms}ms`;
  }

  function submitFeedback(rating) {
    const payload = {
      type: "feedback",
      rating: rating,
      latency_ms: sessionLatency,
      session_id: currentSessionId
    };
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
    log(`Feedback sent: ${rating}`);
    const panel = $("#feedbackPanel");
    if (panel) panel.classList.remove("visible");
  }

  function updateUI(state, prevState) {
    const statusBar = $(".ws-status-bar");
    if (statusBar) {
      statusBar.className = "ws-status-bar " + state;
      const labels = {
        idle: "🟢 Ready", listening: "🎤 Listening...", thinking: "🧠 Thinking...",
        speaking: "🔊 Speaking...", interrupted: "⚡ Interrupted", error: "❌ Error", connecting: "🔄 Connecting..."
      };
      statusBar.textContent = labels[state] || state;
    }

    const legacyMicOrb = $(".mic-orb");
    if (legacyMicOrb && !$("#voiceOrb")) {
      legacyMicOrb.classList.toggle("recording", state === "listening");
      legacyMicOrb.classList.toggle("thinking", state === "thinking");
      legacyMicOrb.classList.toggle("speaking", state === "speaking");
    }

    const pttBtn = $(".btn-ptt");
    if (pttBtn) pttBtn.classList.toggle("recording", state === "listening");

    // New UI Hooks
    const voiceOrb = $("#voiceOrb");
    if (voiceOrb) {
      voiceOrb.className = "voice-orb"; // reset
      if (state !== 'idle' && state !== 'error') voiceOrb.classList.add(state);
    }

    const connectBtnText = $("#btnConnectText");
    if (connectBtnText) {
      connectBtnText.textContent = state === "idle" ? "Disconnect" : (state === "connecting" ? "Connecting..." : "Live");
    }

    const wsStatusPill = $("#wsStatusPill");
    if (wsStatusPill) {
       wsStatusPill.innerHTML = (ws && ws.readyState === WebSocket.OPEN) ? "🟢 Online" : "⚫ Offline";
    }

    const feedbackPanel = $("#feedbackPanel");
    if (feedbackPanel) {
       if (state === "idle" && prevState === "speaking") {
          feedbackPanel.classList.add("visible");
       } else if (state === "listening" || state === "thinking") {
          feedbackPanel.classList.remove("visible"); 
       }
    }
  }

  let reconnectAttempts = 0;
  function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}${WS_PATH}?user_id=web_user&language=${language}`;

    setState("connecting");
    ws = new WebSocket(url);

    ws.onopen = () => {
      log("WebSocket connected");
      reconnectAttempts = 0;
      updateUI(currentState, currentState);
    };

    ws.onmessage = (event) => {
      try { handleServerMessage(JSON.parse(event.data)); }
      catch (err) { console.warn("[Duplex] Parse error:", err); }
    };

    ws.onclose = (ev) => {
      log(`WebSocket closed: ${ev.code}`);
      ws = null;
      setState("idle");
      if (ev.code !== 1000 && reconnectAttempts < 5) {
        const backoff = Math.pow(2, reconnectAttempts) * 1000;
        reconnectAttempts++;
        log(`Reconnecting in ${backoff}ms...`);
        setTimeout(connect, backoff);
      }
    };

    ws.onerror = (err) => {
      setState("error");
    };
  }

  function disconnect() {
    if (ws) {
      sendJSON({ type: "close" });
      ws.close(1000, "User disconnected");
      ws = null;
    }
    stopRecording();
    stopPlayback();
    setState("idle");
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "ready":
        currentSessionId = msg.session_id || "unknown";
        setState("idle");
        addChatBubble("system", "Session ready. Say something!");
        break;
      case "pipeline_state":
        if (msg.state === "thinking" && currentState === "listening") {
            lastVadEnd = performance.now();
        } else if (msg.state === "idle" && msg.latency_ms) {
            updateLatencyUI(Math.round(msg.latency_ms));
        }
        handlePipelineState(msg.state, msg);
        break;
      case "response_audio":
        if (currentState !== "speaking" && lastVadEnd > 0) {
            updateLatencyUI(Math.round(performance.now() - lastVadEnd));
            lastVadEnd = 0;
        }
        setState("speaking");
        queueAudioChunk(msg.audio_base64, msg.format, msg.is_last);
        break;
      case "response_sentence":
        addChatBubble("agent", msg.text);
        break;
      case "bargein":
        setState("interrupted");
        stopPlayback();
        setTimeout(() => setState("listening"), 200);
        break;
      case "error":
        setState("error"); break;
    }
  }

  function handlePipelineState(state, msg) {
    switch (state) {
      case "transcribing": setState("thinking"); break;
      case "thinking":
        setState("thinking");
        if (msg.text) addChatBubble("user", msg.text);
        break;
      case "speaking": setState("speaking"); break;
      case "interrupted": setState("interrupted"); break;
      case "idle": break;
    }
  }

  async function startRecording() {
    if (isRecording) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connect();
      await new Promise((resolve) => {
        const check = setInterval(() => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            clearInterval(check); resolve();
          }
        }, 100);
        setTimeout(() => { clearInterval(check); resolve(); }, 3000);
      });
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });

      audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      const source = audioContext.createMediaStreamSource(mediaStream);

      try {
        await audioContext.audioWorklet.addModule('./assets/js/voice-processor.js');
        workletNode = new AudioWorkletNode(audioContext, 'voice-processor');
        workletNode.port.onmessage = (e) => {
          if (!isRecording) return;
          const float32Data = e.data;
          const int16Data = float32ToInt16(float32Data);
          const base64 = arrayBufferToBase64(int16Data.buffer);
          sendJSON({ type: "audio_chunk", audio_base64: base64 });
        };
        source.connect(workletNode);
        workletNode.connect(audioContext.destination);
      } catch (err) {
        log("Worklet fallback due to error: " + err.message);
        fallbackProcessor = audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);
        fallbackProcessor.onaudioprocess = (e) => {
          if (!isRecording) return;
          const float32Data = e.inputBuffer.getChannelData(0);
          const int16Data = float32ToInt16(float32Data);
          const base64 = arrayBufferToBase64(int16Data.buffer);
          sendJSON({ type: "audio_chunk", audio_base64: base64 });
        };
        source.connect(fallbackProcessor);
        fallbackProcessor.connect(audioContext.destination);
      }

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
    sendJSON({ type: "audio_end" });

    if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (fallbackProcessor) { fallbackProcessor.disconnect(); fallbackProcessor = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    log("Recording stopped");
  }

  function triggerBargein() {
    if (currentState === "speaking") {
      stopPlayback();
      sendJSON({ type: "bargein" });
      setState("interrupted");
    }
  }

  function sendJSON(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
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
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  function addChatBubble(role, text) {
    const thread = $(".chat-thread");
    if (thread) {
      const wrap = document.createElement("div");
      wrap.className = `chat-bubble-wrap ${role}`;
      const bubble = document.createElement("div");
      bubble.className = `chat-bubble ${role}`;
      bubble.textContent = text;
      wrap.appendChild(bubble);
      thread.appendChild(wrap);
      thread.scrollTop = thread.scrollHeight;
    }

    if (role === 'user') {
       const userEl = $('#transcriptUser');
       if (userEl) userEl.textContent = text;
    } else if (role === 'agent' || role === 'system') {
       const agentEl = $('#transcriptAgent');
       if (agentEl) {
           agentEl.textContent = text;
       }
    }
  }

  function log(msg) {
    const timeline = $(".event-timeline");
    if (timeline) {
      const row = document.createElement("div");
      row.className = "event-row";
      row.innerHTML = `<span class="event-dot system"></span><span class="event-time">${new Date().toLocaleTimeString()}</span><span class="event-text">${msg}</span>`;
      timeline.appendChild(row);
      timeline.scrollTop = timeline.scrollHeight;
    }
    console.log(`[Duplex] ${msg}`);
  }

  function setLanguage(lang) {
    language = lang;
    sendJSON({ type: "language_hint", language: lang });
    log(`Language set to: ${lang}`);
  }

  function bindPTT(button) {
    if (!button) return;
    let isDown = false;
    const onDown = (e) => { e.preventDefault(); if (isDown) return; isDown = true; if (currentState === "speaking") triggerBargein(); startRecording(); };
    const onUp = (e) => { e.preventDefault(); if (!isDown) return; isDown = false; stopRecording(); };
    button.addEventListener("mousedown", onDown); button.addEventListener("mouseup", onUp); button.addEventListener("mouseleave", onUp);
    button.addEventListener("touchstart", onDown, { passive: false }); button.addEventListener("touchend", onUp, { passive: false });
  }

  function init() {
    bindPTT($(".btn-ptt"));
    
    // Legacy Mic Orb
    const orb = $(".mic-orb");
    if (orb) orb.addEventListener("click", () => {
      if (isRecording) stopRecording(); else { if (currentState === "speaking") triggerBargein(); startRecording(); }
    });

    // New Orb
    const voiceOrb = $("#voiceOrb");
    if (voiceOrb) voiceOrb.addEventListener("click", () => {
      if (isRecording) stopRecording(); else { if (currentState === "speaking") triggerBargein(); startRecording(); }
    });

    const langSelect = $(".duplex-lang-select") || $("#langSelect");
    if (langSelect) langSelect.addEventListener("change", (e) => setLanguage(e.target.value));

    const connectBtn = $(".duplex-connect-btn") || $("#btnConnectToggle");
    if (connectBtn) connectBtn.addEventListener("click", () => {
      if (ws && ws.readyState === WebSocket.OPEN) disconnect(); else connect();
    });

    const fbUp = $("#btnFeedbackUp");
    if (fbUp) fbUp.addEventListener("click", () => submitFeedback("up"));
    const fbDown = $("#btnFeedbackDown");
    if (fbDown) fbDown.addEventListener("click", () => submitFeedback("down"));

    log("Duplex client initialized");
  }

  window.CropFreshDuplex = {
    connect, disconnect, startRecording, stopRecording, triggerBargein, setLanguage, getState: () => currentState, init,
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
