import { createDuplexAudio } from "./audio.js";
import { bindDuplexControls } from "./controls.js";
import { createDuplexSocket } from "./socket.js";
import { createDuplexUI } from "./ui.js";

export function createDuplexApp() {
  let currentState = "idle";
  let language = "hi";
  let currentSessionId = "";
  let sessionTiming = {};
  let fallbackStart = 0;
  let socket = null;

  const ui = createDuplexUI({
    getSocketState: () => socket?.getReadyState() ?? WebSocket.CLOSED,
  });
  socket = createDuplexSocket({
    getLanguage: () => language,
    onBootstrap: (bootstrap) => {
      ui.updateTransport(bootstrap);
      if (bootstrap.bootstrap_error) {
        ui.log(`Gateway bootstrap fell back to websocket: ${bootstrap.bootstrap_error}`);
        return;
      }
      ui.log(
        bootstrap.mode === "bridge"
          ? "Gateway bootstrap succeeded. Static client is still using websocket relay for Sprint 08."
          : "Gateway bootstrap selected direct websocket fallback.",
      );
    },
    onError: () => ui.log("WebSocket error"),
    onLog: ui.log,
    onMessage: handleServerMessage,
    onStateChange: setState,
  });
  const audio = createDuplexAudio({
    onAudioChunk: (audioBase64) =>
      socket.sendJSON({ type: "audio_chunk", audio_base64: audioBase64 }),
    onError: (error) => {
      ui.log(`Microphone error: ${error.message}`);
      setState("error");
    },
    onPlaybackIdle: () => {
      if (currentState === "speaking") {
        setState("idle");
      }
    },
  });

  function setState(nextState) {
    const previousState = currentState;
    currentState = nextState;
    ui.updateState(nextState, previousState);
  }

  function updateTiming(timing = {}) {
    sessionTiming = { ...sessionTiming, ...timing };
    ui.updateMetrics(sessionTiming);
  }

  function connect() { socket.connect(); }
  async function waitForSocket() { return socket.waitForOpen(); }

  async function startRecording() {
    if (audio.isRecording()) {
      return;
    }
    if (currentState === "speaking") {
      triggerBargein();
    }

    const socketReady = await waitForSocket();
    if (!socketReady) {
      setState("error");
      ui.log("Timed out waiting for WebSocket connection");
      return;
    }

    await audio.startCapture();
    setState("listening");
    ui.log("Recording started");
  }

  async function stopRecording() {
    if (!audio.isRecording()) {
      return;
    }
    fallbackStart = performance.now();
    await audio.stopCapture();
    socket.sendJSON({ type: "audio_end" });
    ui.log("Recording stopped");
  }

  function triggerBargein() {
    if (currentState !== "speaking") {
      return;
    }
    audio.stopPlayback();
    socket.sendJSON({ type: "bargein" });
    setState("interrupted");
  }

  function disconnect() {
    if (audio.isRecording()) {
      void stopRecording();
    }
    audio.stopPlayback();
    socket.disconnect();
  }

  function handlePipelineState(state, message) {
    if (state === "thinking" && message.text) {
      ui.addChatBubble("user", message.text);
    }
    if (["transcribing", "thinking"].includes(state)) {
      setState("thinking");
      return;
    }
    if (state === "speaking" || state === "interrupted") {
      setState(state);
    }
  }

  function handleServerMessage(message) {
    if (message.timing) updateTiming(message.timing);

    switch (message.type) {
      case "ready":
        currentSessionId = message.session_id || "unknown";
        if (message.session_id) {
          sessionStorage.setItem("voice_duplex_session_id", message.session_id);
        }
        setState("idle");
        if (message.recovered) {
          ui.addChatBubble(
            "system",
            `Session recovered with ${message.recovered_turn_count || 0} saved turns.`,
          );
        } else {
          ui.addChatBubble("system", "Session ready. Say something!");
        }
        break;
      case "pipeline_state":
        handlePipelineState(message.state, message);
        break;
      case "response_audio":
        if (!message.timing && fallbackStart > 0) {
          updateTiming({ first_audio_ms: performance.now() - fallbackStart });
        }
        fallbackStart = 0;
        setState("speaking");
        audio.queuePlayback(message.audio_base64);
        break;
      case "response_sentence":
        ui.addChatBubble("agent", message.text);
        break;
      case "bargein":
        audio.stopPlayback();
        setState("interrupted");
        setTimeout(() => setState("listening"), 200);
        break;
      case "heartbeat_ack":
        ui.log(
          `Heartbeat ack (${message.heartbeat_interval_ms || 0}ms / recovery ${message.session_recovery_ttl_ms || 0}ms)`,
        );
        break;
      case "error":
        ui.log(`Server error: ${message.error || "unknown"}`);
        setState("error");
        break;
      default:
        break;
    }
  }

  function setLanguage(nextLanguage) {
    language = nextLanguage;
    socket.sendJSON({ type: "language_hint", language: nextLanguage });
    ui.log(`Language set to: ${nextLanguage}`);
  }

  function submitFeedback(rating) {
    socket.sendJSON({
      type: "feedback",
      rating,
      latency_ms: sessionTiming.total_ms ?? sessionTiming.first_audio_ms ?? 0,
      session_id: currentSessionId,
    });
    ui.hideFeedback();
    ui.log(`Feedback sent: ${rating}`);
  }

  function init() {
    bindDuplexControls({
      isRecording: audio.isRecording,
      isSocketOpen: socket.isOpen,
      onConnectToggle: (isOpen) => (isOpen ? disconnect() : connect()),
      onFeedbackDown: () => submitFeedback("down"),
      onFeedbackUp: () => submitFeedback("up"),
      onLanguageChange: setLanguage,
      onOrbToggle: (isRecording) =>
        isRecording ? void stopRecording() : void startRecording(),
      query: ui.query,
    });
    ui.updateMetrics();
    ui.updateTransport();
    ui.updateState(currentState, currentState);
    ui.log("Duplex client initialized");
  }

  return {
    connect,
    disconnect,
    getState: () => currentState,
    init,
    setLanguage,
    startRecording: () => void startRecording(),
    stopRecording: () => void stopRecording(),
    triggerBargein,
  };
}
