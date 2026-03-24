let wsSocket = null;
let wsSessionActive = false;
let wsAgentMsgEl = null;
let wsBootstrap = null;
let wsHeartbeatTimer = null;

const wsPlayer = new AudioPlayer();

function wsSetBridgeMode(bootstrap) {
  const badge = document.getElementById("wsBridgeModeBadge");
  if (!badge) return;

  badge.textContent =
    bootstrap?.mode === "bridge" ? "Bridge bootstrap active" : "Direct websocket fallback";
}

function wsStopHeartbeat() {
  if (wsHeartbeatTimer !== null) {
    clearInterval(wsHeartbeatTimer);
    wsHeartbeatTimer = null;
  }
}

function wsStartHeartbeat() {
  wsStopHeartbeat();
  const heartbeatIntervalMs = wsBootstrap?.heartbeat_interval_ms || 10000;
  wsHeartbeatTimer = setInterval(() => {
    if (wsSocket?.readyState === WebSocket.OPEN) {
      wsSocket.send(JSON.stringify({ type: "heartbeat" }));
    }
  }, heartbeatIntervalMs);
}

async function wsResolveBootstrap(userId, language) {
  const sessionId =
    sessionStorage.getItem("voice_hub_session_id") || crypto.randomUUID();
  const reconnectToken =
    sessionStorage.getItem("voice_hub_reconnect_token") || crypto.randomUUID();
  sessionStorage.setItem("voice_hub_session_id", sessionId);
  sessionStorage.setItem("voice_hub_reconnect_token", reconnectToken);

  const helper = window.CropfreshVoiceBootstrap;
  if (!helper?.bootstrapSession) {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    return {
      session_id: sessionId,
      mode: "fallback_ws",
      reconnect_token: reconnectToken,
      heartbeat_interval_ms: 10000,
      fallback_ws_url:
        `${protocol}://${location.host}/api/v1/voice/ws/duplex?` +
        new URLSearchParams({
          reconnect_token: reconnectToken,
          user_id: userId,
          language,
          session_id: sessionId,
        }).toString(),
    };
  }

  const bootstrap = await helper.bootstrapSession({
    language,
    reconnectToken,
    sessionId,
    userId,
  });
  if (bootstrap?.session_id) {
    sessionStorage.setItem("voice_hub_session_id", bootstrap.session_id);
  }
  if (bootstrap?.reconnect_token) {
    sessionStorage.setItem("voice_hub_reconnect_token", bootstrap.reconnect_token);
  }
  return bootstrap;
}

async function wsConnect() {
  if (wsSocket && wsSocket.readyState === WebSocket.OPEN) return;

  window.VoiceHubLab?.resetWsWorkflow();
  const userId = document.getElementById("wsUserId")?.value || "ws-test-user";
  const language = document.getElementById("wsLang")?.value || "hi";

  wsSetStatus("Connecting...", "connecting");
  wsBootstrap = await wsResolveBootstrap(userId, language);
  wsSetBridgeMode(wsBootstrap);
  wsSocket = new WebSocket(wsBootstrap.fallback_ws_url);

  wsSocket.onopen = () => {
    wsSessionActive = true;
    wsStartHeartbeat();
    setStatusPill("wsStatusPill", true, "Connected");
    const modeLabel =
      wsBootstrap?.mode === "bridge" ? "bridge bootstrap" : "fallback websocket";
    wsAddEvent("system", `Connected via ${modeLabel} - user=${userId}`);
    wsSetStatus("Connected - hold PTT to speak", "connected");

    document.getElementById("btnWsConnect")?.setAttribute("disabled", "");
    document.getElementById("btnWsDisconnect")?.removeAttribute("disabled");
    document.getElementById("btnWsPtt")?.removeAttribute("disabled");
  };

  wsSocket.onclose = () => {
    wsSessionActive = false;
    wsStopHeartbeat();
    setStatusPill("wsStatusPill", false, "Disconnected");
    wsAddEvent("system", "Disconnected");
    wsSetStatus("Disconnected", "");
    wsStopMic(false);
    document.getElementById("btnWsConnect")?.removeAttribute("disabled");
    document.getElementById("btnWsDisconnect")?.setAttribute("disabled", "");
    document.getElementById("btnWsPtt")?.setAttribute("disabled", "");
  };

  wsSocket.onerror = () => {
    wsAddEvent("system", "WebSocket error");
    wsSetStatus("Error", "error");
  };

  wsSocket.onmessage = wsHandleMessage;
}

function wsDisconnect() {
  wsStopMic(false);
  wsStopHeartbeat();
  window.VoiceHubLab?.resetWsWorkflow();
  wsSocket?.close();
}

function wsHandleMessage(event) {
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }

  window.VoiceHubLab?.handleWsMessage(msg);

  switch (msg.type) {
    case "ready":
      wsAddEvent("system", `Ready (session: ${msg.session_id})`);
      break;
    case "language_detected": {
      wsAddEvent(
        "system",
        `Detected language: ${msg.language.toUpperCase()}${msg.locked ? " (locked)" : ""}`,
      );
      const langBadge = document.getElementById("wsDetectedLangBadge");
      if (langBadge) {
        langBadge.textContent = msg.language.toUpperCase();
        langBadge.classList.remove("d-none");
      }
      if (msg.locked) {
        const langSelect = document.getElementById("wsLang");
        if (langSelect && langSelect.value !== msg.language) {
          langSelect.value = msg.language;
        }
      }
      break;
    }
    case "pipeline_state":
      if (msg.state === "listening") wsSetStatus("Listening...", "listening");
      else if (msg.state === "thinking") wsSetStatus("Thinking...", "thinking");
      else if (msg.state === "speaking") wsSetStatus("Speaking...", "speaking");
      break;
    case "transcript_final":
      wsAppendBubble("user", msg.text);
      wsAddEvent("vad", `You: ${msg.text}`);
      break;
    case "response_sentence":
      if (!wsAgentMsgEl) {
        wsAgentMsgEl = wsAppendBubble("agent", `${msg.text} `);
      } else {
        wsUpdateBubble(wsAgentMsgEl, `${msg.text} `);
      }
      wsAddEvent("response", msg.text);
      break;
    case "response_audio":
      if (msg.audio_base64) {
        wsPlayer.enqueue(msg.audio_base64, "audio/mpeg");
      }
      break;
    case "response_end":
      if (msg.full_text && wsAgentMsgEl) {
        wsAgentMsgEl.textContent = msg.full_text;
      }
      wsAddEvent("response", "Done");
      wsSetStatus("Ready - hold PTT to speak", "connected");
      wsAgentMsgEl = null;
      break;
    case "bargein":
      wsAddEvent("bargein", "Barge-in - response cancelled");
      wsPlayer.clearQueue();
      wsSetStatus("Listening...", "listening");
      wsAgentMsgEl = null;
      break;
    case "heartbeat_ack":
      wsAddEvent("system", `Heartbeat ack (${msg.heartbeat_interval_ms || 0}ms)`);
      break;
    case "error":
      wsAddEvent("system", `${msg.error || msg.text}`);
      wsAppendBubble("system", `Error: ${msg.error || msg.text}`);
      wsSetStatus("Ready - hold PTT to speak", "connected");
      break;
    default:
      if (msg.type) wsAddEvent("system", `? ${msg.type}`);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btnWsConnect")?.addEventListener("click", () => {
    wsConnect().catch((error) => wsAddEvent("system", error.message));
  });
  document.getElementById("btnWsDisconnect")?.addEventListener("click", wsDisconnect);

  const pttBtn = document.getElementById("btnWsPtt");
  if (pttBtn) {
    pttBtn.addEventListener("mousedown", () =>
      wsStartMic().catch((error) => wsAddEvent("system", error.message)),
    );
    pttBtn.addEventListener("mouseup", () => wsStopMic(true));
    pttBtn.addEventListener("mouseleave", () => {
      if (wsPttActive) wsStopMic(true);
    });
    pttBtn.addEventListener("touchstart", (event) => {
      event.preventDefault();
      wsStartMic().catch((error) => wsAddEvent("system", error.message));
    });
    pttBtn.addEventListener("touchend", (event) => {
      event.preventDefault();
      wsStopMic(true);
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.code === "Space" && event.target === document.body && wsSessionActive && !wsPttActive) {
      event.preventDefault();
      wsStartMic().catch(() => {});
    }
  });
  document.addEventListener("keyup", (event) => {
    if (event.code === "Space" && wsPttActive) {
      event.preventDefault();
      wsStopMic(true);
    }
  });

  document.getElementById("btnWsClearTimeline")?.addEventListener("click", () => {
    const timeline = document.getElementById("wsTimeline");
    if (timeline) timeline.innerHTML = "";
    const chat = wsGetChat();
    if (chat) chat.innerHTML = "";
    window.VoiceHubLab?.resetWsWorkflow();
  });
});
