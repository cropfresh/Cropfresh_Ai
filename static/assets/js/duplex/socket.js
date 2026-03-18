const SOCKET_WAIT_MS = 3000;
const SOCKET_POLL_MS = 100;
const MAX_RECONNECT_ATTEMPTS = 5;
const FALLBACK_HEARTBEAT_MS = 10000;

import { bootstrapVoiceSession } from "./bootstrap.js";

export function createDuplexSocket({
  getLanguage,
  onBootstrap,
  onMessage,
  onLog,
  onStateChange,
  onError,
}) {
  let ws = null;
  let activeBootstrap = null;
  let heartbeatTimer = null;
  let reconnectAttempts = 0;
  let userClosedSocket = false;

  function getReadyState() {
    return ws?.readyState ?? WebSocket.CLOSED;
  }

  function sendJSON(payload) {
    if (getReadyState() === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function stopHeartbeat() {
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  }

  function startHeartbeat() {
    stopHeartbeat();
    const heartbeatIntervalMs =
      activeBootstrap?.heartbeat_interval_ms || FALLBACK_HEARTBEAT_MS;
    heartbeatTimer = setInterval(() => {
      sendJSON({ type: "heartbeat" });
    }, heartbeatIntervalMs);
  }

  async function resolveBootstrap() {
    const browserSessionId =
      sessionStorage.getItem("voice_duplex_session_id") || crypto.randomUUID();
    const reconnectToken =
      sessionStorage.getItem("voice_duplex_reconnect_token") || crypto.randomUUID();
    sessionStorage.setItem("voice_duplex_session_id", browserSessionId);
    sessionStorage.setItem("voice_duplex_reconnect_token", reconnectToken);

    if (activeBootstrap && activeBootstrap.session_id === browserSessionId) {
      return activeBootstrap;
    }

    activeBootstrap = await bootstrapVoiceSession({
      language: getLanguage(),
      reconnectToken,
      sessionId: browserSessionId,
      userId: "web_user",
    });
    if (activeBootstrap.reconnect_token) {
      sessionStorage.setItem("voice_duplex_reconnect_token", activeBootstrap.reconnect_token);
    }
    onBootstrap(activeBootstrap);
    return activeBootstrap;
  }

  async function connect() {
    if ([WebSocket.OPEN, WebSocket.CONNECTING].includes(getReadyState())) {
      return;
    }

    const bootstrap = await resolveBootstrap();
    userClosedSocket = false;
    onStateChange("connecting");
    ws = new WebSocket(bootstrap.fallback_ws_url);
    ws.onopen = () => {
      reconnectAttempts = 0;
      startHeartbeat();
      const modeLabel = bootstrap.mode === "bridge" ? "bridge bootstrap" : "fallback websocket";
      onLog(`WebSocket connected via ${modeLabel}`);
      onStateChange("connecting");
    };
    ws.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch (error) {
        onLog(`Message parse error: ${error.message}`);
      }
    };
    ws.onclose = (event) => {
      stopHeartbeat();
      ws = null;
      onStateChange("idle");
      onLog(`WebSocket closed: ${event.code}`);
      if (!userClosedSocket && event.code !== 1000 && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        const backoffMs = 2 ** reconnectAttempts * 1000;
        reconnectAttempts += 1;
        setTimeout(connect, backoffMs);
      }
    };
    ws.onerror = () => {
      onError();
      onStateChange("error");
    };
  }

  async function waitForOpen() {
    if (getReadyState() === WebSocket.OPEN) {
      return true;
    }

    await connect();
    const deadline = Date.now() + SOCKET_WAIT_MS;
    while (Date.now() < deadline) {
      if (getReadyState() === WebSocket.OPEN) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, SOCKET_POLL_MS));
    }
    return false;
  }

  function disconnect() {
    userClosedSocket = true;
    stopHeartbeat();
    activeBootstrap = null;
    if (ws) {
      sendJSON({ type: "close" });
      ws.close(1000, "User disconnected");
      ws = null;
    }
    onStateChange("idle");
  }

  return {
    connect,
    disconnect,
    getReadyState,
    getBootstrap: () => activeBootstrap,
    isOpen: () => getReadyState() === WebSocket.OPEN,
    sendJSON,
    waitForOpen,
  };
}
