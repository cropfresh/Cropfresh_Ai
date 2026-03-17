const WS_PATH = "/api/v1/voice/ws/duplex";
const SOCKET_WAIT_MS = 3000;
const SOCKET_POLL_MS = 100;
const MAX_RECONNECT_ATTEMPTS = 5;

export function createDuplexSocket({
  getLanguage,
  onMessage,
  onLog,
  onStateChange,
  onError,
}) {
  let ws = null;
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

  function buildUrl() {
    const browserSessionId =
      sessionStorage.getItem("voice_duplex_session_id") || crypto.randomUUID();
    sessionStorage.setItem("voice_duplex_session_id", browserSessionId);
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${location.host}${WS_PATH}?user_id=web_user&language=${getLanguage()}&session_id=${browserSessionId}`;
  }

  function connect() {
    if ([WebSocket.OPEN, WebSocket.CONNECTING].includes(getReadyState())) {
      return;
    }

    userClosedSocket = false;
    onStateChange("connecting");
    ws = new WebSocket(buildUrl());
    ws.onopen = () => {
      reconnectAttempts = 0;
      onLog("WebSocket connected");
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

    connect();
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
    isOpen: () => getReadyState() === WebSocket.OPEN,
    sendJSON,
    waitForOpen,
  };
}
