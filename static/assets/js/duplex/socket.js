const SOCKET_WAIT_MS = 3000;
const SOCKET_POLL_MS = 100;
const MAX_RECONNECT_ATTEMPTS = 5;
const FALLBACK_HEARTBEAT_MS = 10000;

import { createHeartbeatWatchdog, getReconnectDelay, resolveRecoveryPolicy } from "./recovery.js";
import { resolveBootstrapSession } from "./session.js";

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
  let activeRecovery = resolveRecoveryPolicy();
  let heartbeatTimer = null;
  let reconnectAttempts = 0;
  let reconnectTimer = null;
  let userClosedSocket = false;

  const heartbeatWatchdog = createHeartbeatWatchdog({
    getTimeoutMs: () => activeRecovery.dead_peer_timeout_ms,
    onTimeout: () => {
      if (getReadyState() !== WebSocket.OPEN) return;
      heartbeatWatchdog.stop();
      onLog(`Dead peer detected after ${activeRecovery.dead_peer_timeout_ms}ms without heartbeat ack`);
      ws?.close(1011, "dead_peer_timeout");
    },
  });

  function getReadyState() { return ws?.readyState ?? WebSocket.CLOSED; }

  function sendJSON(payload) {
    if (getReadyState() === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }

  function stopHeartbeat() {
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  }

  function clearReconnectTimer() {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function startHeartbeat() {
    stopHeartbeat();
    const heartbeatIntervalMs = activeBootstrap?.heartbeat_interval_ms || FALLBACK_HEARTBEAT_MS;
    heartbeatTimer = setInterval(() => {
      sendJSON({ type: "heartbeat" });
    }, heartbeatIntervalMs);
  }

  async function resolveBootstrap(forceRefresh = false) {
    activeBootstrap = await resolveBootstrapSession({
      activeBootstrap,
      forceRefresh,
      getLanguage,
      onBootstrap,
    });
    activeRecovery = resolveRecoveryPolicy(activeBootstrap);
    return activeBootstrap;
  }

  function scheduleReconnect(reason, refreshBootstrap = false) {
    if (userClosedSocket || reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) return;
    const backoffMs = getReconnectDelay(activeRecovery, reconnectAttempts);
    reconnectAttempts += 1;
    clearReconnectTimer();
    onLog(`Reconnecting live session in ${backoffMs}ms (${reason})`);
    reconnectTimer = setTimeout(() => {
      void connect({
        refreshBootstrap:
          refreshBootstrap ||
          activeRecovery.network_change_recovery ||
          activeRecovery.ice_restart_enabled,
      });
    }, backoffMs);
  }

  async function connect({ refreshBootstrap = false } = {}) {
    if ([WebSocket.OPEN, WebSocket.CONNECTING].includes(getReadyState())) return;
    const bootstrap = await resolveBootstrap(refreshBootstrap || reconnectAttempts > 0);
    userClosedSocket = false;
    onStateChange("connecting");
    ws = new WebSocket(bootstrap.fallback_ws_url);

    ws.onopen = () => {
      clearReconnectTimer();
      reconnectAttempts = 0;
      heartbeatWatchdog.start();
      startHeartbeat();
      const modeLabel = bootstrap.mode === "bridge" ? "bridge bootstrap" : "fallback websocket";
      onLog(`WebSocket connected via ${modeLabel}`);
      if (bootstrap.mode === "bridge" && activeRecovery.ice_restart_enabled) {
        onLog("Bridge recovery contract is active; bootstrap will refresh on reconnects.");
      }
      onStateChange("connecting");
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "heartbeat_ack") {
          heartbeatWatchdog.markAck();
        }
        onMessage(message);
      } catch (error) {
        onLog(`Message parse error: ${error?.message || "unknown"}`);
      }
    };

    ws.onclose = (event) => {
      stopHeartbeat();
      heartbeatWatchdog.stop();
      ws = null;
      onStateChange("idle");
      onLog(`WebSocket closed: ${event.code}`);
      if (!userClosedSocket && event.code !== 1000) {
        scheduleReconnect(event.reason || `code ${event.code}`, true);
      }
    };

    ws.onerror = () => {
      onError();
      onStateChange("error");
    };
  }

  async function waitForOpen() {
    if (getReadyState() === WebSocket.OPEN) return true;
    await connect();
    const deadline = Date.now() + SOCKET_WAIT_MS;
    while (Date.now() < deadline) {
      if (getReadyState() === WebSocket.OPEN) return true;
      await new Promise((resolve) => setTimeout(resolve, SOCKET_POLL_MS));
    }
    return false;
  }

  function disconnect() {
    userClosedSocket = true;
    clearReconnectTimer();
    stopHeartbeat();
    heartbeatWatchdog.stop();
    activeBootstrap = null;
    if (ws) {
      sendJSON({ type: "close" });
      ws.close(1000, "User disconnected");
      ws = null;
    }
    onStateChange("idle");
  }

  window.addEventListener("offline", () => {
    stopHeartbeat();
    heartbeatWatchdog.stop();
    onLog("Network offline, holding reconnect until the browser comes back online.");
  });
  window.addEventListener("online", () => {
    if (userClosedSocket || !activeRecovery.network_change_recovery || getReadyState() === WebSocket.OPEN) return;
    onLog("Network restored, reconnecting the live session.");
    void connect({ refreshBootstrap: true });
  });

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
