const DEFAULT_WS_PATH = "/api/v1/voice/ws/duplex";

function buildFallbackWsUrl({ language, reconnectToken, sessionId, userId }) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({
    reconnect_token: reconnectToken,
    user_id: userId,
    language,
    session_id: sessionId,
  });

  return `${protocol}//${location.host}${DEFAULT_WS_PATH}?${params.toString()}`;
}

function resolveGatewayBaseUrl() {
  return document.body?.dataset.voiceGatewayUrl || window.CROPFRESH_VOICE_GATEWAY_URL || "";
}

export async function bootstrapVoiceSession({
  language,
  reconnectToken,
  sessionId,
  userId = "web_user",
}) {
  const resolvedReconnectToken = reconnectToken || crypto.randomUUID();
  const fallbackWsUrl = buildFallbackWsUrl({
    language,
    reconnectToken: resolvedReconnectToken,
    sessionId,
    userId,
  });
  const gatewayBaseUrl = resolveGatewayBaseUrl().trim().replace(/\/$/, "");

  // //! Until LiveKit media wiring lands, the static client always keeps a websocket fallback ready.
  if (!gatewayBaseUrl) {
    return {
      session_id: sessionId,
      mode: "fallback_ws",
      reconnect_token: resolvedReconnectToken,
      heartbeat_interval_ms: 10000,
      session_recovery_ttl_ms: 300000,
      fallback_ws_url: fallbackWsUrl,
      features: {
        fallback_enabled: true,
        livekit: false,
        rms_gate: false,
        ring_buffer: false,
        vad_service: false,
      },
      bootstrap_source: "local_fallback",
    };
  }

  try {
    const response = await fetch(`${gatewayBaseUrl}/sessions/bootstrap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        language,
        reconnect_token: resolvedReconnectToken,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`bootstrap failed with ${response.status}`);
    }

    const payload = await response.json();
    const resolvedFallbackUrl = new URL(payload.fallback_ws_url || fallbackWsUrl);
    resolvedFallbackUrl.searchParams.set("user_id", userId);
    resolvedFallbackUrl.searchParams.set("language", language);
    resolvedFallbackUrl.searchParams.set("session_id", payload.session_id || sessionId);
    resolvedFallbackUrl.searchParams.set(
      "reconnect_token",
      payload.reconnect_token || resolvedReconnectToken,
    );
    return {
      ...payload,
      session_id: payload.session_id || sessionId,
      reconnect_token: payload.reconnect_token || resolvedReconnectToken,
      fallback_ws_url: resolvedFallbackUrl.toString(),
      bootstrap_source: "gateway",
    };
  } catch (error) {
    return {
      session_id: sessionId,
      mode: "fallback_ws",
      reconnect_token: resolvedReconnectToken,
      heartbeat_interval_ms: 10000,
      session_recovery_ttl_ms: 300000,
      fallback_ws_url: fallbackWsUrl,
      features: {
        fallback_enabled: true,
        livekit: false,
        rms_gate: false,
        ring_buffer: false,
        vad_service: false,
      },
      bootstrap_error: error.message,
      bootstrap_source: "gateway_fallback",
    };
  }
}
