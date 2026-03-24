import { bootstrapVoiceSession } from "./bootstrap.js";

export async function resolveBootstrapSession({
  activeBootstrap,
  forceRefresh = false,
  getLanguage,
  onBootstrap,
}) {
  const browserSessionId =
    sessionStorage.getItem("voice_duplex_session_id") || crypto.randomUUID();
  const reconnectToken =
    sessionStorage.getItem("voice_duplex_reconnect_token") || crypto.randomUUID();
  sessionStorage.setItem("voice_duplex_session_id", browserSessionId);
  sessionStorage.setItem("voice_duplex_reconnect_token", reconnectToken);

  if (!forceRefresh && activeBootstrap?.session_id === browserSessionId) {
    return activeBootstrap;
  }

  const nextBootstrap = await bootstrapVoiceSession({
    language: getLanguage(),
    reconnectToken,
    sessionId: browserSessionId,
    userId: "web_user",
  });
  if (nextBootstrap.reconnect_token) {
    sessionStorage.setItem("voice_duplex_reconnect_token", nextBootstrap.reconnect_token);
  }
  onBootstrap(nextBootstrap);
  return nextBootstrap;
}
