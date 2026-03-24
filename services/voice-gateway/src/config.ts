export interface VoiceGatewayConfig {
  host: string;
  port: number;
  corsOrigin: string;
  serviceVersion: string;
  enableLiveKitBridge: boolean;
  fallbackWsUrl: string;
  liveKitUrl: string;
  liveKitApiKey: string;
  liveKitApiSecret: string;
  liveKitTokenTtl: string;
  vadServiceBaseUrl: string;
  heartbeatIntervalMs: number;
  deadPeerTimeoutMs: number;
  reconnectBackoffMs: number[];
  sessionRecoveryTtlMs: number;
  continuityWindowMs: number;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value == null) {
    return fallback;
  }

  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (value == null) {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseNumberList(value: string | undefined, fallback: number[]): number[] {
  if (value == null) {
    return fallback;
  }

  const parsed = value
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item) && item > 0);
  return parsed.length > 0 ? parsed : fallback;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): VoiceGatewayConfig {
  return {
    host: env.VOICE_GATEWAY_HOST ?? "0.0.0.0",
    port: parseNumber(env.VOICE_GATEWAY_PORT, 3101),
    corsOrigin: env.VOICE_GATEWAY_CORS_ORIGIN ?? "*",
    serviceVersion: env.VOICE_GATEWAY_VERSION ?? "0.1.0",
    enableLiveKitBridge: parseBoolean(env.VOICE_GATEWAY_ENABLE_LIVEKIT_BRIDGE, false),
    fallbackWsUrl:
      env.VOICE_GATEWAY_FALLBACK_WS_URL ?? "ws://localhost:8000/api/v1/voice/ws/duplex",
    liveKitUrl: env.LIVEKIT_WS_URL ?? "",
    liveKitApiKey: env.LIVEKIT_API_KEY ?? "",
    liveKitApiSecret: env.LIVEKIT_API_SECRET ?? "",
    liveKitTokenTtl: env.VOICE_GATEWAY_LIVEKIT_TTL ?? "15m",
    vadServiceBaseUrl: env.VOICE_GATEWAY_VAD_SERVICE_BASE_URL ?? "http://localhost:8101",
    heartbeatIntervalMs: parseNumber(env.VOICE_GATEWAY_HEARTBEAT_INTERVAL_MS, 10_000),
    deadPeerTimeoutMs: parseNumber(env.VOICE_GATEWAY_DEAD_PEER_TIMEOUT_MS, 30_000),
    reconnectBackoffMs: parseNumberList(
      env.VOICE_GATEWAY_RECONNECT_BACKOFF_MS,
      [1_000, 2_000, 4_000, 8_000, 16_000],
    ),
    sessionRecoveryTtlMs: parseNumber(env.VOICE_GATEWAY_SESSION_RECOVERY_TTL_MS, 300_000),
    continuityWindowMs: parseNumber(env.VOICE_GATEWAY_CONTINUITY_WINDOW_MS, 400),
  };
}

export function isLiveKitConfigured(config: VoiceGatewayConfig): boolean {
  return Boolean(config.liveKitUrl && config.liveKitApiKey && config.liveKitApiSecret);
}
