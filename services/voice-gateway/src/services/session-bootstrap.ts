import { randomBytes, randomUUID } from "node:crypto";

import { AccessToken } from "livekit-server-sdk";

import { VoiceGatewayConfig, isLiveKitConfigured } from "../config.js";
import { bootstrapRequestsTotal } from "../metrics.js";

export interface BootstrapRequest {
  userId?: string;
  language?: string;
  requestedMode?: string;
  sessionId?: string;
  reconnectToken?: string;
}

export interface BootstrapFeatures {
  fallback_enabled: boolean;
  livekit: boolean;
  rms_gate: boolean;
  ring_buffer: boolean;
  vad_service: boolean;
}

export interface BootstrapRecoveryPolicy {
  dead_peer_timeout_ms: number;
  ice_restart_enabled: boolean;
  network_change_recovery: boolean;
  reconnect_token_required: boolean;
  retry_backoff_ms: number[];
}

export interface BootstrapResponse {
  session_id: string;
  mode: "bridge" | "fallback_ws";
  livekit_url?: string;
  token?: string;
  reconnect_token: string;
  heartbeat_interval_ms: number;
  session_recovery_ttl_ms: number;
  fallback_ws_url: string;
  features: BootstrapFeatures;
  recovery: BootstrapRecoveryPolicy;
}

export class SessionBootstrapService {
  constructor(private readonly config: VoiceGatewayConfig) {}

  async createSession(request: BootstrapRequest): Promise<BootstrapResponse> {
    const sessionId = request.sessionId?.trim() || randomUUID();
    const userId = request.userId?.trim() || "web_user";
    const reconnectToken = request.reconnectToken?.trim() || randomBytes(24).toString("base64url");
    const features = this.buildFeatures();
    const recovery = this.buildRecoveryPolicy(features.livekit);
    const liveKitReady = this.config.enableLiveKitBridge && isLiveKitConfigured(this.config);

    if (request.requestedMode === "fallback_ws" || !liveKitReady) {
      bootstrapRequestsTotal.inc({ mode: "fallback_ws", outcome: "ok" });
      return {
        session_id: sessionId,
        mode: "fallback_ws",
        reconnect_token: reconnectToken,
        heartbeat_interval_ms: this.config.heartbeatIntervalMs,
        session_recovery_ttl_ms: this.config.sessionRecoveryTtlMs,
        fallback_ws_url: this.config.fallbackWsUrl,
        features,
        recovery,
      };
    }

    try {
      const token = await this.createLiveKitToken(sessionId, userId, request.language ?? "hi");
      bootstrapRequestsTotal.inc({ mode: "bridge", outcome: "ok" });
      return {
        session_id: sessionId,
        mode: "bridge",
        livekit_url: this.config.liveKitUrl,
        token,
        reconnect_token: reconnectToken,
        heartbeat_interval_ms: this.config.heartbeatIntervalMs,
        session_recovery_ttl_ms: this.config.sessionRecoveryTtlMs,
        fallback_ws_url: this.config.fallbackWsUrl,
        features,
        recovery,
      };
    } catch (error) {
      bootstrapRequestsTotal.inc({ mode: "bridge", outcome: "fallback" });
      return {
        session_id: sessionId,
        mode: "fallback_ws",
        reconnect_token: reconnectToken,
        heartbeat_interval_ms: this.config.heartbeatIntervalMs,
        session_recovery_ttl_ms: this.config.sessionRecoveryTtlMs,
        fallback_ws_url: this.config.fallbackWsUrl,
        features,
        recovery: this.buildRecoveryPolicy(false),
      };
    }
  }

  isReady(): boolean {
    return Boolean(this.config.fallbackWsUrl);
  }

  isBridgeModeAvailable(): boolean {
    return this.config.enableLiveKitBridge && isLiveKitConfigured(this.config);
  }

  private buildFeatures(): BootstrapFeatures {
    return {
      fallback_enabled: true,
      livekit: this.isBridgeModeAvailable(),
      rms_gate: true,
      ring_buffer: true,
      vad_service: Boolean(this.config.vadServiceBaseUrl),
    };
  }

  private buildRecoveryPolicy(livekitEnabled: boolean): BootstrapRecoveryPolicy {
    return {
      dead_peer_timeout_ms: this.config.deadPeerTimeoutMs,
      ice_restart_enabled: livekitEnabled,
      network_change_recovery: true,
      reconnect_token_required: true,
      retry_backoff_ms: [...this.config.reconnectBackoffMs],
    };
  }

  private async createLiveKitToken(
    sessionId: string,
    userId: string,
    language: string,
  ): Promise<string> {
    const token = new AccessToken(this.config.liveKitApiKey, this.config.liveKitApiSecret, {
      identity: userId,
      ttl: this.config.liveKitTokenTtl,
      metadata: JSON.stringify({ language, session_id: sessionId }),
      name: `voice-${userId}`,
    });

    // The initial bridge grant is intentionally narrow until LiveKit media handling lands.
    token.addGrant({
      roomJoin: true,
      room: sessionId,
      canPublish: true,
      canSubscribe: true,
    });

    return token.toJwt();
  }
}
