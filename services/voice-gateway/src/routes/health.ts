import type { Express, Request, Response } from "express";

import { SessionBootstrapService } from "../services/session-bootstrap.js";
import { VoiceGatewayConfig, isLiveKitConfigured } from "../config.js";
import { metricsRegistry } from "../metrics.js";

export function registerHealthRoutes(
  app: Express,
  config: VoiceGatewayConfig,
  bootstrapService: SessionBootstrapService,
): void {
  app.get("/health", (_request: Request, response: Response) => {
    response.json({
      status: "healthy",
      service: "voice-gateway",
      version: config.serviceVersion,
      livekit_configured: isLiveKitConfigured(config),
      bridge_mode_enabled: config.enableLiveKitBridge,
      fallback_ws_url: config.fallbackWsUrl,
      heartbeat_interval_ms: config.heartbeatIntervalMs,
      session_recovery_ttl_ms: config.sessionRecoveryTtlMs,
    });
  });

  app.get("/ready", (_request: Request, response: Response) => {
    const ready = bootstrapService.isReady();
    response.status(ready ? 200 : 503).json({
      status: ready ? "ready" : "not_ready",
      ready,
      bridge_mode_available: bootstrapService.isBridgeModeAvailable(),
      fallback_ws_configured: Boolean(config.fallbackWsUrl),
      vad_service_url: config.vadServiceBaseUrl,
      continuity_window_ms: config.continuityWindowMs,
    });
  });

  app.get("/metrics", async (_request: Request, response: Response) => {
    response.setHeader("Content-Type", metricsRegistry.contentType);
    response.send(await metricsRegistry.metrics());
  });
}
