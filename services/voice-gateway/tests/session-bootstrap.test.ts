import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { buildApp } from "../src/app.js";
import { loadConfig } from "../src/config.js";
import { SessionBootstrapService } from "../src/services/session-bootstrap.js";

const toJwt = vi.fn(async () => "signed-livekit-token");
const addGrant = vi.fn();

vi.mock("livekit-server-sdk", () => ({
  AccessToken: class {
    constructor() {}

    addGrant(grant: unknown) {
      addGrant(grant);
    }

    toJwt() {
      return toJwt();
    }
  },
}));

describe("voice gateway bootstrap routes", () => {
  beforeEach(() => {
    addGrant.mockClear();
    toJwt.mockClear();
  });

  it("returns fallback mode when bridge mode is disabled", async () => {
    const config = loadConfig({
      VOICE_GATEWAY_ENABLE_LIVEKIT_BRIDGE: "false",
      VOICE_GATEWAY_FALLBACK_WS_URL: "ws://localhost:8000/api/v1/voice/ws/duplex",
    });

    const { app } = buildApp(config);
    const response = await request(app).post("/sessions/bootstrap").send({
      language: "kn",
      session_id: "session-123",
    });

    expect(response.status).toBe(200);
    expect(response.body.mode).toBe("fallback_ws");
    expect(response.body.fallback_ws_url).toContain("/api/v1/voice/ws/duplex");
    expect(response.body.features.livekit).toBe(false);
    expect(response.body.reconnect_token).toBeTruthy();
    expect(response.body.heartbeat_interval_ms).toBe(10000);
    expect(response.body.recovery.dead_peer_timeout_ms).toBe(30000);
    expect(response.body.recovery.retry_backoff_ms).toEqual([1000, 2000, 4000, 8000, 16000]);
    expect(response.body.recovery.ice_restart_enabled).toBe(false);
    expect(response.body.session_recovery_ttl_ms).toBe(300000);
  });

  it("returns bridge mode when LiveKit is configured", async () => {
    const config = loadConfig({
      VOICE_GATEWAY_ENABLE_LIVEKIT_BRIDGE: "true",
      VOICE_GATEWAY_FALLBACK_WS_URL: "ws://localhost:8000/api/v1/voice/ws/duplex",
      LIVEKIT_API_KEY: "key",
      LIVEKIT_API_SECRET: "secret",
      LIVEKIT_WS_URL: "ws://localhost:7880",
    });

    const { app } = buildApp(config);
    const response = await request(app).post("/sessions/bootstrap").send({
      language: "hi",
      session_id: "bridge-session",
      user_id: "farmer-1",
    });

    expect(response.status).toBe(200);
    expect(response.body.mode).toBe("bridge");
    expect(response.body.livekit_url).toBe("ws://localhost:7880");
    expect(response.body.token).toBe("signed-livekit-token");
    expect(response.body.reconnect_token).toBeTruthy();
    expect(response.body.recovery.ice_restart_enabled).toBe(true);
    expect(addGrant).toHaveBeenCalledTimes(1);
  });

  it("falls back cleanly when token generation fails", async () => {
    toJwt.mockRejectedValueOnce(new Error("token error"));

    const config = loadConfig({
      VOICE_GATEWAY_ENABLE_LIVEKIT_BRIDGE: "true",
      VOICE_GATEWAY_FALLBACK_WS_URL: "ws://localhost:8000/api/v1/voice/ws/duplex",
      LIVEKIT_API_KEY: "key",
      LIVEKIT_API_SECRET: "secret",
      LIVEKIT_WS_URL: "ws://localhost:7880",
    });

    const service = new SessionBootstrapService(config);
    const bootstrap = await service.createSession({ sessionId: "session-1" });

    expect(bootstrap.mode).toBe("fallback_ws");
    expect(bootstrap.token).toBeUndefined();
  });

  it("reuses a provided reconnect token across bootstrap calls", async () => {
    const config = loadConfig({
      VOICE_GATEWAY_ENABLE_LIVEKIT_BRIDGE: "false",
      VOICE_GATEWAY_FALLBACK_WS_URL: "ws://localhost:8000/api/v1/voice/ws/duplex",
    });

    const service = new SessionBootstrapService(config);
    const bootstrap = await service.createSession({
      reconnectToken: "sticky-token",
      sessionId: "session-2",
    });

    expect(bootstrap.reconnect_token).toBe("sticky-token");
  });

  it("exposes health, readiness, and metrics routes", async () => {
    const config = loadConfig({
      VOICE_GATEWAY_ENABLE_LIVEKIT_BRIDGE: "false",
      VOICE_GATEWAY_FALLBACK_WS_URL: "ws://localhost:8000/api/v1/voice/ws/duplex",
    });

    const { app } = buildApp(config);
    const readyResponse = await request(app).get("/ready");
    const metricsResponse = await request(app).get("/metrics");

    expect(readyResponse.status).toBe(200);
    expect(readyResponse.body.ready).toBe(true);
    expect(readyResponse.body.dead_peer_timeout_ms).toBe(30000);
    expect(readyResponse.body.reconnect_backoff_ms).toEqual([1000, 2000, 4000, 8000, 16000]);
    expect(readyResponse.body.continuity_window_ms).toBe(400);
    expect(metricsResponse.status).toBe(200);
    expect(metricsResponse.text).toContain("voice_gateway_bootstrap_requests_total");
  });
});
