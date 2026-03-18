import request from "supertest";
import { describe, expect, it } from "vitest";

import { RelaySession } from "../src/audio/relay-session.js";
import { buildApp } from "../src/app.js";
import { loadConfig } from "../src/config.js";

describe("RelaySession", () => {
  it("marks continuity fill during short quiet gaps", () => {
    const relay = new RelaySession(0.01, 400);

    relay.pushFrame(new Uint8Array(new Int16Array([0, 12000, -12000, 8000]).buffer), 1000);
    const result = relay.pushFrame(new Uint8Array(new Int16Array([0, 10, -10, 5]).buffer), 1200);

    expect(result.continuity_fill).toBe(true);
    expect(result.continuity_gap_ms).toBe(200);
  });

  it("exposes a relay frame route that wires RMS gate and ring buffer metadata", async () => {
    const config = loadConfig({
      VOICE_GATEWAY_ENABLE_LIVEKIT_BRIDGE: "false",
      VOICE_GATEWAY_FALLBACK_WS_URL: "ws://localhost:8000/api/v1/voice/ws/duplex",
    });
    const { app } = buildApp(config);
    const pcmPayload = Buffer.from(new Int16Array([0, 9000, -9000, 4500]).buffer).toString("base64");

    const response = await request(app)
      .post("/sessions/test-session/relay/frame")
      .send({ pcm16_base64: pcmPayload, timestamp_ms: 2000 });

    expect(response.status).toBe(200);
    expect(response.body.session_id).toBe("test-session");
    expect(response.body.accepted_bytes).toBeGreaterThan(0);
    expect(typeof response.body.rms).toBe("number");
    expect(typeof response.body.ring_buffer_fill_ratio).toBe("number");
    expect(["normal", "high"]).toContain(response.body.ring_buffer_watermark);
  });
});
