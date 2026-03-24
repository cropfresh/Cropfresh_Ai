import { describe, expect, it } from "vitest";

import { withRelayDebug } from "../src/routes/relay-debug.js";

describe("withRelayDebug", () => {
  it("extracts continuity and interruption metadata for debugging", () => {
    const payload = withRelayDebug({
      downstream: {
        response_audio: [],
        response_end: {
          timing: {
            bargein_reaction_ms: 120,
            interrupted_ms: 540,
            total_ms: 980,
          },
          type: "response_end",
        },
        response_sentences: [],
        session_id: "session-1",
        transport: "duplex_ws" as const,
      },
      flushed: true,
      relay: {
        accepted_bytes: 1024,
        buffered_bytes: 2048,
        continuity_fill: true,
        continuity_fill_bytes: 512,
        continuity_fill_mode: "comfort_noise" as const,
        continuity_gap_ms: 180,
        gate_active: false,
        ring_buffer_fill_ratio: 0.4,
        ring_buffer_watermark: "normal" as const,
        rms: 0.01,
      },
      session_id: "session-1",
    });

    expect(payload.debug.continuity?.continuity_fill_mode).toBe("comfort_noise");
    expect(payload.debug.continuity?.continuity_gap_ms).toBe(180);
    expect(payload.debug.interruption?.bargein_reaction_ms).toBe(120);
    expect(payload.debug.interruption?.interrupted_ms).toBe(540);
  });
});
