import { describe, expect, it, vi } from "vitest";

import { RelaySessionStore } from "../src/audio/relay-session.js";
import { RelayCoordinator } from "../src/services/relay-coordinator.js";

class SemanticVadClient {
  analyzeFrame = vi.fn(async () => ({
    end_of_segment: true,
    probability: 0.88,
    rms: 0.2,
    segment_id: "segment-1",
    sequence: 1,
    session_id: "session-1",
    state: "speech_end",
  }));

  evaluateSegment = vi.fn(async () => ({
    language: "en",
    reason: "heuristic_hold",
    semantic_hold_ms: 200,
    session_id: "session-1",
    should_flush: false,
    timed_out: false,
    transcript: "one second",
    used_llm: false,
  }));

  resetSession = vi.fn(async () => true);
}

class DownstreamRelay {
  relayBufferedAudio = vi.fn(async () => ({
    response_audio: [],
    response_end: { type: "response_end" },
    response_sentences: [],
    session_id: "session-1",
    transport: "duplex_ws" as const,
  }));
}

describe("RelayCoordinator semantic endpointing", () => {
  it("keeps buffering when semantic endpointing asks the gateway to hold", async () => {
    const vadClient = new SemanticVadClient();
    const downstreamRelay = new DownstreamRelay();
    const coordinator = new RelayCoordinator(
      new RelaySessionStore(0.01, 400),
      vadClient as never,
      downstreamRelay as never,
      "ws://localhost:8000/api/v1/voice/ws/duplex",
    );

    const response = await coordinator.processFrame({
      forceFlush: false,
      language: "en",
      pcm16: new Uint8Array(new Int16Array([0, 9000, -9000, 4500]).buffer),
      sampleRate: 16000,
      semanticTranscript: "one second",
      sequence: 1,
      sessionId: "session-1",
      timestampMs: 1000,
      userId: "farmer-1",
    });

    expect(response.flushed).toBe(false);
    expect(response.segment_decision?.should_flush).toBe(false);
    expect(response.segment_decision?.reason).toBe("heuristic_hold");
    expect(downstreamRelay.relayBufferedAudio).not.toHaveBeenCalled();
  });

  it("flushes when the semantic decision returns a timeout-safe flush", async () => {
    const vadClient = new SemanticVadClient();
    vadClient.evaluateSegment.mockResolvedValueOnce({
      language: "en",
      reason: "semantic_hold_timeout",
      semantic_hold_ms: 810,
      session_id: "session-1",
      should_flush: true,
      timed_out: false,
      transcript: "one second",
      used_llm: false,
    });
    const downstreamRelay = new DownstreamRelay();
    const coordinator = new RelayCoordinator(
      new RelaySessionStore(0.01, 400),
      vadClient as never,
      downstreamRelay as never,
      "ws://localhost:8000/api/v1/voice/ws/duplex",
    );

    const response = await coordinator.processFrame({
      forceFlush: false,
      language: "en",
      pcm16: new Uint8Array(new Int16Array([0, 9000, -9000, 4500]).buffer),
      sampleRate: 16000,
      semanticTranscript: "one second",
      sequence: 2,
      sessionId: "session-1",
      timestampMs: 1300,
      userId: "farmer-1",
    });

    expect(response.flushed).toBe(true);
    expect(response.segment_decision?.reason).toBe("semantic_hold_timeout");
    expect(downstreamRelay.relayBufferedAudio).toHaveBeenCalledTimes(1);
  });
});
