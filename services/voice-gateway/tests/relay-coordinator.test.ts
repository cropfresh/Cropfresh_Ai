import { describe, expect, it, vi } from "vitest";

import { RelaySessionStore } from "../src/audio/relay-session.js";
import { RelayCoordinator } from "../src/services/relay-coordinator.js";

class FakeVadClient {
  analyzeFrame = vi.fn(async () => null);
  resetSession = vi.fn(async () => true);
}

class FakeDownstreamRelay {
  relayBufferedAudio = vi.fn(async () => ({
    response_audio: [
      {
        audio_base64: "ZmFrZQ==",
        chunk_index: 0,
        format: "audio/mpeg",
        is_last: true,
        sample_rate: 24000,
      },
    ],
    response_end: { full_text: "Namaskara", type: "response_end" },
    response_sentences: ["Namaskara"],
    session_id: "session-1",
    transport: "duplex_ws" as const,
  }));
}

describe("RelayCoordinator", () => {
  it("keeps buffering when VAD has not ended the segment yet", async () => {
    const relayStore = new RelaySessionStore(0.01, 400);
    const vadClient = new FakeVadClient();
    vadClient.analyzeFrame.mockResolvedValueOnce({
      end_of_segment: false,
      probability: 0.82,
      rms: 0.21,
      segment_id: "segment-1",
      sequence: 8,
      session_id: "session-1",
      state: "speech",
    });
    const downstreamRelay = new FakeDownstreamRelay();
    const coordinator = new RelayCoordinator(
      relayStore,
      vadClient as never,
      downstreamRelay as never,
      "ws://localhost:8000/api/v1/voice/ws/duplex",
    );

    const response = await coordinator.processFrame({
      forceFlush: false,
      language: "kn",
      pcm16: new Uint8Array(new Int16Array([0, 14000, -14000, 8000]).buffer),
      sampleRate: 16000,
      sequence: 8,
      sessionId: "session-1",
      timestampMs: 1000,
      userId: "farmer-1",
    });

    expect(response.flushed).toBe(false);
    expect(response.vad?.state).toBe("speech");
    expect(response.relay.buffered_bytes).toBeGreaterThan(0);
    expect(downstreamRelay.relayBufferedAudio).not.toHaveBeenCalled();
  });

  it("flushes buffered audio into the downstream duplex relay on demand", async () => {
    const relayStore = new RelaySessionStore(0.01, 400);
    const vadClient = new FakeVadClient();
    const downstreamRelay = new FakeDownstreamRelay();
    const coordinator = new RelayCoordinator(
      relayStore,
      vadClient as never,
      downstreamRelay as never,
      "ws://localhost:8000/api/v1/voice/ws/duplex",
    );

    const response = await coordinator.processFrame({
      forceFlush: true,
      language: "hi",
      pcm16: new Uint8Array(new Int16Array([0, 12000, -12000, 6000]).buffer),
      sampleRate: 16000,
      sequence: 1,
      sessionId: "session-1",
      timestampMs: 1000,
      userId: "web_user",
    });

    expect(response.flushed).toBe(true);
    expect(response.downstream?.response_end.full_text).toBe("Namaskara");
    expect(downstreamRelay.relayBufferedAudio).toHaveBeenCalledTimes(1);
    expect(vadClient.resetSession).toHaveBeenCalledWith("session-1");
  });

  it("preserves buffered audio when the downstream relay fails", async () => {
    const relayStore = new RelaySessionStore(0.01, 400);
    const vadClient = new FakeVadClient();
    const downstreamRelay = new FakeDownstreamRelay();
    downstreamRelay.relayBufferedAudio.mockRejectedValueOnce(new Error("duplex unavailable"));
    const coordinator = new RelayCoordinator(
      relayStore,
      vadClient as never,
      downstreamRelay as never,
      "ws://localhost:8000/api/v1/voice/ws/duplex",
    );

    await coordinator.processFrame({
      forceFlush: false,
      language: "hi",
      pcm16: new Uint8Array(new Int16Array([0, 12000, -12000, 6000]).buffer),
      sampleRate: 16000,
      sequence: 1,
      sessionId: "session-2",
      timestampMs: 1000,
      userId: "web_user",
    });
    const flushResponse = await coordinator.flushSession({
      language: "hi",
      sessionId: "session-2",
      userId: "web_user",
    });

    expect(flushResponse.flushed).toBe(false);
    expect(flushResponse.downstream_error).toContain("duplex unavailable");
    expect(relayStore.get("session-2").hasBufferedAudio()).toBe(true);
  });
});
