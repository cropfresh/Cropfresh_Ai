import { SharedRingBuffer, secondsToPcm16Bytes } from "./ring-buffer.js";
import { evaluateRmsGate } from "./rms-gate.js";

export interface RelayFrameResult {
  accepted_bytes: number;
  continuity_fill: boolean;
  continuity_fill_bytes: number;
  continuity_gap_ms: number;
  gate_active: boolean;
  ring_buffer_fill_ratio: number;
  ring_buffer_watermark: "normal" | "high";
  rms: number;
}

export class RelaySession {
  private readonly ringBuffer = new SharedRingBuffer(secondsToPcm16Bytes(5));
  private lastFrameAtMs: number | null = null;

  constructor(
    private readonly rmsThreshold = 0.015,
    private readonly continuityWindowMs = 400,
  ) {}

  pushFrame(payload: Uint8Array, nowMs = Date.now()): RelayFrameResult {
    const gate = evaluateRmsGate(payload, this.rmsThreshold);
    const continuityGapMs = this.lastFrameAtMs == null ? 0 : Math.max(0, nowMs - this.lastFrameAtMs);
    const continuityFill = !gate.isActive && continuityGapMs > 0 && continuityGapMs <= this.continuityWindowMs;

    if (gate.isActive || continuityFill) {
      this.ringBuffer.write(payload);
    }

    this.lastFrameAtMs = nowMs;
    const snapshot = this.ringBuffer.snapshot();
    const ringBufferFillRatio = snapshot.capacityBytes === 0 ? 0 : snapshot.sizeBytes / snapshot.capacityBytes;

    return {
      accepted_bytes: payload.length,
      continuity_fill: continuityFill,
      continuity_fill_bytes: continuityFill ? payload.length : 0,
      continuity_gap_ms: continuityGapMs,
      gate_active: gate.isActive,
      ring_buffer_fill_ratio: Number(ringBufferFillRatio.toFixed(4)),
      ring_buffer_watermark: ringBufferFillRatio >= 0.8 ? "high" : "normal",
      rms: Number(gate.rms.toFixed(6)),
    };
  }
}

export class RelaySessionStore {
  private readonly sessions = new Map<string, RelaySession>();

  constructor(
    private readonly rmsThreshold = 0.015,
    private readonly continuityWindowMs = 400,
  ) {}

  get(sessionId: string): RelaySession {
    const existing = this.sessions.get(sessionId);
    if (existing) {
      return existing;
    }

    const created = new RelaySession(this.rmsThreshold, this.continuityWindowMs);
    this.sessions.set(sessionId, created);
    return created;
  }
}
