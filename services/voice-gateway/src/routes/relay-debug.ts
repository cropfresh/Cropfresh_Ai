import type { RelayFrameResult } from "../audio/relay-session.js";
import type { DownstreamRelayResult } from "../services/downstream-relay.js";

interface RelayResponseShape {
  downstream?: DownstreamRelayResult;
  relay?: RelayFrameResult;
}

type TimingValue = number | null | undefined;

function readTimingValue(record: Record<string, unknown>, key: string): TimingValue {
  const value = record[key];
  return typeof value === "number" ? value : undefined;
}

function readTimingRecord(downstream?: DownstreamRelayResult): Record<string, unknown> {
  const timing = downstream?.response_end?.timing;
  return timing && typeof timing === "object" ? (timing as Record<string, unknown>) : {};
}

export function withRelayDebug<T extends RelayResponseShape>(payload: T): T & {
  debug: {
    continuity?: {
      continuity_fill: boolean;
      continuity_fill_bytes: number;
      continuity_fill_mode: "comfort_noise" | "none";
      continuity_gap_ms: number;
      ring_buffer_watermark: "normal" | "high";
    };
    interruption?: {
      bargein_reaction_ms?: TimingValue;
      interrupted_ms?: TimingValue;
      total_ms?: TimingValue;
    };
  };
} {
  const timing = readTimingRecord(payload.downstream);
  return {
    ...payload,
    debug: {
      continuity: payload.relay
        ? {
            continuity_fill: payload.relay.continuity_fill,
            continuity_fill_bytes: payload.relay.continuity_fill_bytes,
            continuity_fill_mode: payload.relay.continuity_fill_mode,
            continuity_gap_ms: payload.relay.continuity_gap_ms,
            ring_buffer_watermark: payload.relay.ring_buffer_watermark,
          }
        : undefined,
      interruption: payload.downstream
        ? {
            bargein_reaction_ms: readTimingValue(timing, "bargein_reaction_ms"),
            interrupted_ms: readTimingValue(timing, "interrupted_ms"),
            total_ms: readTimingValue(timing, "total_ms"),
          }
        : undefined,
    },
  };
}
