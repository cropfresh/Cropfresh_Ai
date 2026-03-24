import { Counter, Gauge, Registry, collectDefaultMetrics } from "prom-client";

import type { RelayFrameResult } from "./audio/relay-session.js";

export const metricsRegistry = new Registry();

collectDefaultMetrics({
  prefix: "voice_gateway_",
  register: metricsRegistry,
});

export const bootstrapRequestsTotal = new Counter({
  name: "voice_gateway_bootstrap_requests_total",
  help: "Total gateway bootstrap requests partitioned by final mode and outcome.",
  labelNames: ["mode", "outcome"] as const,
  registers: [metricsRegistry],
});

export const liveKitBridgeEnabled = new Gauge({
  name: "voice_gateway_livekit_bridge_enabled",
  help: "Whether bridge mode is enabled by configuration.",
  registers: [metricsRegistry],
});

export const liveKitBridgeConfigured = new Gauge({
  name: "voice_gateway_livekit_bridge_configured",
  help: "Whether the LiveKit URL and credentials are fully configured.",
  registers: [metricsRegistry],
});

export const relayContinuityEventsTotal = new Counter({
  name: "voice_gateway_relay_continuity_events_total",
  help: "Relay continuity fills and watermarks emitted while buffering audio.",
  labelNames: ["continuity_fill", "watermark"] as const,
  registers: [metricsRegistry],
});

export const relayJointDecisionsTotal = new Counter({
  name: "voice_gateway_relay_joint_decisions_total",
  help: "Joint acoustic and semantic relay decisions by final outcome and reason.",
  labelNames: ["outcome", "reason"] as const,
  registers: [metricsRegistry],
});

export function updateConfigurationMetrics(enabled: boolean, configured: boolean): void {
  liveKitBridgeEnabled.set(enabled ? 1 : 0);
  liveKitBridgeConfigured.set(configured ? 1 : 0);
}

export function recordRelayFrameMetrics(result: RelayFrameResult): void {
  relayContinuityEventsTotal.inc({
    continuity_fill: result.continuity_fill ? "true" : "false",
    watermark: result.ring_buffer_watermark,
  });
}

export function recordRelayDecisionMetrics(outcome: "flush" | "hold", reason: string): void {
  relayJointDecisionsTotal.inc({
    outcome,
    reason,
  });
}
