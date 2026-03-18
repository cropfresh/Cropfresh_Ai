import { Counter, Gauge, Registry, collectDefaultMetrics } from "prom-client";

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

export function updateConfigurationMetrics(enabled: boolean, configured: boolean): void {
  liveKitBridgeEnabled.set(enabled ? 1 : 0);
  liveKitBridgeConfigured.set(configured ? 1 : 0);
}
