import cors from "cors";
import express from "express";

import { RelaySessionStore } from "./audio/relay-session.js";
import { VoiceGatewayConfig, isLiveKitConfigured } from "./config.js";
import { updateConfigurationMetrics } from "./metrics.js";
import { registerHealthRoutes } from "./routes/health.js";
import { registerRelayRoutes } from "./routes/relay.js";
import { registerSessionRoutes } from "./routes/sessions.js";
import { SessionBootstrapService } from "./services/session-bootstrap.js";

export function buildApp(
  config: VoiceGatewayConfig,
  bootstrapService = new SessionBootstrapService(config),
) {
  const app = express();
  const relayStore = new RelaySessionStore(0.015, config.continuityWindowMs);

  app.use(cors({ origin: config.corsOrigin === "*" ? true : config.corsOrigin }));
  app.use(express.json({ limit: "1mb" }));

  updateConfigurationMetrics(config.enableLiveKitBridge, isLiveKitConfigured(config));
  registerHealthRoutes(app, config, bootstrapService);
  registerSessionRoutes(app, bootstrapService);
  registerRelayRoutes(app, relayStore);

  return { app, bootstrapService };
}
