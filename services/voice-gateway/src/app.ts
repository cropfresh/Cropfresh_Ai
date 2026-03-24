import cors from "cors";
import express from "express";

import { RelaySessionStore } from "./audio/relay-session.js";
import { VoiceGatewayConfig, isLiveKitConfigured } from "./config.js";
import { updateConfigurationMetrics } from "./metrics.js";
import { registerHealthRoutes } from "./routes/health.js";
import { registerRelayRoutes } from "./routes/relay.js";
import { registerSessionRoutes } from "./routes/sessions.js";
import { DuplexWebsocketRelay } from "./services/downstream-relay.js";
import { RelayCoordinator } from "./services/relay-coordinator.js";
import { SessionBootstrapService } from "./services/session-bootstrap.js";
import { VadServiceClient } from "./services/vad-client.js";

interface BuildAppOptions {
  relayCoordinator?: RelayCoordinator;
}

export function buildApp(
  config: VoiceGatewayConfig,
  bootstrapService = new SessionBootstrapService(config),
  options: BuildAppOptions = {},
) {
  const app = express();
  const relayStore = new RelaySessionStore(0.015, config.continuityWindowMs);
  const relayCoordinator =
    options.relayCoordinator ??
    new RelayCoordinator(
      relayStore,
      new VadServiceClient(config.vadServiceBaseUrl),
      new DuplexWebsocketRelay(),
      config.fallbackWsUrl,
    );

  app.use(cors({ origin: config.corsOrigin === "*" ? true : config.corsOrigin }));
  app.use(express.json({ limit: "1mb" }));

  updateConfigurationMetrics(config.enableLiveKitBridge, isLiveKitConfigured(config));
  registerHealthRoutes(app, config, bootstrapService);
  registerSessionRoutes(app, bootstrapService);
  registerRelayRoutes(app, relayCoordinator);

  return { app, bootstrapService, relayCoordinator };
}
