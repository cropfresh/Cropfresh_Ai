import type { Express, Request, Response } from "express";

import {
  BootstrapRequest,
  SessionBootstrapService,
} from "../services/session-bootstrap.js";

function readOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

export function registerSessionRoutes(
  app: Express,
  bootstrapService: SessionBootstrapService,
): void {
  app.post("/sessions/bootstrap", async (request: Request, response: Response) => {
    const payload = request.body ?? {};
    const bootstrapRequest: BootstrapRequest = {
      userId: readOptionalString(payload.user_id),
      language: readOptionalString(payload.language),
      requestedMode: readOptionalString(payload.requested_mode),
      sessionId: readOptionalString(payload.session_id),
      reconnectToken: readOptionalString(payload.reconnect_token),
    };

    const bootstrap = await bootstrapService.createSession(bootstrapRequest);
    response.json(bootstrap);
  });
}
