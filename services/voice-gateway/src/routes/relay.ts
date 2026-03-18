import type { Express, Request, Response } from "express";

import { RelaySessionStore } from "../audio/relay-session.js";

function readPcmBase64(value: unknown): Buffer | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }

  try {
    return Buffer.from(value, "base64");
  } catch {
    return null;
  }
}

export function registerRelayRoutes(
  app: Express,
  relayStore: RelaySessionStore,
): void {
  app.post("/sessions/:sessionId/relay/frame", (request: Request, response: Response) => {
    const sessionId = request.params.sessionId?.trim();
    const pcm = readPcmBase64(request.body?.pcm16_base64);
    const timestampMs =
      typeof request.body?.timestamp_ms === "number" ? request.body.timestamp_ms : Date.now();

    if (!sessionId || pcm == null) {
      response.status(400).json({ error: "sessionId and pcm16_base64 are required" });
      return;
    }

    const relaySession = relayStore.get(sessionId);
    const result = relaySession.pushFrame(new Uint8Array(pcm), timestampMs);
    response.json({ session_id: sessionId, ...result });
  });
}
