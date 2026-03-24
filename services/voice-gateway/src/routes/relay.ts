import type { Express, Request, Response } from "express";

import {
  RelayCoordinator,
  RelayFlushRequest,
  RelayFrameRequest,
} from "../services/relay-coordinator.js";
import { withRelayDebug } from "./relay-debug.js";

function readPathString(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }

  return undefined;
}

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
  relayCoordinator: RelayCoordinator,
): void {
  app.post("/sessions/:sessionId/relay/frame", async (request: Request, response: Response) => {
    const sessionId = readPathString(request.params.sessionId);
    const pcm = readPcmBase64(request.body?.pcm16_base64);
    const timestampMs =
      typeof request.body?.timestamp_ms === "number" ? request.body.timestamp_ms : Date.now();
    const sequence =
      typeof request.body?.sequence === "number" ? request.body.sequence : timestampMs;
    const sampleRate =
      typeof request.body?.sample_rate === "number" ? request.body.sample_rate : 16_000;
    const forceFlush = request.body?.force_flush === true;
    const semanticTranscript =
      typeof request.body?.semantic_transcript === "string" && request.body.semantic_transcript.trim()
        ? request.body.semantic_transcript.trim()
        : undefined;
    const semanticLanguage =
      typeof request.body?.semantic_language === "string" && request.body.semantic_language.trim()
        ? request.body.semantic_language.trim()
        : undefined;
    const userId =
      typeof request.body?.user_id === "string" && request.body.user_id.trim()
        ? request.body.user_id.trim()
        : "web_user";
    const language =
      typeof request.body?.language === "string" && request.body.language.trim()
        ? request.body.language.trim()
        : "hi";
    const reconnectToken =
      typeof request.body?.reconnect_token === "string" && request.body.reconnect_token.trim()
        ? request.body.reconnect_token.trim()
        : undefined;

    if (!sessionId || pcm == null) {
      response.status(400).json({ error: "sessionId and pcm16_base64 are required" });
      return;
    }

    const relayRequest: RelayFrameRequest = {
      forceFlush,
      language,
      pcm16: new Uint8Array(pcm),
      reconnectToken,
      sampleRate,
      semanticLanguage,
      semanticTranscript,
      sequence,
      sessionId,
      timestampMs,
      userId,
    };
    const result = await relayCoordinator.processFrame(relayRequest);
    response.json(withRelayDebug(result));
  });

  app.post("/sessions/:sessionId/relay/flush", async (request: Request, response: Response) => {
    const sessionId = readPathString(request.params.sessionId);
    if (!sessionId) {
      response.status(400).json({ error: "sessionId is required" });
      return;
    }

    const flushRequest: RelayFlushRequest = {
      language:
        typeof request.body?.language === "string" && request.body.language.trim()
          ? request.body.language.trim()
          : "hi",
      reconnectToken:
        typeof request.body?.reconnect_token === "string" && request.body.reconnect_token.trim()
          ? request.body.reconnect_token.trim()
          : undefined,
      sessionId,
      userId:
        typeof request.body?.user_id === "string" && request.body.user_id.trim()
          ? request.body.user_id.trim()
          : "web_user",
    };
    const result = await relayCoordinator.flushSession(flushRequest);
    response.json(withRelayDebug(result));
  });

  app.delete("/sessions/:sessionId/relay", async (request: Request, response: Response) => {
    const sessionId = readPathString(request.params.sessionId);
    if (!sessionId) {
      response.status(400).json({ error: "sessionId is required" });
      return;
    }

    const result = await relayCoordinator.resetSession(sessionId);
    response.json(result);
  });
}
