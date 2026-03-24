import { RelayFrameResult, RelaySessionStore } from "../audio/relay-session.js";
import { recordRelayDecisionMetrics, recordRelayFrameMetrics } from "../metrics.js";

import { DuplexWebsocketRelay, DownstreamRelayResult } from "./downstream-relay.js";
import {
  VadAnalyzeFrameResponse,
  VadEvaluateSegmentResponse,
  VadServiceClient,
} from "./vad-client.js";

export interface RelayFrameRequest {
  forceFlush: boolean;
  language: string;
  pcm16: Uint8Array;
  reconnectToken?: string;
  sampleRate: number;
  semanticLanguage?: string;
  semanticTranscript?: string;
  sequence: number;
  sessionId: string;
  timestampMs: number;
  userId: string;
}

export interface RelayFrameResponse {
  downstream?: DownstreamRelayResult;
  downstream_error?: string;
  fallback_ws_url: string;
  flushed: boolean;
  relay: RelayFrameResult;
  session_id: string;
  segment_decision?: VadEvaluateSegmentResponse | null;
  vad?: VadAnalyzeFrameResponse | null;
  vad_error?: string;
}

export interface RelayFlushRequest {
  language: string;
  reconnectToken?: string;
  sessionId: string;
  userId: string;
}

export interface RelayFlushResponse {
  downstream?: DownstreamRelayResult;
  downstream_error?: string;
  fallback_ws_url: string;
  flushed: boolean;
  session_id: string;
}

export interface RelayResetResponse {
  cleared: boolean;
  session_id: string;
  vad_cleared: boolean;
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class RelayCoordinator {
  constructor(
    private readonly relayStore: RelaySessionStore,
    private readonly vadClient: VadServiceClient,
    private readonly downstreamRelay: DuplexWebsocketRelay,
    private readonly fallbackWsUrl: string,
  ) {}

  async processFrame(request: RelayFrameRequest): Promise<RelayFrameResponse> {
    const relaySession = this.relayStore.get(request.sessionId);
    const relay = relaySession.pushFrame(request.pcm16, request.timestampMs);
    recordRelayFrameMetrics(relay);
    let vad: VadAnalyzeFrameResponse | null = null;
    let segmentDecision: VadEvaluateSegmentResponse | null = null;
    let vadError: string | undefined;

    if (!request.forceFlush) {
      try {
        vad = await this.vadClient.analyzeFrame({
          sessionId: request.sessionId,
          sequence: request.sequence,
          sampleRate: request.sampleRate,
          pcm16Base64: Buffer.from(request.pcm16).toString("base64"),
        });
      } catch (error) {
        vadError = toErrorMessage(error);
      }
    }

    if (!request.forceFlush && vad?.end_of_segment) {
      try {
        segmentDecision = await this.vadClient.evaluateSegment({
          sessionId: request.sessionId,
          transcript: request.semanticTranscript?.trim() || " ",
          language: request.semanticLanguage?.trim() || request.language,
        });
      } catch (error) {
        vadError = vadError ?? toErrorMessage(error);
      }
    }

    const decisionReason = this.resolveDecisionReason(request.forceFlush, vad, segmentDecision);
    const shouldFlush =
      request.forceFlush || Boolean(vad?.end_of_segment && (segmentDecision?.should_flush ?? true));
    if (!shouldFlush) {
      recordRelayDecisionMetrics("hold", decisionReason);
      return {
        fallback_ws_url: this.fallbackWsUrl,
        flushed: false,
        relay,
        session_id: request.sessionId,
        segment_decision: segmentDecision,
        vad,
        vad_error: vadError,
      };
    }

    recordRelayDecisionMetrics("flush", decisionReason);
    const flushResult = await this.flushSession({
      language: request.language,
      reconnectToken: request.reconnectToken,
      sessionId: request.sessionId,
      userId: request.userId,
    });
    return {
      ...flushResult,
      relay,
      segment_decision: segmentDecision,
      vad,
      vad_error: vadError,
    };
  }

  async flushSession(request: RelayFlushRequest): Promise<RelayFlushResponse> {
    const relaySession = this.relayStore.get(request.sessionId);
    if (!relaySession.hasBufferedAudio()) {
      return {
        fallback_ws_url: this.fallbackWsUrl,
        flushed: false,
        session_id: request.sessionId,
      };
    }

    try {
      const downstream = await this.downstreamRelay.relayBufferedAudio({
        fallbackWsUrl: this.fallbackWsUrl,
        language: request.language,
        pcm16: relaySession.readBufferedAudio(),
        reconnectToken: request.reconnectToken,
        sessionId: request.sessionId,
        userId: request.userId,
      });
      relaySession.clearBufferedAudio();
      await this.safeResetVadSession(request.sessionId);
      return {
        downstream,
        fallback_ws_url: this.fallbackWsUrl,
        flushed: true,
        session_id: request.sessionId,
      };
    } catch (error) {
      return {
        downstream_error: toErrorMessage(error),
        fallback_ws_url: this.fallbackWsUrl,
        flushed: false,
        session_id: request.sessionId,
      };
    }
  }

  async resetSession(sessionId: string): Promise<RelayResetResponse> {
    const relaySession = this.relayStore.peek(sessionId);
    relaySession?.clearBufferedAudio();
    const cleared = relaySession ? this.relayStore.delete(sessionId) : false;
    const vadCleared = await this.safeResetVadSession(sessionId);
    return {
      cleared,
      session_id: sessionId,
      vad_cleared: vadCleared,
    };
  }

  private async safeResetVadSession(sessionId: string): Promise<boolean> {
    try {
      return await this.vadClient.resetSession(sessionId);
    } catch {
      return false;
    }
  }

  private resolveDecisionReason(
    forceFlush: boolean,
    vad: VadAnalyzeFrameResponse | null,
    segmentDecision: VadEvaluateSegmentResponse | null,
  ): string {
    if (forceFlush) {
      return "force_flush";
    }
    if (segmentDecision?.reason) {
      return segmentDecision.reason;
    }
    if (vad?.end_of_segment) {
      return "acoustic_end";
    }
    return "buffering";
  }
}
