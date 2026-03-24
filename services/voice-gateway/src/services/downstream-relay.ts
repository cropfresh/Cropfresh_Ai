export interface DownstreamRelayRequest {
  fallbackWsUrl: string;
  language: string;
  pcm16: Uint8Array;
  reconnectToken?: string;
  sessionId: string;
  userId: string;
}

export interface DownstreamRelayAudioChunk {
  audio_base64: string;
  chunk_index: number;
  format: string;
  is_last: boolean;
  sample_rate: number;
  timing?: Record<string, unknown>;
}

export interface DownstreamRelayResult {
  response_audio: DownstreamRelayAudioChunk[];
  response_end: Record<string, unknown>;
  response_sentences: string[];
  session_id: string;
  transport: "duplex_ws";
}

function buildRelayUrl(request: DownstreamRelayRequest): string {
  const url = new URL(request.fallbackWsUrl);
  url.searchParams.set("language", request.language);
  url.searchParams.set("session_id", request.sessionId);
  url.searchParams.set("user_id", request.userId);
  if (request.reconnectToken) {
    url.searchParams.set("reconnect_token", request.reconnectToken);
  }
  return url.toString();
}

function splitPcmFrames(payload: Uint8Array, frameBytes = 1024): Uint8Array[] {
  if (payload.length === 0) {
    return [];
  }

  const frames: Uint8Array[] = [];
  for (let offset = 0; offset < payload.length; offset += frameBytes) {
    frames.push(payload.subarray(offset, Math.min(payload.length, offset + frameBytes)));
  }
  return frames;
}

function safeStringify(payload: Record<string, unknown>): string {
  return JSON.stringify(payload);
}

export class DuplexWebsocketRelay {
  constructor(private readonly relayTimeoutMs = 20_000) {}

  async relayBufferedAudio(request: DownstreamRelayRequest): Promise<DownstreamRelayResult> {
    if (request.pcm16.length === 0) {
      throw new Error("buffered PCM audio is required");
    }

    const relayUrl = buildRelayUrl(request);
    const frames = splitPcmFrames(request.pcm16);

    return await new Promise<DownstreamRelayResult>((resolve, reject) => {
      const socket = new WebSocket(relayUrl);
      const audioChunks: DownstreamRelayAudioChunk[] = [];
      const responseSentences: string[] = [];
      let readySessionId = request.sessionId;
      let audioSent = false;
      let settled = false;

      const timeoutHandle = setTimeout(() => {
        fail(new Error("downstream relay timed out waiting for response_end"));
      }, this.relayTimeoutMs);

      const cleanup = () => {
        clearTimeout(timeoutHandle);
      };

      const closeSocket = () => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(safeStringify({ type: "close" }));
        }
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          socket.close();
        }
      };

      const fail = (error: Error) => {
        if (settled) {
          return;
        }

        settled = true;
        cleanup();
        closeSocket();
        reject(error);
      };

      const finish = (payload: Record<string, unknown>) => {
        if (settled) {
          return;
        }

        settled = true;
        cleanup();
        closeSocket();
        resolve({
          response_audio: audioChunks,
          response_end: payload,
          response_sentences: responseSentences,
          session_id: readySessionId,
          transport: "duplex_ws",
        });
      };

      socket.onerror = () => {
        fail(new Error("downstream websocket error"));
      };

      socket.onclose = () => {
        if (!settled) {
          fail(new Error("downstream websocket closed before response_end"));
        }
      };

      socket.onmessage = (event) => {
        let message: Record<string, unknown>;
        try {
          const raw =
            typeof event.data === "string"
              ? event.data
              : Buffer.from(event.data as ArrayBuffer).toString("utf-8");
          message = JSON.parse(raw) as Record<string, unknown>;
        } catch {
          fail(new Error("downstream websocket returned invalid JSON"));
          return;
        }

        switch (message.type) {
          case "ready":
            readySessionId =
              typeof message.session_id === "string" ? message.session_id : readySessionId;
            if (!audioSent) {
              audioSent = true;
              for (const frame of frames) {
                socket.send(
                  safeStringify({
                    type: "audio_chunk",
                    audio_base64: Buffer.from(frame).toString("base64"),
                  }),
                );
              }
              socket.send(safeStringify({ type: "audio_end" }));
            }
            return;
          case "response_sentence":
            if (typeof message.text === "string" && message.text) {
              responseSentences.push(message.text);
            }
            return;
          case "response_audio":
            if (typeof message.audio_base64 !== "string") {
              return;
            }
            audioChunks.push({
              audio_base64: message.audio_base64,
              chunk_index: Number(message.chunk_index ?? audioChunks.length),
              format: String(message.format ?? "audio/mpeg"),
              is_last: Boolean(message.is_last),
              sample_rate: Number(message.sample_rate ?? 0),
              timing:
                message.timing && typeof message.timing === "object"
                  ? (message.timing as Record<string, unknown>)
                  : undefined,
            });
            return;
          case "response_end":
            finish(message);
            return;
          case "error":
            fail(new Error(String(message.error ?? "downstream relay error")));
            return;
          default:
            return;
        }
      };
    });
  }
}
