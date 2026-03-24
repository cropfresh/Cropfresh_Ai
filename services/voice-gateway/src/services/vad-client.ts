export interface VadAnalyzeFrameRequest {
  sessionId: string;
  sequence: number;
  sampleRate: number;
  pcm16Base64: string;
}

export interface VadAnalyzeFrameResponse {
  session_id: string;
  sequence: number;
  state: string;
  probability: number;
  rms: number;
  segment_id: string | null;
  end_of_segment: boolean;
}

export interface VadEvaluateSegmentRequest {
  sessionId: string;
  transcript: string;
  language: string;
}

export interface VadEvaluateSegmentResponse {
  session_id: string;
  transcript: string;
  language: string;
  should_flush: boolean;
  reason: string;
  semantic_hold_ms: number;
  used_llm: boolean;
  timed_out: boolean;
}

export class VadServiceClient {
  constructor(private readonly baseUrl: string) {}

  isConfigured(): boolean {
    return Boolean(this.baseUrl.trim());
  }

  async analyzeFrame(
    request: VadAnalyzeFrameRequest,
  ): Promise<VadAnalyzeFrameResponse | null> {
    if (!this.isConfigured()) {
      return null;
    }

    const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}/v1/vad/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: request.sessionId,
        sequence: request.sequence,
        sample_rate: request.sampleRate,
        pcm16_base64: request.pcm16Base64,
      }),
    });

    if (!response.ok) {
      throw new Error(`vad analyze failed with ${response.status}`);
    }

    return (await response.json()) as VadAnalyzeFrameResponse;
  }

  async evaluateSegment(
    request: VadEvaluateSegmentRequest,
  ): Promise<VadEvaluateSegmentResponse | null> {
    if (!this.isConfigured()) {
      return null;
    }

    const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}/v1/vad/segments/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: request.sessionId,
        transcript: request.transcript,
        language: request.language,
      }),
    });

    if (!response.ok) {
      throw new Error(`vad semantic evaluate failed with ${response.status}`);
    }

    return (await response.json()) as VadEvaluateSegmentResponse;
  }

  async resetSession(sessionId: string): Promise<boolean> {
    if (!this.isConfigured()) {
      return false;
    }

    const response = await fetch(
      `${this.baseUrl.replace(/\/$/, "")}/v1/vad/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "DELETE",
      },
    );
    if (!response.ok) {
      throw new Error(`vad reset failed with ${response.status}`);
    }

    const payload = (await response.json()) as { cleared?: boolean };
      return Boolean(payload.cleared);
  }
}
