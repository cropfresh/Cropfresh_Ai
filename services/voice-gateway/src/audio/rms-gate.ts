export interface RmsGateResult {
  rms: number;
  isActive: boolean;
  sampleCount: number;
}

function toInt16View(payload: Buffer | Uint8Array | Int16Array): Int16Array {
  if (payload instanceof Int16Array) {
    return payload;
  }

  const safeLength = payload.byteLength - (payload.byteLength % Int16Array.BYTES_PER_ELEMENT);

  return new Int16Array(payload.buffer, payload.byteOffset, safeLength / Int16Array.BYTES_PER_ELEMENT);
}

export function computeNormalizedRms(payload: Buffer | Uint8Array | Int16Array): number {
  const samples = toInt16View(payload);
  if (samples.length === 0) {
    return 0;
  }

  let sumSquares = 0;
  for (const sample of samples) {
    const normalized = sample / 32768;
    sumSquares += normalized * normalized;
  }

  return Math.sqrt(sumSquares / samples.length);
}

export function evaluateRmsGate(
  payload: Buffer | Uint8Array | Int16Array,
  threshold = 0.015,
): RmsGateResult {
  const samples = toInt16View(payload);
  const rms = computeNormalizedRms(samples);

  return {
    rms,
    isActive: rms >= threshold,
    sampleCount: samples.length,
  };
}
