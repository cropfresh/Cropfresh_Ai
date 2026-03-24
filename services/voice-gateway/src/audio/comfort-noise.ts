export function buildComfortNoiseFrame(byteLength: number, amplitude = 64): Uint8Array {
  if (byteLength <= 0) {
    return new Uint8Array(0);
  }

  const sampleCount = Math.ceil(byteLength / Int16Array.BYTES_PER_ELEMENT);
  const samples = new Int16Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) {
    // Alternate the polarity so short fills stay low-energy but not completely dead.
    samples[index] = index % 2 === 0 ? amplitude : -amplitude;
  }

  return new Uint8Array(samples.buffer.slice(0, byteLength));
}
