export interface RingBufferSnapshot {
  capacityBytes: number;
  sizeBytes: number;
  writeIndex: number;
}

export class SharedRingBuffer {
  private readonly data: Uint8Array;
  private readonly state: Int32Array;

  constructor(private readonly capacityBytes: number) {
    if (!Number.isInteger(capacityBytes) || capacityBytes <= 0) {
      throw new Error("capacityBytes must be a positive integer");
    }

    this.data = new Uint8Array(new SharedArrayBuffer(capacityBytes));
    this.state = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT * 2));
  }

  write(chunk: Uint8Array): RingBufferSnapshot {
    if (chunk.length === 0) {
      return this.snapshot();
    }

    const writeIndex = Atomics.load(this.state, 0);
    const firstCopyLength = Math.min(chunk.length, this.capacityBytes - writeIndex);

    this.data.set(chunk.subarray(0, firstCopyLength), writeIndex);

    const secondCopyLength = chunk.length - firstCopyLength;
    if (secondCopyLength > 0) {
      this.data.set(chunk.subarray(firstCopyLength), 0);
    }

    const nextWriteIndex = (writeIndex + chunk.length) % this.capacityBytes;
    const currentSize = Atomics.load(this.state, 1);
    const nextSize = Math.min(this.capacityBytes, currentSize + chunk.length);

    Atomics.store(this.state, 0, nextWriteIndex);
    Atomics.store(this.state, 1, nextSize);
    return this.snapshot();
  }

  readLatest(length: number): Uint8Array {
    if (length <= 0) {
      return new Uint8Array();
    }

    const currentSize = Atomics.load(this.state, 1);
    if (currentSize === 0) {
      return new Uint8Array();
    }

    const actualLength = Math.min(length, currentSize);
    const writeIndex = Atomics.load(this.state, 0);

    //! The ring buffer stays byte-oriented so downstream PCM framing can reconstruct exact windows.
    const startIndex = (writeIndex - actualLength + this.capacityBytes) % this.capacityBytes;
    const output = new Uint8Array(actualLength);
    const firstCopyLength = Math.min(actualLength, this.capacityBytes - startIndex);

    output.set(this.data.subarray(startIndex, startIndex + firstCopyLength), 0);

    if (actualLength > firstCopyLength) {
      output.set(this.data.subarray(0, actualLength - firstCopyLength), firstCopyLength);
    }

    return output;
  }

  clear(): void {
    Atomics.store(this.state, 0, 0);
    Atomics.store(this.state, 1, 0);
  }

  snapshot(): RingBufferSnapshot {
    return {
      capacityBytes: this.capacityBytes,
      sizeBytes: Atomics.load(this.state, 1),
      writeIndex: Atomics.load(this.state, 0),
    };
  }
}

export function secondsToPcm16Bytes(
  seconds: number,
  sampleRate = 16_000,
  channels = 1,
): number {
  return Math.round(seconds * sampleRate * channels * 2);
}
