import { describe, expect, it } from "vitest";
import { SharedRingBuffer, secondsToPcm16Bytes } from "../src/audio/ring-buffer.js";
describe("SharedRingBuffer", () => {
    it("keeps only the latest bytes when writes exceed capacity", () => {
        const buffer = new SharedRingBuffer(8);
        buffer.write(Uint8Array.from([1, 2, 3, 4, 5]));
        buffer.write(Uint8Array.from([6, 7, 8, 9]));
        expect(Array.from(buffer.readLatest(8))).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
    });
    it("reads a shorter latest snapshot without mutating state", () => {
        const buffer = new SharedRingBuffer(8);
        buffer.write(Uint8Array.from([10, 11, 12, 13]));
        expect(Array.from(buffer.readLatest(2))).toEqual([12, 13]);
        expect(buffer.snapshot().sizeBytes).toBe(4);
    });
    it("calculates five seconds of 16kHz PCM16 audio as 160000 bytes", () => {
        expect(secondsToPcm16Bytes(5)).toBe(160000);
    });
});
