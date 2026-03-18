import { describe, expect, it } from "vitest";

import { computeNormalizedRms, evaluateRmsGate } from "../src/audio/rms-gate.js";

describe("RMS gate", () => {
  it("returns zero rms for empty payloads", () => {
    expect(computeNormalizedRms(new Int16Array())).toBe(0);
  });

  it("marks energetic audio as active", () => {
    const samples = new Int16Array([0, 12000, -12000, 6000]);
    const result = evaluateRmsGate(samples, 0.1);

    expect(result.isActive).toBe(true);
    expect(result.rms).toBeGreaterThan(0.1);
  });

  it("marks quiet audio as inactive", () => {
    const samples = new Int16Array([0, 10, -10, 5]);
    const result = evaluateRmsGate(samples, 0.01);

    expect(result.isActive).toBe(false);
  });
});
