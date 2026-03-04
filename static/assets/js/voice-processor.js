// assets/js/voice-processor.js
class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 30ms at 16000Hz = 480 samples
    this.bufferSize = 480;
    this.buffer = new Float32Array(this.bufferSize);
    this.bytesWritten = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0];
      for (let i = 0; i < channelData.length; i++) {
        this.buffer[this.bytesWritten++] = channelData[i];
        if (this.bytesWritten >= this.bufferSize) {
          // Send to main thread
          this.port.postMessage(this.buffer);
          this.buffer = new Float32Array(this.bufferSize);
          this.bytesWritten = 0;
        }
      }
    }
    // Keep processor alive
    return true;
  }
}

registerProcessor("voice-processor", VoiceProcessor);
