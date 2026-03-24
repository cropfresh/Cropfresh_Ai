const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 480;
const WORKLET_URL = new URL("../voice-processor.js", import.meta.url);

import { createPlaybackController } from "./playback.js";

function float32ToInt16(float32Array) {
  const int16 = new Int16Array(float32Array.length);
  for (let index = 0; index < float32Array.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, float32Array[index]));
    int16[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return int16;
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.byteLength; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
}

export function createDuplexAudio({ onAudioChunk, onError, onPlaybackIdle }) {
  let mediaStream = null;
  let audioContext = null;
  let workletNode = null;
  let fallbackProcessor = null;
  let isRecording = false;

  const playback = createPlaybackController({ onPlaybackIdle });

  function emitChunk(float32Data) {
    if (!isRecording) {
      return;
    }
    const base64 = arrayBufferToBase64(float32ToInt16(float32Data).buffer);
    onAudioChunk(base64);
  }

  async function startCapture() {
    if (isRecording) {
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      const source = audioContext.createMediaStreamSource(mediaStream);

      try {
        await audioContext.audioWorklet.addModule(WORKLET_URL);
        workletNode = new AudioWorkletNode(audioContext, "voice-processor");
        workletNode.port.onmessage = ({ data }) => emitChunk(data);
        source.connect(workletNode);
        workletNode.connect(audioContext.destination);
      } catch {
        fallbackProcessor = audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);
        fallbackProcessor.onaudioprocess = ({ inputBuffer }) => {
          emitChunk(inputBuffer.getChannelData(0));
        };
        source.connect(fallbackProcessor);
        fallbackProcessor.connect(audioContext.destination);
      }

      isRecording = true;
    } catch (error) {
      onError(error);
      throw error;
    }
  }

  async function stopCapture() {
    if (!isRecording) {
      return;
    }

    isRecording = false;
    mediaStream?.getTracks().forEach((track) => track.stop());
    mediaStream = null;
    workletNode?.disconnect();
    workletNode = null;
    fallbackProcessor?.disconnect();
    fallbackProcessor = null;

    if (audioContext) {
      await audioContext.close();
      audioContext = null;
    }
  }

  return {
    isRecording: () => isRecording,
    queuePlayback(base64Audio) {
      playback.queue(base64Audio);
    },
    startCapture,
    async stopCapture() {
      await stopCapture();
    },
    stopPlayback() {
      playback.stop();
    },
  };
}
