const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 480;
const WORKLET_URL = new URL("../voice-processor.js", import.meta.url);

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
  let playbackContext = null;
  let playbackQueue = [];
  let currentSource = null;
  let isPlaying = false;
  let isRecording = false;

  function emitChunk(float32Data) {
    if (!isRecording) {
      return;
    }
    const base64 = arrayBufferToBase64(float32ToInt16(float32Data).buffer);
    onAudioChunk(base64);
  }

  function ensurePlaybackContext() {
    if (!playbackContext) {
      playbackContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000,
      });
    }
    return playbackContext;
  }

  async function playNextChunk() {
    const nextChunk = playbackQueue.shift();
    if (!nextChunk) {
      isPlaying = false;
      onPlaybackIdle();
      return;
    }

    isPlaying = true;
    try {
      const context = ensurePlaybackContext();
      const audioBytes = Uint8Array.from(atob(nextChunk), (char) => char.charCodeAt(0));
      const decoded = await context.decodeAudioData(audioBytes.buffer.slice(0));
      const source = context.createBufferSource();
      currentSource = source;
      source.buffer = decoded;
      source.connect(context.destination);
      source.onended = () => {
        currentSource = null;
        void playNextChunk();
      };
      source.start(0);
    } catch (error) {
      console.warn("[Duplex] Audio decode error:", error);
      currentSource = null;
      void playNextChunk();
    }
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
      } catch (error) {
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
      playbackQueue.push(base64Audio);
      if (!isPlaying) {
        void playNextChunk();
      }
    },
    startCapture,
    async stopCapture() {
      await stopCapture();
    },
    stopPlayback() {
      playbackQueue = [];
      currentSource?.stop();
      currentSource = null;
      isPlaying = false;
    },
  };
}
