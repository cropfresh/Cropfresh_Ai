const COMFORT_NOISE_WINDOW_MS = 180;
const GRACEFUL_STOP_MS = 120;
const PLAYBACK_SAMPLE_RATE = 24000;
const COMFORT_NOISE_GAIN = 0.0008;

function decodeBase64Audio(base64Audio) {
  return Uint8Array.from(atob(base64Audio), (char) => char.charCodeAt(0));
}

function createComfortNoiseBuffer(context) {
  const frameCount = Math.round((PLAYBACK_SAMPLE_RATE * COMFORT_NOISE_WINDOW_MS) / 1000);
  const buffer = context.createBuffer(1, frameCount, PLAYBACK_SAMPLE_RATE);
  const channel = buffer.getChannelData(0);

  for (let index = 0; index < frameCount; index += 1) {
    channel[index] = index % 2 === 0 ? COMFORT_NOISE_GAIN : -COMFORT_NOISE_GAIN;
  }

  return buffer;
}

export function createPlaybackController({ onPlaybackIdle }) {
  let playbackContext = null;
  let playbackQueue = [];
  let currentGain = null;
  let currentSource = null;
  let comfortNoiseSource = null;
  let comfortNoiseTimer = null;
  let isPlaying = false;
  let suppressIdleFill = false;

  function ensurePlaybackContext() {
    if (!playbackContext) {
      playbackContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: PLAYBACK_SAMPLE_RATE,
      });
    }
    return playbackContext;
  }

  function clearComfortNoiseTimer() {
    if (comfortNoiseTimer !== null) {
      clearTimeout(comfortNoiseTimer);
      comfortNoiseTimer = null;
    }
  }

  function stopComfortNoise() {
    clearComfortNoiseTimer();
    try {
      comfortNoiseSource?.stop();
    } catch {
      // Ignore InvalidStateError for one-shot sources that already finished.
    }
    comfortNoiseSource = null;
  }

  function finishPlaybackIdle() {
    isPlaying = false;
    onPlaybackIdle();
  }

  function maybePlayComfortNoise() {
    if (suppressIdleFill) {
      finishPlaybackIdle();
      return;
    }

    const context = ensurePlaybackContext();
    stopComfortNoise();
    comfortNoiseSource = context.createBufferSource();
    comfortNoiseSource.buffer = createComfortNoiseBuffer(context);
    comfortNoiseSource.connect(context.destination);
    comfortNoiseSource.start(0);
    comfortNoiseTimer = setTimeout(() => {
      stopComfortNoise();
      finishPlaybackIdle();
    }, COMFORT_NOISE_WINDOW_MS);
  }

  async function playNextChunk() {
    const nextChunk = playbackQueue.shift();
    if (!nextChunk) {
      maybePlayComfortNoise();
      return;
    }

    stopComfortNoise();
    isPlaying = true;
    try {
      const context = ensurePlaybackContext();
      const decoded = await context.decodeAudioData(decodeBase64Audio(nextChunk).buffer.slice(0));
      const gainNode = context.createGain();
      const source = context.createBufferSource();
      currentSource = source;
      currentGain = gainNode;
      source.buffer = decoded;
      source.connect(gainNode);
      gainNode.connect(context.destination);
      source.onended = () => {
        if (currentSource === source) {
          currentSource = null;
          currentGain = null;
          void playNextChunk();
        }
      };
      source.start(0);
    } catch (error) {
      console.warn("[Duplex] Audio decode error:", error);
      currentSource = null;
      currentGain = null;
      void playNextChunk();
    }
  }

  function queue(base64Audio) {
    suppressIdleFill = false;
    playbackQueue.push(base64Audio);
    if (!isPlaying) {
      void playNextChunk();
    }
  }

  function stop() {
    playbackQueue = [];
    suppressIdleFill = true;
    stopComfortNoise();
    if (!currentSource || !currentGain || !playbackContext) {
      currentSource?.stop();
      currentSource = null;
      currentGain = null;
      finishPlaybackIdle();
      return;
    }

    const source = currentSource;
    const gainNode = currentGain;
    const now = playbackContext.currentTime;
    gainNode.gain.cancelScheduledValues(now);
    gainNode.gain.setValueAtTime(gainNode.gain.value, now);
    gainNode.gain.linearRampToValueAtTime(0.0001, now + GRACEFUL_STOP_MS / 1000);
    setTimeout(() => {
      if (currentSource === source) {
        currentSource = null;
        currentGain = null;
      }
      try {
        source.stop();
      } catch {
        // Ignore InvalidStateError when the source already ended naturally.
      }
      finishPlaybackIdle();
    }, GRACEFUL_STOP_MS);
  }

  return {
    queue,
    stop,
  };
}
