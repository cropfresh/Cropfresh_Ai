// * Voice UI test page using REST endpoints.
let recorder = null;
let recordedChunks = [];
let isRecording = false;

/**
 * Pushes message to chat stream.
 * @param {'user'|'assistant'} role
 * @param {string} text
 */
function addVoiceMessage(role, text) {
  const streamElement = document.getElementById('voiceChatStream');
  const messageElement = document.createElement('div');
  messageElement.className = `msg ${role}`;
  messageElement.textContent = text;
  streamElement.appendChild(messageElement);
  streamElement.scrollTop = streamElement.scrollHeight;
}

/**
 * Checks voice health endpoint.
 * @returns {Promise<void>}
 */
async function checkVoiceHealth() {
  const response = await requestJson('GET', '/api/v1/voice/health');
  renderResult('voiceHealthResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

/**
 * Processes recorded blob through voice/process endpoint.
 * @param {Blob} blob
 * @returns {Promise<void>}
 */
async function processRecordedVoice(blob) {
  const languageValue = document.getElementById('voiceLanguage').value;
  const formData = new FormData();
  formData.append('audio', blob, 'voice.webm');
  formData.append('language', languageValue);
  formData.append('user_id', 'voice-ui-user');
  const start = performance.now();
  const response = await fetch(`${BASE_API_URL}/api/v1/voice/process`, {
    method: 'POST',
    body: formData
  });
  const latency = Math.round(performance.now() - start);
  const data = await response.json();
  renderResult('voiceProcessResult', {
    latency_ms: latency,
    payload: data
  }, !response.ok);
  if (data.transcription) {
    addVoiceMessage('user', data.transcription);
  }
  if (data.response_text) {
    addVoiceMessage('assistant', data.response_text);
  }
}

/**
 * Starts microphone capture.
 * @returns {Promise<void>}
 */
async function startRecording() {
  const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  recorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm' });
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  recorder.onstop = async () => {
    mediaStream.getTracks().forEach((track) => track.stop());
    const blob = new Blob(recordedChunks, { type: 'audio/webm' });
    await processRecordedVoice(blob);
    document.getElementById('voiceState').textContent = 'Idle';
  };
  recorder.start();
  isRecording = true;
  document.getElementById('btnRecordToggle').textContent = 'Stop Recording';
  document.getElementById('voiceState').textContent = 'Recording...';
}

/**
 * Stops active recording session.
 */
function stopRecording() {
  if (recorder && recorder.state !== 'inactive') {
    recorder.stop();
  }
  isRecording = false;
  document.getElementById('btnRecordToggle').textContent = 'Start Recording';
}

/**
 * Toggles microphone recording state.
 * @returns {Promise<void>}
 */
async function toggleRecording() {
  if (isRecording) {
    stopRecording();
    return;
  }
  try {
    await startRecording();
  } catch (err) {
    renderResult('voiceProcessResult', `Error: ${err.message}`, true);
  }
}

/**
 * Sends text for TTS synthesis.
 * @returns {Promise<void>}
 */
async function synthesizeText() {
  const textValue = document.getElementById('ttsText').value.trim();
  const languageValue = document.getElementById('voiceLanguage').value;
  if (!textValue) {
    renderResult('ttsResult', 'Please enter text.', true);
    return;
  }
  const response = await requestJson('POST', '/api/v1/voice/synthesize', {
    text: textValue,
    language: languageValue
  });
  renderResult('ttsResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
  const base64Audio = response.data?.audio_base64;
  if (base64Audio) {
    const audioElement = document.getElementById('ttsAudio');
    audioElement.src = `data:audio/wav;base64,${base64Audio}`;
    audioElement.style.display = 'block';
    audioElement.play().catch(() => undefined);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnVoiceHealth').addEventListener('click', () => {
    checkVoiceHealth().catch((err) => renderResult('voiceHealthResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnRecordToggle').addEventListener('click', () => {
    toggleRecording().catch((err) => renderResult('voiceProcessResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnTts').addEventListener('click', () => {
    synthesizeText().catch((err) => renderResult('ttsResult', `Error: ${err.message}`, true));
  });
});
