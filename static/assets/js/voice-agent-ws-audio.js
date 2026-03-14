let wsAudioCtx     = null;
let wsProcessor    = null;
let wsMicStream    = null;
let wsPttActive    = false;

async function wsStartMic() {
  if (!wsSocket || wsSocket.readyState !== WebSocket.OPEN) return;
  if (wsPttActive) return;

  wsPttActive = true;
  wsSetStatus('🎙 Recording… (release to send)', 'listening');
  wsAddEvent('vad', '🎙 PTT start');

  try {
    wsMicStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    wsAudioCtx  = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = wsAudioCtx.createMediaStreamSource(wsMicStream);
    wsProcessor  = wsAudioCtx.createScriptProcessor(4096, 1, 1);

    source.connect(wsProcessor);
    wsProcessor.connect(wsAudioCtx.destination);

    wsProcessor.onaudioprocess = (e) => {
      if (!wsSocket || wsSocket.readyState !== WebSocket.OPEN || !wsPttActive) return;
      const f32   = e.inputBuffer.getChannelData(0);
      const i16   = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        i16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32768));
      }
      const bytes = new Uint8Array(i16.buffer);
      let bin = '';
      for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
      const b64 = btoa(bin);
      wsSocket.send(JSON.stringify({ type: 'audio_chunk', audio_base64: b64 }));
    };

    const btn = document.getElementById('btnWsPtt');
    if (btn) { btn.textContent = '🔴 Release to Send'; btn.classList.add('recording'); }

  } catch (err) {
    wsAddEvent('system', `⚠ Mic: ${err.message}`);
    wsPttActive = false;
    wsSetStatus('🟢 Ready — hold PTT to speak', 'connected');
  }
}

function wsStopMic(sendFlush = true) {
  if (!wsPttActive && !wsProcessor) return;

  wsPttActive = false;
  wsProcessor?.disconnect();
  wsAudioCtx?.close();
  wsMicStream?.getTracks().forEach(t => t.stop());
  wsProcessor = null;
  wsAudioCtx  = null;
  wsMicStream = null;

  const btn = document.getElementById('btnWsPtt');
  if (btn) { btn.textContent = '🎤 Hold to Speak (PTT)'; btn.classList.remove('recording'); }

  // Signal that user finished chunk block but don't close socket
  if (sendFlush && wsSocket?.readyState === WebSocket.OPEN) {
    wsSocket.send(JSON.stringify({ type: 'audio_end' }));
    wsAddEvent('vad', '🔇 PTT released → processing');
    wsSetStatus('⏳ Transcribing…', 'thinking');
  }
}
