let wsSocket = null;
let wsSessionActive = false;
let wsAgentMsgEl = null;

// Relies on AudioPlayer from shared.js, and wsUI / wsAudio objects
const wsPlayer = new AudioPlayer();

function wsConnect() {
  if (wsSocket && wsSocket.readyState === WebSocket.OPEN) return;
  const userId = document.getElementById('wsUserId')?.value || 'ws-test-user';
  const lang   = document.getElementById('wsLang')?.value || 'hi';
  const proto  = location.protocol === 'https:' ? 'wss' : 'ws';
  
  const url = `${proto}://${location.host}/api/v1/voice/ws/duplex?user_id=${encodeURIComponent(userId)}&language=${lang}`;
  wsSetStatus('🔄 Connecting…', 'connecting');
  wsSocket = new WebSocket(url);

  wsSocket.onopen = () => {
    wsSessionActive = true;
    setStatusPill('wsStatusPill', true, 'Connected');
    wsAddEvent('system', `✔ Connected · user=${userId}`);
    wsSetStatus('🟢 Connected — hold PTT to speak', 'connected');

    document.getElementById('btnWsConnect')?.setAttribute('disabled', '');
    document.getElementById('btnWsDisconnect')?.removeAttribute('disabled');
    document.getElementById('btnWsPtt')?.removeAttribute('disabled');
  };

  wsSocket.onclose = () => {
    wsSessionActive = false;
    setStatusPill('wsStatusPill', false, 'Disconnected');
    wsAddEvent('system', '✘ Disconnected');
    wsSetStatus('⚫ Disconnected', '');
    wsStopMic(false);
    document.getElementById('btnWsConnect')?.removeAttribute('disabled');
    document.getElementById('btnWsDisconnect')?.setAttribute('disabled', '');
    document.getElementById('btnWsPtt')?.setAttribute('disabled', '');
  };

  wsSocket.onerror = () => {
    wsAddEvent('system', '⚠ WebSocket error');
    wsSetStatus('🔴 Error', 'error');
  };

  wsSocket.onmessage = wsHandleMessage;
}

function wsDisconnect() {
  wsStopMic(false);
  wsSocket?.close();
}

function wsHandleMessage(event) {
  let msg;
  try { msg = JSON.parse(event.data); } catch { return; }

  switch (msg.type) {
    case 'ready':
      wsAddEvent('system', `▶ Ready (session: ${msg.session_id})`);
      break;

    case 'language_detected':
      wsAddEvent('system', `🗣 Detected Language: ${msg.language.toUpperCase()}${msg.locked ? ' (Locked)' : ''}`);
      const langBadge = document.getElementById('wsDetectedLangBadge');
      if (langBadge) {
        langBadge.textContent = msg.language.toUpperCase() + (msg.locked ? ' 🔒' : '');
        langBadge.classList.remove('d-none');
      }
      if (msg.locked) {
        const langSelect = document.getElementById('wsLang');
        if (langSelect && langSelect.value !== msg.language) {
          langSelect.value = msg.language;
        }
      }
      break;

    case 'pipeline_state':
      if (msg.state === 'listening') wsSetStatus('🎙 Listening…', 'listening');
      else if (msg.state === 'thinking') wsSetStatus('⏳ Thinking…', 'thinking');
      else if (msg.state === 'speaking') wsSetStatus('🔊 Speaking…', 'speaking');
      break;

    case 'transcript_final':
      wsAppendBubble('user', msg.text);
      wsAddEvent('vad', `✔ You: ${msg.text}`);
      break;

    case 'response_sentence':
      if (!wsAgentMsgEl) {
        wsAgentMsgEl = wsAppendBubble('agent', msg.text + " ");
      } else {
        wsUpdateBubble(wsAgentMsgEl, msg.text + " ");
      }
      wsAddEvent('response', `💬 ${msg.text}`);
      break;

    case 'response_audio':
      if (msg.audio_base64) {
        wsPlayer.enqueue(msg.audio_base64, 'audio/mpeg');
      }
      break;

    case 'response_end':
      wsAddEvent('response', `✔ Done`);
      wsSetStatus('🟢 Ready — hold PTT to speak', 'connected');
      wsAgentMsgEl = null;
      break;

    case 'bargein':
      wsAddEvent('bargein', '⚡ Barge-in — response cancelled');
      wsPlayer.clearQueue();
      wsSetStatus('🎙 Listening…', 'listening');
      wsAgentMsgEl = null;
      break;

    case 'error':
      wsAddEvent('system', `⚠ ${msg.text}`);
      wsAppendBubble('system', `Error: ${msg.text}`);
      wsSetStatus('🟢 Ready — hold PTT to speak', 'connected');
      break;
    
    default:
      if (msg.type) wsAddEvent('system', `? ${msg.type}`);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnWsConnect')?.addEventListener('click', wsConnect);
  document.getElementById('btnWsDisconnect')?.addEventListener('click', wsDisconnect);

  const pttBtn = document.getElementById('btnWsPtt');
  if (pttBtn) {
    pttBtn.addEventListener('mousedown', () => wsStartMic().catch(e => wsAddEvent('system', `⚠ ${e.message}`)));
    pttBtn.addEventListener('mouseup',   () => wsStopMic(true));
    pttBtn.addEventListener('mouseleave',() => { if (wsPttActive) wsStopMic(true); });
    pttBtn.addEventListener('touchstart', e => { e.preventDefault(); wsStartMic().catch(e => wsAddEvent('system', `⚠ ${e.message}`)); });
    pttBtn.addEventListener('touchend',   e => { e.preventDefault(); wsStopMic(true); });
  }

  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body && wsSessionActive && !wsPttActive) {
      e.preventDefault();
      wsStartMic().catch(() => {});
    }
  });
  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && wsPttActive) {
      e.preventDefault();
      wsStopMic(true);
    }
  });

  document.getElementById('btnWsClearTimeline')?.addEventListener('click', () => {
    const tl = document.getElementById('wsTimeline');
    if (tl) tl.innerHTML = '';
    const chat = wsGetChat();
    if (chat) chat.innerHTML = '';
  });
});
