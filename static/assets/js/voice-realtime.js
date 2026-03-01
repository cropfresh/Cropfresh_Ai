// * ChatGPT-style live streaming UI using /api/v1/chat/stream SSE.
let streamAbortController = null;
let streamIsActive = false;

/**
 * Adds a chat bubble and returns the element.
 * @param {'user'|'assistant'|'system'} role
 * @param {string} text
 * @returns {HTMLDivElement}
 */
function appendMessage(role, text) {
  const streamElement = document.getElementById('chatStream');
  const messageElement = document.createElement('div');
  messageElement.className = `chat-message ${role}`;
  messageElement.textContent = text;
  streamElement.appendChild(messageElement);
  streamElement.scrollTop = streamElement.scrollHeight;
  return messageElement;
}

/**
 * Parses SSE chunks from fetch stream and calls message handler.
 * @param {ReadableStreamDefaultReader<Uint8Array>} reader
 * @param {(msg: any) => void} onMessage
 * @returns {Promise<void>}
 */
async function readSseStream(reader, onMessage) {
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';
    for (const event of events) {
      const dataLine = event.split('\n').find((line) => line.startsWith('data: '));
      if (!dataLine) {
        continue;
      }
      const jsonPayload = dataLine.slice(6);
      try {
        onMessage(JSON.parse(jsonPayload));
      } catch (_error) {
        onMessage({ type: 'error', content: `Invalid stream payload: ${jsonPayload}` });
      }
    }
  }
}

/**
 * Streams chat response token-by-token.
 * @param {string} prompt
 * @returns {Promise<void>}
 */
async function streamChat(prompt) {
  if (streamIsActive) {
    appendMessage('system', 'A stream is already in progress. Stop it first.');
    return;
  }
  streamAbortController = new AbortController();
  streamIsActive = true;
  setStatusPill('wsStatusPill', true, 'Streaming...');

  appendMessage('user', prompt);
  const assistantMessage = appendMessage('assistant', '');
  assistantMessage.classList.add('typing-caret');

  try {
    const response = await fetch(`${window.location.origin}/api/v1/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: prompt }),
      signal: streamAbortController.signal
    });

    if (!response.ok || !response.body) {
      throw new Error(`Stream request failed (${response.status})`);
    }

    await readSseStream(response.body.getReader(), (message) => {
      if (message.type === 'session' && message.session_id) {
        document.getElementById('streamSession').textContent = `Session: ${message.session_id}`;
      } else if (message.type === 'agent' && message.agent) {
        document.getElementById('streamAgent').textContent = `Agent: ${message.agent}`;
      } else if (message.type === 'token') {
        assistantMessage.textContent += message.content || '';
      } else if (message.type === 'done') {
        assistantMessage.classList.remove('typing-caret');
        setStatusPill('wsStatusPill', true, 'Stream Complete');
      } else if (message.type === 'error') {
        assistantMessage.classList.remove('typing-caret');
        appendMessage('system', `Error: ${message.content || 'Unknown stream error'}`);
        setStatusPill('wsStatusPill', false, 'Stream Error');
      }
      const streamElement = document.getElementById('chatStream');
      streamElement.scrollTop = streamElement.scrollHeight;
    });
  } catch (error) {
    assistantMessage.classList.remove('typing-caret');
    if (error.name === 'AbortError') {
      appendMessage('system', 'Stream stopped by user.');
      setStatusPill('wsStatusPill', false, 'Stopped');
    } else {
      appendMessage('system', `Stream failed: ${error.message}`);
      setStatusPill('wsStatusPill', false, 'Failed');
    }
  } finally {
    assistantMessage.classList.remove('typing-caret');
    streamIsActive = false;
    streamAbortController = null;
  }
}

/**
 * Stops active stream if any.
 */
function stopStream() {
  if (!streamIsActive || !streamAbortController) {
    appendMessage('system', 'No active stream to stop.');
    return;
  }
  streamAbortController.abort();
}

/**
 * Clears all chat bubbles and metadata.
 */
function clearStream() {
  if (streamIsActive) {
    stopStream();
  }
  document.getElementById('chatStream').innerHTML = '';
  document.getElementById('streamAgent').textContent = 'Agent: —';
  document.getElementById('streamSession').textContent = 'Session: —';
  setStatusPill('wsStatusPill', false, 'Idle');
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnSendStream').addEventListener('click', () => {
    const promptInput = document.getElementById('streamPrompt');
    const prompt = promptInput.value.trim();
    if (!prompt) {
      appendMessage('system', 'Enter a message before sending.');
      return;
    }
    streamChat(prompt).catch((error) => {
      appendMessage('system', `Unexpected stream error: ${error.message}`);
      setStatusPill('wsStatusPill', false, 'Failed');
    });
  });
  document.getElementById('btnStopStream').addEventListener('click', stopStream);
  document.getElementById('btnClearStream').addEventListener('click', clearStream);
  document.getElementById('streamPrompt').addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      document.getElementById('btnSendStream').click();
    }
  });
});
