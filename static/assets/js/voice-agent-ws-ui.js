// UI logic for WS tab
function wsGetChat() {
  return document.getElementById("wsChat");
}

function wsAppendBubble(role, text) {
  const chat = wsGetChat();
  if (!chat) return null;
  const wrap = document.createElement("div");
  wrap.className = `chat-bubble-wrap ${role}`;
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function wsUpdateBubble(el, text) {
  if (el) el.textContent += text;
  const chat = wsGetChat();
  if (chat) chat.scrollTop = chat.scrollHeight;
}

function wsSetStatus(text, cls = "") {
  const el = document.getElementById("wsLiveStatus");
  if (!el) return;
  el.textContent = text;
  el.className = `ws-status-bar ${cls}`;
}

function wsAddEvent(type, text) {
  const tl = document.getElementById("wsTimeline");
  if (!tl) return;
  const now = new Date().toLocaleTimeString("en-US", { hour12: false });
  const row = document.createElement("div");
  row.className = "event-row";
  row.innerHTML = `<span class="event-dot ${type}"></span><span class="event-time">${now}</span><span class="event-text">${text}</span>`;
  tl.appendChild(row);
  tl.scrollTop = tl.scrollHeight;
}
