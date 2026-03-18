// REST tab: record mic, call /api/v1/voice/process, then render chat plus route board.
const restRecorder = new MicRecorder();
const restVisualizer = (() => {
  const canvas = document.getElementById("restWaveform");
  return canvas ? new MicVisualizer(canvas) : null;
})();
const restPlayer = new AudioPlayer();
let lastVoiceResult = null;
let restSessionId = sessionStorage.getItem("voice_rest_session_id") || crypto.randomUUID();
sessionStorage.setItem("voice_rest_session_id", restSessionId);
function restGetChat() { return document.getElementById("restChat"); }
function restAppendBubble(role, text) {
  const chat = restGetChat();
  if (!chat) return;
  const wrap = document.createElement("div");
  const bubble = document.createElement("div");
  wrap.className = `chat-bubble-wrap ${role}`;
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}
function restSetStatus(message, statusClass = "") {
  const statusElement = document.getElementById("restStatus");
  if (!statusElement) return;
  statusElement.textContent = message;
  statusElement.className = `rest-status ${statusClass}`;
}
function setEl(id, text, className) {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = text;
  if (className) element.className = className;
}
function renderBadgeStrip(id, badges) {
  const strip = document.getElementById(id);
  if (!strip) return;
  strip.innerHTML = "";
  badges.forEach(({ cls, label }) => {
    const chip = document.createElement("span");
    chip.className = `badge ${cls}`;
    chip.textContent = label;
    strip.appendChild(chip);
  });
}
function appendEntityBadges(id, entities) {
  const strip = document.getElementById(id);
  if (!strip) return;
  Object.entries(entities || {}).forEach(([key, value]) => {
    const chip = document.createElement("span");
    chip.className = "badge badge-entity";
    chip.textContent = `${key}: ${Array.isArray(value) ? value.join(", ") : value}`;
    strip.appendChild(chip);
  });
}
function setMicOrbState(isRecording) {
  const orb = document.getElementById("restMicOrb");
  const label = document.getElementById("restMicLabel");
  if (!orb) return;
  orb.classList.toggle("recording", isRecording);
  if (!label) return;
  label.textContent = isRecording ? "Recording... tap to stop" : "Tap to record";
  label.classList.toggle("recording", isRecording);
}
function renderRestResult(data) {
  lastVoiceResult = data;
  window.LabState?.saveVoiceHandoff(data);
  restAppendBubble("user", data.transcription?.trim() || "(no speech detected)");
  restAppendBubble("agent", data.response_text?.trim() || "(no response)");
  renderBadgeStrip(
    "restBadges",
    [
      data.language ? { cls: "badge-lang", label: `LANG ${data.language.toUpperCase()}` } : null,
      data.intent ? { cls: "badge-intent", label: `INTENT ${data.intent}` } : null,
      data.confidence != null ? { cls: "badge-conf", label: `CONF ${Math.round(data.confidence * 100)}%` } : null,
    ].filter(Boolean),
  );
  appendEntityBadges("restBadges", data.entities || {});
  if (data.response_audio_base64) {
    restPlayer.enqueue(data.response_audio_base64, "audio/mpeg");
    restSetStatus("Playing response...", "speaking");
  } else {
    restSetStatus("Done", "");
  }
  if (typeof updateToolsInspector === "function") updateToolsInspector(data);
  window.AgentWorkflows?.renderVoiceWorkflow("restWorkflowBoard", data);
}
function renderRestError(message) {
  restAppendBubble("system", `Error: ${message}`);
  restSetStatus("Error", "error");
}
async function toggleRestRecording() {
  if (restRecorder.isRecording) {
    restRecorder.stop();
    restVisualizer?.stop();
    setMicOrbState(false);
    return;
  }
  try {
    await restRecorder.start();
    setMicOrbState(true);
    restSetStatus("Recording... tap orb to stop", "listening");
    restVisualizer?.start(restRecorder.getStream());
  } catch (error) {
    renderRestError(error.message);
    setMicOrbState(false);
  }
}
restRecorder.onStop = async (blob) => {
  restSetStatus("Processing voice...", "thinking");
  const language = document.getElementById("restLang")?.value || "auto";
  const form = new FormData();
  form.append("audio", blob, "voice.webm");
  form.append("language", language);
  form.append("user_id", "voice-hub-user");
  form.append("session_id", restSessionId);
  try {
    const start = performance.now();
    const response = await fetch(`${BASE_API_URL}/api/v1/voice/process`, { method: "POST", body: form });
    const latencyMs = Math.round(performance.now() - start);
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    const data = await response.json();
    if (data.session_id) {
      restSessionId = data.session_id;
      sessionStorage.setItem("voice_rest_session_id", restSessionId);
    }
    renderRestResult(data);
    setEl("restLatency", `-> ${latencyMs}ms`, "");
  } catch (error) {
    renderRestError(error.message);
  }
};
async function transcribeOnly() {
  const button = document.getElementById("btnTranscribeToggle");
  if (!button) return;
  if (restRecorder.isRecording) {
    restRecorder.stop();
    restVisualizer?.stop();
    setMicOrbState(false);
    button.textContent = "Transcribe only";
    return;
  }
  const originalOnStop = restRecorder.onStop;
  restRecorder.onStop = async (blob) => {
    restRecorder.onStop = originalOnStop;
    const form = new FormData();
    form.append("audio", blob, "voice.webm");
    form.append("language", document.getElementById("restLang")?.value || "auto");
    restSetStatus("Transcribing...", "thinking");
    try {
      const response = await fetch(`${BASE_API_URL}/api/v1/voice/transcribe`, { method: "POST", body: form });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      restAppendBubble("user", data.text || "(empty)");
      renderBadgeStrip("restBadges", [
        { cls: "badge-lang", label: `LANG ${(data.language || "?").toUpperCase()}` },
        { cls: "badge-conf", label: `CONF ${Math.round((data.confidence || 0) * 100)}%` },
        { cls: "badge-provider", label: `STT ${data.provider || "?"}` },
      ]);
      restSetStatus("Transcription done", "");
      button.textContent = "Transcribe only";
      window.AgentWorkflows?.renderVoiceWorkflow("restWorkflowBoard", null);
    } catch (error) {
      renderRestError(error.message);
    }
  };
  try {
    await restRecorder.start();
    setMicOrbState(true);
    restSetStatus("Recording (transcribe only)...", "listening");
    restVisualizer?.start(restRecorder.getStream());
    button.textContent = "Stop transcription";
  } catch (error) {
    renderRestError(error.message);
  }
}
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("restMicOrb")?.addEventListener("click", () => toggleRestRecording().catch((error) => renderRestError(error.message)));
  document.getElementById("btnTranscribeToggle")?.addEventListener("click", () => transcribeOnly().catch((error) => renderRestError(error.message)));
});
