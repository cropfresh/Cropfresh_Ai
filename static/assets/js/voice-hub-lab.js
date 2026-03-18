(function () {
  const wsState = {
    sessionId: "n/a",
    language: "n/a",
    state: "idle",
    transcript: "",
    reply: "",
    features: [],
    notes: [
      "Duplex is the canonical realtime path, but it does not emit intent or entity metadata yet.",
      "Use the REST tab and inspector when you need route-level validation for listing, pricing, buyer match, or quality flows.",
    ],
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function pushNote(text) {
    if (!text || wsState.notes.includes(text)) return;
    wsState.notes = [text, ...wsState.notes].slice(0, 4);
  }

  function renderWsWorkflow() {
    const root = document.getElementById("wsWorkflowBoard");
    if (!root) return;
    const liveClass = wsState.state === "speaking" || wsState.state === "listening" ? "is-live" : "is-ready";
    const features = wsState.features.length ? wsState.features.join(", ") : "No feature flags received yet";
    root.innerHTML = `
      <div class="workflow-head">
        <div>
          <span class="workflow-kicker">Duplex contract board</span>
          <h3 class="workflow-title">Realtime transport and session visibility</h3>
          <p class="workflow-copy">This board shows the current websocket session, stage transitions, and the contract gap between transport events and route-level metadata.</p>
        </div>
        <span class="workflow-status ${liveClass}">${escapeHtml(wsState.state)}</span>
      </div>
      <div class="workflow-grid">
        <div class="workflow-steps">
          <div class="workflow-step is-done"><strong>Session</strong><span class="workflow-text">${escapeHtml(wsState.sessionId)}</span></div>
          <div class="workflow-step is-done"><strong>Detected language</strong><span class="workflow-text">${escapeHtml(wsState.language)}</span></div>
          <div class="workflow-step ${wsState.state === "thinking" ? "is-active" : "is-done"}"><strong>Pipeline state</strong><span class="workflow-text">${escapeHtml(wsState.state)}</span></div>
          <div class="workflow-step is-pending"><strong>Route metadata</strong><span class="workflow-text">Intent and entities are not part of the duplex contract yet, so use the REST lane when you need that detail.</span></div>
        </div>
        <div class="workflow-side">
          <div class="workflow-pair"><span class="workflow-label">Transcript</span><span class="workflow-value">${escapeHtml(wsState.transcript || "Waiting for transcript_final")}</span></div>
          <div class="workflow-pair"><span class="workflow-label">Agent reply</span><span class="workflow-value">${escapeHtml(wsState.reply || "Waiting for response_sentence")}</span></div>
          <div class="workflow-pair"><span class="workflow-label">Server features</span><span class="workflow-value">${escapeHtml(features)}</span></div>
        </div>
      </div>
      <div class="flow-note">
        <span class="workflow-label">Operator notes</span>
        <ul class="workflow-list">${wsState.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>
      </div>
    `;
  }

  function handleWsMessage(message) {
    switch (message.type) {
      case "ready":
        wsState.sessionId = message.session_id || wsState.sessionId;
        wsState.state = "ready";
        wsState.features = Object.keys(message.features || {}).filter((key) => Boolean(message.features[key]));
        break;
      case "language_detected":
        wsState.language = (message.language || wsState.language).toUpperCase();
        break;
      case "pipeline_state":
        wsState.state = message.state || wsState.state;
        break;
      case "transcript_final":
        wsState.transcript = message.text || wsState.transcript;
        break;
      case "response_sentence":
        wsState.reply = `${wsState.reply} ${message.text || ""}`.trim();
        break;
      case "response_end":
        wsState.reply = message.full_text || wsState.reply;
        wsState.state = "ready";
        break;
      case "bargein":
        wsState.state = "listening";
        pushNote("Barge-in interrupted playback and moved the session back into listening mode.");
        break;
      case "error":
        wsState.state = "error";
        pushNote(message.error || message.text || "Websocket error returned from the duplex pipeline.");
        break;
      default:
        break;
    }
    renderWsWorkflow();
  }

  function resetWsWorkflow() {
    wsState.sessionId = "n/a";
    wsState.language = "n/a";
    wsState.state = "idle";
    wsState.transcript = "";
    wsState.reply = "";
    wsState.features = [];
    wsState.notes = [
      "Duplex is the canonical realtime path, but it does not emit intent or entity metadata yet.",
      "Use the REST tab and inspector when you need route-level validation for listing, pricing, buyer match, or quality flows.",
    ];
    renderWsWorkflow();
  }

  document.addEventListener("DOMContentLoaded", () => {
    window.AgentWorkflows?.renderScenarioCatalog("voiceScenarioCatalog", window.AgentWorkflows.voiceScenarios);
    window.AgentWorkflows?.renderVoiceWorkflow("restWorkflowBoard", null);
    renderWsWorkflow();
  });

  window.VoiceHubLab = {
    handleWsMessage,
    resetWsWorkflow,
  };
})();
