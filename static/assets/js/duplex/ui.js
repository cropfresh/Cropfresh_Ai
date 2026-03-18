const STATE_LABELS = {
  idle: "Ready",
  listening: "Listening...",
  thinking: "Thinking...",
  speaking: "Speaking...",
  interrupted: "Interrupted",
  error: "Error",
  connecting: "Connecting...",
};

const STATE_COPY = {
  idle: {
    title: "Ready for live voice",
    body: "Tap the orb or connect the session to start a duplex conversation.",
  },
  listening: {
    title: "Listening for the farmer",
    body: "Speak naturally. The live session is capturing your next turn.",
  },
  thinking: {
    title: "Thinking through the reply",
    body: "The agent is transcribing, reasoning, and preparing the first audio chunk.",
  },
  speaking: {
    title: "Speaking with premium voice",
    body: "Audio is streaming back now. Watch the metrics and transcript update live.",
  },
  interrupted: {
    title: "Interrupted for barge-in",
    body: "The previous reply was cut so the next farmer utterance can start immediately.",
  },
  error: {
    title: "Session needs attention",
    body: "Reconnect the live session and try again once the voice link is healthy.",
  },
  connecting: {
    title: "Connecting the live agent",
    body: "Opening the duplex websocket and preparing the session for live speech.",
  },
};

const $ = (selector) => document.querySelector(selector);

function renderMetric(id, label, value) {
  const node = $(`#${id}`);
  if (!node) {
    return;
  }

  const suffix = value == null ? "--ms" : `${Math.round(value)}ms`;
  node.textContent = `${label} ${suffix}`;
}

export function createDuplexUI({ getSocketState }) {
  function updateState(state, previousState) {
    document.body.dataset.voiceState = state;

    const voiceOrb = $("#voiceOrb");
    if (voiceOrb) {
      voiceOrb.className = "voice-orb";
      if (!["idle", "error", "connecting"].includes(state)) {
        voiceOrb.classList.add(state);
      }
    }

    const transcriptAgent = $("#transcriptAgent");
    if (transcriptAgent && state === "connecting") {
      transcriptAgent.textContent = "Connecting to the duplex voice agent...";
    }

    const stageCopy = STATE_COPY[state] || STATE_COPY.idle;
    const voiceStageLabel = $("#voiceStageLabel");
    if (voiceStageLabel) {
      voiceStageLabel.textContent = stageCopy.title;
    }

    const voiceStatusCopy = $("#voiceStatusCopy");
    if (voiceStatusCopy) {
      voiceStatusCopy.textContent = stageCopy.body;
    }

    const connectBtnText = $("#btnConnectText");
    if (connectBtnText) {
      const socketOpen = getSocketState() === WebSocket.OPEN;
      if (state === "connecting") {
        connectBtnText.textContent = "Connecting...";
      } else if (socketOpen) {
        connectBtnText.textContent = "Disconnect";
      } else {
        connectBtnText.textContent = "Connect";
      }
    }

    const wsStatusPill = $("#wsStatusPill");
    if (wsStatusPill) {
      const socketOpen = getSocketState() === WebSocket.OPEN;
      wsStatusPill.dataset.state = socketOpen ? state : "offline";
      wsStatusPill.textContent = socketOpen ? STATE_LABELS[state] || "Live" : "Offline";
    }

    const connectButton = $("#btnConnectToggle");
    if (connectButton) {
      connectButton.setAttribute("aria-label", STATE_LABELS[state] || state);
    }

    const feedbackPanel = $("#feedbackPanel");
    if (feedbackPanel) {
      if (state === "idle" && previousState === "speaking") {
        feedbackPanel.classList.add("visible");
      }
      if (["listening", "thinking", "connecting"].includes(state)) {
        feedbackPanel.classList.remove("visible");
      }
    }
  }

  function updateMetrics(timing = {}) {
    renderMetric("metricLatency", "FIRST", timing.first_audio_ms);
    renderMetric("metricTranscription", "STT", timing.transcription_ms);
    renderMetric("metricTotal", "TOTAL", timing.total_ms);
  }

  function updateTransport(bootstrap = {}) {
    const modePill = $("#bridgeModePill");
    if (!modePill) {
      return;
    }

    const mode = bootstrap.mode || "fallback_ws";
    modePill.dataset.transportMode = mode;
    modePill.textContent = mode === "bridge" ? "Bridge bootstrap" : "Fallback WS";
  }

  function addChatBubble(role, text) {
    const userEl = $("#transcriptUser");
    const agentEl = $("#transcriptAgent");
    if (role === "user" && userEl) {
      userEl.textContent = text;
    }
    if (["agent", "system"].includes(role) && agentEl) {
      agentEl.textContent = text;
    }
  }

  function hideFeedback() {
    $("#feedbackPanel")?.classList.remove("visible");
  }

  function log(message) {
    const timeline = $(".event-timeline");
    if (timeline) {
      const row = document.createElement("div");
      row.className = "event-row";
      const time = document.createElement("span");
      time.className = "event-time";
      time.textContent = new Date().toLocaleTimeString();
      const text = document.createElement("span");
      text.className = "event-text";
      text.textContent = message;
      row.append(time, text);
      timeline.appendChild(row);
      timeline.scrollTop = timeline.scrollHeight;
    }
    console.log(`[Duplex] ${message}`);
  }

  return {
    addChatBubble,
    hideFeedback,
    log,
    query: $,
    updateMetrics,
    updateTransport,
    updateState,
  };
}
