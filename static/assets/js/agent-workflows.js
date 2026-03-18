(function () {
  const workflowData = window.AgentWorkflowData || { voiceFlows: {}, voiceScenarios: [] };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function readEntityValue(entities, keys) {
    for (const key of keys) {
      const value = entities?.[key];
      if (value !== undefined && value !== null && value !== "") {
        return Array.isArray(value) ? value.join(", ") : String(value);
      }
    }
    return "";
  }

  function getFlow(intent) {
    return workflowData.voiceFlows[intent] || {
      title: intent || "Voice route",
      agent: "VoiceAgent",
      downstream: "Follow the returned prompt and inspect the payload",
      required: [],
      note: "This intent is not yet modeled in the static flow map, so use the JSON inspector for full detail.",
    };
  }

  function renderScenarioCatalog(elementId, scenarios = workflowData.voiceScenarios) {
    const root = document.getElementById(elementId);
    if (!root) return;
    root.innerHTML = scenarios
      .map(
        (scenario) => `
          <article class="scenario-card">
            <span class="scenario-meta">${escapeHtml(scenario.meta)}</span>
            <strong>${escapeHtml(scenario.title)}</strong>
            <p class="workflow-text">${escapeHtml(scenario.prompt)}</p>
            <div class="scenario-links"><a href="${escapeHtml(scenario.href)}">${escapeHtml(scenario.label)}</a></div>
          </article>
        `,
      )
      .join("");
  }

  function renderDashboardRoute(elementId, payload, meta = {}) {
    const root = document.getElementById(elementId);
    if (!root) return;
    if (!payload) {
      root.innerHTML = `
        <div class="flow-note">
          <strong class="workflow-title">No routed run yet</strong>
          <p class="workflow-text">Run a pricing, buyer, or quality check below. This board will show the chosen agent, tools, sources, and what to validate next.</p>
        </div>
      `;
      return;
    }
    const confidence = payload.confidence == null ? "n/a" : `${Math.round(payload.confidence * 100)}%`;
    const tools = (payload.tools_used || []).map(escapeHtml);
    const actions = (payload.suggested_actions || []).map(escapeHtml);
    const steps = (payload.steps || []).map(escapeHtml);
    const sources = (payload.sources || []).map(escapeHtml);
    root.innerHTML = `
      <div class="workflow-head">
        <div>
          <span class="workflow-kicker">${escapeHtml(meta.laneLabel || "Chat route")}</span>
          <h3 class="workflow-title">${escapeHtml(payload.agent_used || "Unknown agent")}</h3>
          <p class="workflow-copy">Triggered by: ${escapeHtml(meta.trigger || "Manual dashboard check")}</p>
        </div>
        <span class="workflow-status is-ready">${escapeHtml(meta.latencyLabel || `${meta.latencyMs || 0} ms`)}</span>
      </div>
      <div class="workflow-meta">
        <span class="workflow-tag">Confidence ${escapeHtml(confidence)}</span>
        <span class="workflow-tag">Session ${escapeHtml(payload.session_id || "n/a")}</span>
        ${tools.map((tool) => `<span class="workflow-tag">Tool ${tool}</span>`).join("")}
      </div>
      <div class="workflow-grid">
        <div class="flow-note">
          <span class="workflow-label">Agent answer</span>
          <p class="workflow-text">${escapeHtml(payload.message || "No message returned.")}</p>
          <span class="workflow-label">Suggested next actions</span>
          <ul class="workflow-list">${(actions.length ? actions : ["No suggested follow-ups returned."]).map((item) => `<li>${item}</li>`).join("")}</ul>
        </div>
        <div class="workflow-side">
          <div class="workflow-pair"><span class="workflow-label">Sources</span><span class="workflow-value">${sources.join(", ") || "No sources"}</span></div>
          <div class="workflow-pair"><span class="workflow-label">Steps</span><span class="workflow-value">${steps.join(" -> ") || "No intermediate steps"}</span></div>
        </div>
      </div>
    `;
  }

  function renderVoiceWorkflow(elementId, payload) {
    const root = document.getElementById(elementId);
    if (!root) return;
    if (!payload) {
      root.innerHTML = `
        <div class="flow-note">
          <strong class="workflow-title">No voice route yet</strong>
          <p class="workflow-text">Record a farmer query on the REST tab. This board will show the detected language, intent, missing fields, and downstream service path.</p>
        </div>
      `;
      return;
    }
    const flow = getFlow(payload.intent);
    const missing = flow.required.filter((field) => !readEntityValue(payload.entities, field.keys));
    const lastListingId = payload.workflow_context?.last_listing_id || readEntityValue(payload.entities, ["listing_id"]);
    const handoffLink = lastListingId
      ? `
        <div class="flow-note">
          <span class="workflow-label">Vision handoff</span>
          <p class="workflow-text">Listing ${escapeHtml(lastListingId)} is ready for a grade attach flow in Vision Lab.</p>
          <div class="scenario-links"><a href="./vision_lab.html?listing_id=${encodeURIComponent(lastListingId)}">Open Vision Lab</a></div>
        </div>
      `
      : "";
    const entityRows = Object.entries(payload.entities || {})
      .map(([key, value]) => `<li>${escapeHtml(key)}: ${escapeHtml(Array.isArray(value) ? value.join(", ") : value)}</li>`)
      .join("");
    const stageClass = missing.length ? "is-pending" : "is-ready";
    const stageLabel = missing.length ? `Waiting for ${missing.map((field) => field.label).join(", ")}` : "Ready for downstream handoff";
    root.innerHTML = `
      <div class="workflow-head">
        <div>
          <span class="workflow-kicker">Voice routing board</span>
          <h3 class="workflow-title">${escapeHtml(flow.title)}</h3>
          <p class="workflow-copy">${escapeHtml(flow.note)}</p>
        </div>
        <span class="workflow-status ${stageClass}">${escapeHtml(stageLabel)}</span>
      </div>
      <div class="workflow-meta">
        <span class="workflow-tag">Language ${(payload.language || "unknown").toUpperCase()}</span>
        <span class="workflow-tag">Intent ${escapeHtml(payload.intent || "UNKNOWN")}</span>
        <span class="workflow-tag">Confidence ${payload.confidence == null ? "n/a" : `${Math.round(payload.confidence * 100)}%`}</span>
      </div>
      <div class="workflow-grid">
        <div class="workflow-steps">
          <div class="workflow-step is-done"><strong>Speech captured</strong><span class="workflow-text">${escapeHtml(payload.transcription || "No transcription returned.")}</span></div>
          <div class="workflow-step is-done"><strong>Intent resolved</strong><span class="workflow-text">${escapeHtml(flow.agent)}</span></div>
          <div class="workflow-step ${missing.length ? "is-active" : "is-done"}"><strong>Entity collection</strong><span class="workflow-text">${missing.length ? `Still need ${escapeHtml(missing.map((field) => field.label).join(", "))}` : "Minimum fields are present in the payload."}</span></div>
          <div class="workflow-step ${missing.length ? "is-pending" : "is-done"}"><strong>Downstream handoff</strong><span class="workflow-text">${escapeHtml(flow.downstream)}</span></div>
        </div>
        <div class="workflow-side">
          <div class="workflow-pair"><span class="workflow-label">Session</span><span class="workflow-value">${escapeHtml(payload.session_id || "n/a")}</span></div>
          <div class="workflow-pair"><span class="workflow-label">Last listing</span><span class="workflow-value">${escapeHtml(lastListingId || "Not created in this turn")}</span></div>
          <div class="workflow-pair"><span class="workflow-label">Entities</span><ul class="workflow-list">${entityRows || "<li>No entities extracted yet.</li>"}</ul></div>
          <div class="workflow-pair"><span class="workflow-label">Agent reply</span><span class="workflow-value">${escapeHtml(payload.response_text || "No response text returned.")}</span></div>
        </div>
      </div>
      ${handoffLink}
    `;
  }

  window.AgentWorkflows = {
    renderDashboardRoute,
    renderScenarioCatalog,
    renderVoiceWorkflow,
    voiceScenarios: workflowData.voiceScenarios,
    visionScenarios: workflowData.visionScenarios || [],
  };
})();
