let latestVisionAssessment = null;

function escapeVisionHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fieldValue(id) {
  return document.getElementById(id)?.value.trim() || "";
}

function setFieldValue(id, value) {
  const element = document.getElementById(id);
  if (element && !element.value && value) element.value = value;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",").pop() || "");
    reader.onerror = () => reject(new Error("Could not read image file."));
    reader.readAsDataURL(file);
  });
}

function renderVisionBoard(payload) {
  const root = document.getElementById("visionPipelineBoard");
  if (!root) return;
  if (!payload) {
    root.innerHTML = `
      <div class="flow-note">
        <strong class="workflow-title">No vision run yet</strong>
        <p class="workflow-text">Run an assessment with a description or image. This board will show model mode, HITL, and the listing-grade handoff.</p>
      </div>
    `;
    return;
  }
  const sourceLabel = payload.vision_ready ? "Vision models active" : "Rule-based fallback";
  const attachReady = payload.listing_id && payload.listing_id !== "vision-lab-preview";
  root.innerHTML = `
    <div class="workflow-head">
      <div>
        <span class="workflow-kicker">Vision pipeline board</span>
        <h3 class="workflow-title">${escapeVisionHtml(payload.commodity)} graded ${escapeVisionHtml(payload.grade)}</h3>
        <p class="workflow-copy">${escapeVisionHtml(payload.message)}</p>
      </div>
      <span class="workflow-status ${payload.hitl_required ? "is-pending" : "is-ready"}">${payload.hitl_required ? "HITL review" : "Ready to attach"}</span>
    </div>
    <div class="workflow-meta">
      <span class="workflow-tag">Mode ${escapeVisionHtml(payload.assessment_mode)}</span>
      <span class="workflow-tag">Confidence ${Math.round((payload.confidence || 0) * 100)}%</span>
      <span class="workflow-tag">${escapeVisionHtml(sourceLabel)}</span>
      <span class="workflow-tag">Shelf life ${escapeVisionHtml(payload.shelf_life_days)}d</span>
    </div>
    <div class="workflow-grid">
      <div class="workflow-steps">
        <div class="workflow-step is-done"><strong>Intake</strong><span class="workflow-text">Commodity and context captured for ${escapeVisionHtml(payload.commodity)}.</span></div>
        <div class="workflow-step is-done"><strong>Assessment</strong><span class="workflow-text">${escapeVisionHtml(payload.assessment_mode)} produced grade ${escapeVisionHtml(payload.grade)} with ${escapeVisionHtml(payload.defect_count)} defect(s).</span></div>
        <div class="workflow-step ${payload.hitl_required ? "is-active" : "is-done"}"><strong>Review gate</strong><span class="workflow-text">${payload.hitl_required ? "Human verification is recommended before upgrading trust." : "No manual review required for this result."}</span></div>
        <div class="workflow-step ${attachReady ? "is-done" : "is-pending"}"><strong>Listing attach</strong><span class="workflow-text">${attachReady ? `Listing ${escapeVisionHtml(payload.listing_id)} is ready for /listings/{id}/grade.` : "Provide a real listing id to attach this grade."}</span></div>
      </div>
      <div class="workflow-side">
        <div class="workflow-pair"><span class="workflow-label">Defects</span><span class="workflow-value">${escapeVisionHtml((payload.defects || []).join(", ") || "No visible defects")}</span></div>
        <div class="workflow-pair"><span class="workflow-label">Assessment ID</span><span class="workflow-value">${escapeVisionHtml(payload.assessment_id)}</span></div>
        <div class="workflow-pair"><span class="workflow-label">Digital twin</span><span class="workflow-value">${payload.digital_twin_linked ? "Linked" : "Not linked"}</span></div>
      </div>
    </div>
  `;
}

function renderVoiceHandoff() {
  const root = document.getElementById("visionVoiceHandoff");
  const handoff = window.LabState?.readVoiceHandoff();
  if (!root) return;
  if (!handoff) {
    root.innerHTML = "Run a voice listing in Voice Hub first if you want a real listing id carried into this page.";
    return;
  }
  setFieldValue("visionListingId", handoff.last_listing_id);
  setFieldValue("attachListingId", handoff.last_listing_id);
  setFieldValue("visionCommodity", handoff.commodity);
  root.innerHTML = `
    <strong>${escapeVisionHtml(handoff.intent || "voice_flow")}</strong>
    <span>${escapeVisionHtml(handoff.transcription || "No transcript stored.")}</span>
    <span>Listing: ${escapeVisionHtml(handoff.last_listing_id || "not created yet")}</span>
    <span>Session: ${escapeVisionHtml(handoff.session_id)}</span>
  `;
}

async function loadVisionHealth() {
  const response = await requestJson("GET", "/api/v1/vision/health");
  setStatusPill(
    "visionStatusPill",
    response.ok && response.data.service_ready,
    response.ok ? (response.data.vision_ready ? "Vision Ready" : "Fallback Mode") : "Vision Error",
  );
  renderResult("visionHealthResult", { latency_ms: response.ms, payload: response.data }, !response.ok);
}

async function runVisionAssessment() {
  const imageFile = document.getElementById("visionImage")?.files?.[0];
  const payload = {
    commodity: fieldValue("visionCommodity") || "Tomato",
    listing_id: fieldValue("visionListingId") || null,
    description: fieldValue("visionDescription"),
    require_upgrade_review: Boolean(document.getElementById("visionUpgradeReview")?.checked),
  };
  if (imageFile) payload.image_b64 = await fileToBase64(imageFile);
  const response = await requestJson("POST", "/api/v1/vision/assess", payload);
  renderResult("visionAssessResult", { latency_ms: response.ms, payload: response.data }, !response.ok);
  if (!response.ok) return;
  latestVisionAssessment = response.data;
  window.LabState?.saveVisionAssessment(response.data);
  setFieldValue("attachListingId", response.data.listing_id);
  renderVisionBoard(response.data);
}

async function attachVisionGrade() {
  const listingId = fieldValue("attachListingId");
  if (!listingId) {
    renderResult("visionAttachResult", "Provide a listing id before attaching the grade.", true);
    return;
  }
  if (!latestVisionAssessment?.grade_attach_preview) {
    renderResult("visionAttachResult", "Run a vision assessment first so there is a grade to attach.", true);
    return;
  }
  const response = await requestJson(
    "POST",
    `/api/v1/listings/${encodeURIComponent(listingId)}/grade`,
    latestVisionAssessment.grade_attach_preview,
  );
  renderResult("visionAttachResult", { latency_ms: response.ms, payload: response.data }, !response.ok);
}

document.addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(window.location.search);
  window.AgentWorkflows?.renderScenarioCatalog("visionScenarioCatalog", window.AgentWorkflows.visionScenarios);
  setFieldValue("visionListingId", params.get("listing_id") || "");
  setFieldValue("attachListingId", params.get("listing_id") || "");
  renderVoiceHandoff();
  renderVisionBoard(window.LabState?.readVisionAssessment());
  latestVisionAssessment = window.LabState?.readVisionAssessment();
  setFieldValue("attachListingId", latestVisionAssessment?.listing_id);
  loadVisionHealth().catch((error) => renderResult("visionHealthResult", `Error: ${error.message}`, true));
  document.getElementById("btnVisionAssess")?.addEventListener("click", () => {
    runVisionAssessment().catch((error) => renderResult("visionAssessResult", `Error: ${error.message}`, true));
  });
  document.getElementById("btnAttachVisionGrade")?.addEventListener("click", () => {
    attachVisionGrade().catch((error) => renderResult("visionAttachResult", `Error: ${error.message}`, true));
  });
});
