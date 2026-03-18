// * Dashboard behavior and quick API checks.

/**
 * Loads system health and updates the dashboard chips.
 * @returns {Promise<void>}
 */
async function loadDashboardHealth() {
  const healthResponse = await requestJson('GET', '/health');
  const readyResponse = await requestJson('GET', '/health/ready');
  setStatusPill('apiStatusPill', readyResponse.ok, readyResponse.ok ? 'API Ready' : 'API Not Ready');
  const readyChecks = readyResponse.data?.checks ?? {};
  const llmState = readyChecks.llm_initialized ? 'Ready' : 'Not Ready';
  const redisState = readyChecks.redis ? 'Connected' : 'Off';
  const supervisorState = readyChecks.supervisor ? 'Up' : 'Down';
  document.getElementById('kpiLlm').textContent = llmState;
  document.getElementById('kpiRedis').textContent = redisState;
  document.getElementById('kpiSupervisor').textContent = supervisorState;
  renderResult('healthResult', {
    health: healthResponse.data,
    ready: readyResponse.data
  }, !healthResponse.ok || !readyResponse.ok);
}

/**
 * Performs a sample RAG query from the landing page.
 * @returns {Promise<void>}
 */
async function runQuickRagQuery() {
  const question = document.getElementById('quickRagQuestion').value.trim();
  if (!question) {
    renderResult('quickRagResult', 'Please enter a question.', true);
    return;
  }
  const response = await requestJson('POST', '/api/v1/rag/query', { question });
  renderResult('quickRagResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

/**
 * Performs a quick pricing question through chat route.
 * @returns {Promise<void>}
 */
async function runQuickPricingCheck() {
  const commodity = document.getElementById('quickCommodity').value.trim() || 'Tomato';
  const location = document.getElementById('quickLocation').value.trim() || 'Kolar';
  const message = `What is the current price of ${commodity} in ${location}?`;
  const response = await requestJson('POST', '/api/v1/chat', { message });
  renderResult('quickPricingResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
  window.AgentWorkflows?.renderDashboardRoute('dashboardRouteBoard', response.data, {
    laneLabel: 'Quick pricing check',
    trigger: message,
    latencyMs: response.ms,
  });
}

/**
 * Performs a quick buyer matching request through chat route.
 * @returns {Promise<void>}
 */
async function runQuickBuyerMatch() {
  const commodity = document.getElementById('quickMatchCommodity').value.trim() || 'Tomato';
  const quantity = Number(document.getElementById('quickMatchQuantity').value) || 100;
  const message = `Find buyer matches for ${quantity} kg ${commodity} listing`;
  const response = await requestJson('POST', '/api/v1/chat', { message });
  renderResult('quickMatchResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
  window.AgentWorkflows?.renderDashboardRoute('dashboardRouteBoard', response.data, {
    laneLabel: 'Buyer matching check',
    trigger: message,
    latencyMs: response.ms,
  });
}

/**
 * Performs a quick quality grading request through chat route.
 * @returns {Promise<void>}
 */
async function runQuickQualityCheck() {
  const commodity = document.getElementById('quickQualityCommodity').value.trim() || 'Tomato';
  const description = document.getElementById('quickQualityDescription').value.trim();
  if (!description) {
    renderResult('quickQualityResult', 'Please add produce condition details.', true);
    return;
  }
  const response = await requestJson('POST', '/api/v1/vision/assess', {
    commodity,
    description,
  });
  renderResult('quickQualityResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
  const routePayload = response.ok ? {
    agent_used: 'quality_assessment_agent',
    confidence: response.data.confidence,
    message: response.data.message,
    suggested_actions: response.data.hitl_required
      ? ['Review the result with HITL before trusting the grade.', 'Open Vision Lab for image-based validation.']
      : ['Attach this grade to a listing in Vision Lab.', 'Compare the result with a voice-created listing if needed.'],
    steps: ['vision_assess', response.data.assessment_mode, 'listing_grade_ready'],
    sources: [response.data.vision_ready ? 'vision_models' : 'rule_based_fallback'],
    session_id: response.data.assessment_id,
  } : null;
  window.AgentWorkflows?.renderDashboardRoute('dashboardRouteBoard', routePayload, {
    laneLabel: 'Vision grading check',
    trigger: `Assess ${commodity} quality from dashboard`,
    latencyMs: response.ms,
  });
}

document.addEventListener('DOMContentLoaded', () => {
  window.AgentWorkflows?.renderScenarioCatalog('dashboardScenarioCatalog', window.AgentWorkflows.voiceScenarios);
  window.AgentWorkflows?.renderDashboardRoute('dashboardRouteBoard', null);
  loadDashboardHealth().catch((err) => {
    renderResult('healthResult', `Error: ${err.message}`, true);
    setStatusPill('apiStatusPill', false, 'API Error');
  });
  document.getElementById('btnRefreshHealth').addEventListener('click', () => {
    loadDashboardHealth().catch((err) => renderResult('healthResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnQuickRag').addEventListener('click', () => {
    runQuickRagQuery().catch((err) => renderResult('quickRagResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnQuickPricing').addEventListener('click', () => {
    runQuickPricingCheck().catch((err) => renderResult('quickPricingResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnQuickMatch').addEventListener('click', () => {
    runQuickBuyerMatch().catch((err) => renderResult('quickMatchResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnQuickQuality').addEventListener('click', () => {
    runQuickQualityCheck().catch((err) => renderResult('quickQualityResult', `Error: ${err.message}`, true));
  });
});
