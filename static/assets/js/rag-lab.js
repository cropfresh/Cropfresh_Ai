// * RAG lab interactions with separated JS logic.

/**
 * Activates selected RAG tab panel.
 * @param {string} panelId
 */
function switchRagTab(panelId) {
  document.querySelectorAll('.tab-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === panelId);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === panelId);
  });
}

/**
 * Routes text through adaptive router endpoint.
 * @returns {Promise<void>}
 */
async function runRouterTest() {
  const query = document.getElementById('routerQuery').value.trim();
  if (!query) {
    renderResult('routerResult', 'Please provide a query.', true);
    return;
  }
  const response = await requestJson('POST', '/api/v1/rag/route', { query });
  renderResult('routerResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

/**
 * Normalizes bilingual agri terms through API.
 * @returns {Promise<void>}
 */
async function runNormalizer() {
  const text = document.getElementById('normalizeInput').value.trim();
  if (!text) {
    renderResult('normalizeResult', 'Please provide text to normalize.', true);
    return;
  }
  const response = await requestJson('POST', '/api/v1/rag/normalize', { text });
  renderResult('normalizeResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

/**
 * Submits a RAG query and displays structured response.
 * @returns {Promise<void>}
 */
async function runRagQuery() {
  const question = document.getElementById('ragQuestion').value.trim();
  const context = document.getElementById('ragContext').value.trim();
  if (!question) {
    renderResult('queryResult', 'Please provide a question.', true);
    return;
  }
  const body = context ? { question, context } : { question };
  const response = await requestJson('POST', '/api/v1/rag/query', body);
  renderResult('queryResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

/**
 * Runs semantic search with optional top_k.
 * @returns {Promise<void>}
 */
async function runRagSearch() {
  const query = document.getElementById('searchQuery').value.trim();
  const topK = Number(document.getElementById('searchTopK').value || 5);
  if (!query) {
    renderResult('searchResult', 'Please provide search query.', true);
    return;
  }
  const response = await requestJson('GET', `/api/v1/rag/search?query=${encodeURIComponent(query)}&top_k=${topK}`);
  renderResult('searchResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

/**
 * Loads KB stats endpoint.
 * @returns {Promise<void>}
 */
async function loadKbStats() {
  const response = await requestJson('GET', '/api/v1/rag/stats');
  renderResult('statsResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

/**
 * Sends one document for manual ingestion.
 * @returns {Promise<void>}
 */
async function ingestDocument() {
  const text = document.getElementById('ingestText').value.trim();
  const source = document.getElementById('ingestSource').value.trim() || 'manual';
  const category = document.getElementById('ingestCategory').value;
  if (!text) {
    renderResult('ingestResult', 'Document text cannot be empty.', true);
    return;
  }
  const response = await requestJson('POST', '/api/v1/rag/ingest', {
    documents: [{ text, source, category }]
  });
  renderResult('ingestResult', {
    latency_ms: response.ms,
    payload: response.data
  }, !response.ok);
}

document.addEventListener('DOMContentLoaded', () => {
  switchRagTab('tab-router');
  document.querySelectorAll('.tab-btn').forEach((button) => {
    button.addEventListener('click', () => switchRagTab(button.dataset.tab));
  });
  document.getElementById('btnRoute').addEventListener('click', () => {
    runRouterTest().catch((err) => renderResult('routerResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnNormalize').addEventListener('click', () => {
    runNormalizer().catch((err) => renderResult('normalizeResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnQuery').addEventListener('click', () => {
    runRagQuery().catch((err) => renderResult('queryResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnSearch').addEventListener('click', () => {
    runRagSearch().catch((err) => renderResult('searchResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnStats').addEventListener('click', () => {
    loadKbStats().catch((err) => renderResult('statsResult', `Error: ${err.message}`, true));
  });
  document.getElementById('btnIngest').addEventListener('click', () => {
    ingestDocument().catch((err) => renderResult('ingestResult', `Error: ${err.message}`, true));
  });
});
