// * Shared helpers for static UIs.
const BASE_API_URL = window.location.origin;

/**
 * Executes a JSON API request and returns payload + metadata.
 * @param {string} method
 * @param {string} path
 * @param {Record<string, unknown> | null} body
 * @returns {Promise<{ok: boolean, status: number, data: any, ms: number}>}
 */
async function requestJson(method, path, body = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json'
    }
  };
  if (body) {
    options.body = JSON.stringify(body);
  }
  const start = performance.now();
  const response = await fetch(`${BASE_API_URL}${path}`, options);
  const latency = Math.round(performance.now() - start);
  let responseData = null;
  try {
    responseData = await response.json();
  } catch (_err) {
    responseData = { detail: 'Response is not valid JSON.' };
  }
  return {
    ok: response.ok,
    status: response.status,
    data: responseData,
    ms: latency
  };
}

/**
 * Renders result JSON/text into a target container.
 * @param {string} elementId
 * @param {unknown} value
 * @param {boolean} isError
 */
function renderResult(elementId, value, isError = false) {
  const targetElement = document.getElementById(elementId);
  if (!targetElement) {
    return;
  }
  const outputText = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  targetElement.className = `result${isError ? ' error' : ''}`;
  targetElement.textContent = outputText;
}

/**
 * Updates status badge classes and label.
 * @param {string} elementId
 * @param {boolean} isOnline
 * @param {string} label
 */
function setStatusPill(elementId, isOnline, label) {
  const pillElement = document.getElementById(elementId);
  if (!pillElement) {
    return;
  }
  pillElement.classList.remove('online', 'offline');
  pillElement.classList.add(isOnline ? 'online' : 'offline');
  const textElement = pillElement.querySelector('[data-status-text]');
  if (textElement) {
    textElement.textContent = label;
  }
}
