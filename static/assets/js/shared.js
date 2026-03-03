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
      "Content-Type": "application/json",
    },
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
    responseData = { detail: "Response is not valid JSON." };
  }
  return {
    ok: response.ok,
    status: response.status,
    data: responseData,
    ms: latency,
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
  const outputText =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
  targetElement.className = `result${isError ? " error" : ""}`;
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
  pillElement.classList.remove("online", "offline");
  pillElement.classList.add(isOnline ? "online" : "offline");
  const textElement = pillElement.querySelector("[data-status-text]");
  if (textElement) {
    textElement.textContent = label;
  }
}

/**
 * Decodes a base64 string to a Blob URL for audio playback.
 * @param {string} b64
 * @param {string} mime
 * @returns {string} objectURL
 */
function base64ToBlob(b64, mime = "audio/wav") {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

/**
 * Plays audio from an object URL; resolves when playback ends.
 * @param {string} objectUrl
 * @returns {Promise<void>}
 */
function playAudioBlob(objectUrl) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(objectUrl);
    audio.onended = () => {
      URL.revokeObjectURL(objectUrl);
      resolve();
    };
    audio.onerror = (e) => {
      URL.revokeObjectURL(objectUrl);
      reject(e);
    };
    audio.play().catch(reject);
  });
}

/**
 * Formats a duration in seconds.
 * @param {number} seconds
 * @returns {string} e.g. '1.24s'
 */
function formatDuration(seconds) {
  if (seconds == null) return "—";
  return `${seconds.toFixed(2)}s`;
}
