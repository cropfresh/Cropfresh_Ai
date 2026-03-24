/**
 * voice-agent-tools.js
 * Tools Inspector tab: language tables, health status, and
 * JSON viewer for last voice call (intent / entities / language / confidence).
 */

// ── Inspector state ────────────────────────────────────────────────────────

/** Last full voice result object for display */
let _inspectorPayload = null;

/**
 * Called by voice-agent-rest.js after every successful process call.
 * @param {object} data
 */
function updateToolsInspector(data) {
  _inspectorPayload = data;
  renderInspectorPayload();
}

// ── JSON viewer ────────────────────────────────────────────────────────────

function renderInspectorPayload() {
  const viewer = document.getElementById('inspectorJsonView');
  if (!viewer) return;

  if (!_inspectorPayload) {
    viewer.textContent = '// No voice call yet — run a REST or WS request first.';
    return;
  }

  // Only show the most useful fields
  const subset = {
    transcription: _inspectorPayload.transcription,
    language: _inspectorPayload.language,
    intent: _inspectorPayload.intent,
    confidence: _inspectorPayload.confidence,
    entities: _inspectorPayload.entities,
    session_id: _inspectorPayload.session_id,
    response_text: _inspectorPayload.response_text,
  };
  viewer.textContent = JSON.stringify(subset, null, 2);
}

// ── Copy JSON ──────────────────────────────────────────────────────────────

function copyInspectorJson() {
  const viewer = document.getElementById('inspectorJsonView');
  if (!viewer?.textContent) return;
  navigator.clipboard.writeText(viewer.textContent).then(() => {
    const btn = document.getElementById('btnCopyJson');
    if (btn) {
      btn.textContent = '✓ Copied';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
    }
  }).catch(() => {/* clipboard blocked */});
}

// ── Health status ──────────────────────────────────────────────────────────

async function loadVoiceHealth() {
  const healthRow = document.getElementById('inspectorHealthRow');
  if (!healthRow) return;
  try {
    const res = await requestJson('GET', '/api/v1/voice/health');
    const data = res.data || {};
    const sttProviders = data.stt_providers || [];
    const ttsProvider = data.tts_provider || '?';
    healthRow.innerHTML = `
      <div class="health-chip ${res.ok ? 'ok' : 'err'}">
        ${res.ok ? '✔' : '✘'} Voice API
      </div>
      <div class="health-chip ok">⚙ STT: ${sttProviders.join(', ') || '?'}</div>
      <div class="health-chip ok">🔊 TTS: ${ttsProvider}</div>
      <div class="health-chip">
        🌐 ${(data.languages || []).join(' · ') || '—'}
      </div>
    `;
  } catch (err) {
    healthRow.innerHTML = `<div class="health-chip err">✘ ${err.message}</div>`;
  }
}

// ── Language tables ────────────────────────────────────────────────────────

async function loadLanguageTables() {
  const sttTable = document.getElementById('inspectorSttTable');
  const ttsTable = document.getElementById('inspectorTtsTable');
  if (!sttTable || !ttsTable) return;

  try {
    const res = await requestJson('GET', '/api/v1/voice/languages');
    const data = res.data || {};

    const sttLangs = data.stt_languages || [];
    const ttsLangs = data.tts_languages || [];

    sttTable.innerHTML = buildLangRows(sttLangs);
    ttsTable.innerHTML = buildLangRows(ttsLangs);
  } catch (err) {
    const errRow = `<tr><td colspan="2" style="color:var(--danger)">Error: ${err.message}</td></tr>`;
    sttTable.innerHTML = errRow;
    ttsTable.innerHTML = errRow;
  }
}

/**
 * @param {string[]} codes
 * @returns {string} HTML table rows
 */
function buildLangRows(codes) {
  const NAMES = {
    hi: 'Hindi', kn: 'Kannada', te: 'Telugu', ta: 'Tamil',
    mr: 'Marathi', gu: 'Gujarati', pa: 'Punjabi', bn: 'Bengali',
    ml: 'Malayalam', or: 'Odia', en: 'English', auto: 'Auto-detect',
  };
  if (codes.length === 0) return '<tr><td colspan="2" style="color:var(--text-muted)">—</td></tr>';
  return codes.map((code) => `
    <tr>
      <td>${code}</td>
      <td>${NAMES[code] || code}</td>
    </tr>
  `).join('');
}

// ── Refresh button ─────────────────────────────────────────────────────────

async function refreshInspector() {
  await Promise.all([loadVoiceHealth(), loadLanguageTables()]);
  renderInspectorPayload();
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnRefreshInspector')?.addEventListener('click', () =>
    refreshInspector().catch(() => {/* health error already displayed */}));

  document.getElementById('btnCopyJson')?.addEventListener('click', copyInspectorJson);

  // Auto-load when tab is first opened
  document.querySelectorAll('.tab-btn[data-tab="tabTools"]').forEach((btn) => {
    btn.addEventListener('click', () =>
      refreshInspector().catch(() => {}), { once: true });
  });

  // Initial call for health + languages on page load
  refreshInspector().catch(() => {});
});
