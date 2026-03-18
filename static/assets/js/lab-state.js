(function () {
  const STORAGE_KEY = "cropfresh-lab-state";

  function readState() {
    try {
      return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "{}");
    } catch (_error) {
      return {};
    }
  }

  function writeState(partialState) {
    const nextState = { ...readState(), ...partialState };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(nextState));
  }

  function saveVoiceHandoff(payload) {
    writeState({
      voiceHandoff: {
        session_id: payload.session_id || "",
        intent: payload.intent || "",
        commodity: payload.entities?.commodity || payload.entities?.crop || "",
        transcription: payload.transcription || "",
        response_text: payload.response_text || "",
        last_listing_id: payload.workflow_context?.last_listing_id || payload.entities?.listing_id || "",
        pending_intent: payload.workflow_context?.pending_intent || "",
        updated_at: new Date().toISOString(),
      },
    });
  }

  function saveVisionAssessment(payload) {
    writeState({
      visionAssessment: {
        ...payload,
        updated_at: new Date().toISOString(),
      },
    });
  }

  window.LabState = {
    readVoiceHandoff: () => readState().voiceHandoff || null,
    readVisionAssessment: () => readState().visionAssessment || null,
    saveVisionAssessment,
    saveVoiceHandoff,
  };
})();
