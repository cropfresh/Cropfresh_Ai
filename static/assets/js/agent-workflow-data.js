(function () {
  window.AgentWorkflowData = {
    voiceFlows: {
      CREATE_LISTING: {
        title: "Voice listing",
        agent: "VoiceAgent -> listing_service",
        downstream: "listing_service.create_listing()",
        required: [
          { label: "crop", keys: ["crop", "commodity"] },
          { label: "quantity", keys: ["quantity", "quantity_kg"] },
          { label: "asking price", keys: ["asking_price", "price"] },
        ],
        note: "Listing stays in multi-turn collection mode until crop, quantity, and asking price are present.",
      },
      CHECK_PRICE: {
        title: "Price discovery",
        agent: "VoiceAgent -> pricing_agent",
        downstream: "pricing_agent.get_recommendation()",
        required: [{ label: "crop", keys: ["crop", "commodity"] }],
        note: "If location is missing, the handler falls back to Kolar for a quick answer.",
      },
      FIND_BUYER: {
        title: "Buyer matching",
        agent: "VoiceAgent -> matching_agent",
        downstream: "matching_agent.find_matches()",
        required: [
          { label: "commodity", keys: ["commodity", "crop"] },
          { label: "quantity", keys: ["quantity_kg", "quantity"] },
        ],
        note: "Buyer matching reuses the last listing id when available, or a temporary voice id during testing.",
      },
      QUALITY_CHECK: {
        title: "Quality grading",
        agent: "VoiceAgent -> quality_agent",
        downstream: "quality_agent.execute()",
        required: [{ label: "commodity", keys: ["commodity", "crop"] }],
        note: "The quality path can request human review if the downstream agent marks the result as uncertain.",
      },
      GET_ADVISORY: {
        title: "Agronomy advisory",
        agent: "VoiceAgent -> agronomy_agent",
        downstream: "agronomy_agent.process()",
        required: [{ label: "crop", keys: ["crop", "commodity"] }],
        note: "The advisory handler converts the voice ask into a short agronomy query and keeps the reply brief for speech.",
      },
    },
    voiceScenarios: [
      {
        title: "Voice listing flow",
        meta: "/api/v1/voice/process -> VoiceAgent -> listing_service",
        prompt: "Say: list 120 kg tomato at 24 rupees per kilo in Kannada or English.",
        href: "./voice_agent.html",
        label: "Open Voice Hub",
      },
      {
        title: "Voice price flow",
        meta: "/api/v1/voice/process -> VoiceAgent -> pricing_agent",
        prompt: "Say: what is tomato price in Kolar today?",
        href: "./voice_agent.html",
        label: "Run REST route",
      },
      {
        title: "Buyer matching flow",
        meta: "/api/v1/chat or voice -> matching agent",
        prompt: "Use voice for sell intent, then confirm buyer matching in the dashboard.",
        href: "./index.html",
        label: "Open Dashboard",
      },
      {
        title: "Quality escalation",
        meta: "VoiceAgent -> quality_agent -> HITL decision",
        prompt: "Ask for a quality check after the listing exists so the workflow board shows the handoff.",
        href: "./voice_agent.html",
        label: "Inspect route",
      },
      {
        title: "Vision grade attach",
        meta: "/api/v1/vision/assess -> /api/v1/listings/{id}/grade",
        prompt: "Run a voice listing, then open Vision Lab to assess produce quality and attach the grade to the stored listing.",
        href: "./vision_lab.html",
        label: "Open Vision Lab",
      },
    ],
    visionScenarios: [
      {
        title: "Voice listing to vision grade",
        meta: "Voice Hub -> session handoff -> Vision Lab",
        prompt: "Create a listing in Voice Hub first, then use the carried listing id here for the grade attach step.",
        href: "./voice_agent.html",
        label: "Open Voice Hub",
      },
      {
        title: "Image grading path",
        meta: "/api/v1/vision/assess",
        prompt: "Upload a produce image to verify whether the shared quality agent is running in full vision mode or rule-based fallback.",
        href: "./vision_lab.html",
        label: "Run Vision Lab",
      },
    ],
  };
})();
