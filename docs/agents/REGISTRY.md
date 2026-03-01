# Agent Registry - CropFresh AI

| Agent | Module | Status | Description |
|-------|--------|--------|-------------|
| Supervisor | src/agents/supervisor_agent.py | Active | Routes queries to specialist agents |
| Agronomy | src/agents/agronomy_agent.py | Active | Agricultural knowledge & advisory |
| Pricing | src/agents/pricing_agent.py | Active | Market price analysis & prediction |
| Commerce | src/agents/commerce_agent.py | Active | Transaction & marketplace operations |
| Voice | src/agents/voice_agent.py | Active | Kannada voice interaction |
| Knowledge | src/agents/knowledge_agent.py | Active | RAG-based knowledge retrieval |
| Platform | src/agents/platform_agent.py | Active | Platform admin operations |
| General | src/agents/general_agent.py | Active | General queries & fallback |
| Browser | src/agents/browser_agent.py | Active | Web browsing & scraping |
| Research | src/agents/research/ | Active | Multi-step research tasks |
| Web Scraping | src/agents/web_scraping_agent.py | Active | Data collection from web |
| Crop Listing | src/agents/crop_listing/ | ? Active (Task 7) | NL listing creation + ListingService: auto-price, shelf-life expiry, QR, ADCL tag, quality trigger |
| Quality Assessment | src/agents/quality_assessment/ | ✅ Active (Task 3) | CV-based grading + HITL threshold + digital twin linkage |
| **Digital Twin Engine** | src/agents/digital_twin/ | ✅ Active (Task 10) | Departure snapshot, arrival diff, SSIM/pHash/rule-based similarity, 6-rule liability matrix |
| Buyer Matching | src/agents/buyer_matching/ | ? Active (Task 2) | 5-factor scoring, reverse matching, 5-min cache |
|| Price Prediction | src/agents/price_prediction/ | ? Active (Task 5) | Rule-based + numpy trend + Karnataka seasonal calendar + LLM fallback |
| Registration & Auth | src/api/services/registration_service.py + src/api/routers/auth.py | Active (Task 9) | OTP flow, stdlib JWT, farmer/buyer profile CRUD, voice compat |
| WhatsApp Bot | src/agents/whatsapp_bot/ | Planned | WhatsApp conversation agent |
