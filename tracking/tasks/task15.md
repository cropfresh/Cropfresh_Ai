# Task 15: Implement WhatsApp Bot Agent

> **Priority:** 🟡 P2 | **Phase:** 4 | **Effort:** 3–4 days  
> **Files:** `src/agents/whatsapp_bot/agent.py`, `src/api/routers/whatsapp.py` [NEW]  
> **Score Target:** 9/10 — Full WhatsApp integration with text, voice notes, and images

---

## 📌 Problem Statement

`src/agents/whatsapp_bot/agent.py` is a stub. Need to implement Twilio/Meta WhatsApp Business API integration for farmers who prefer WhatsApp over the app.

---

## 🔬 Research Findings (2025 WhatsApp Business API)

### API Architecture
- **Meta Cloud API** is now standard (on-premise deprecated)
- Per-delivered-template-message pricing (since July 2025)
- Webhook-based: Meta → Your Server → Response
- Supports: text, images, documents, voice notes, location, buttons

### Message Types
| Type | Use Case | Rate |
|------|----------|------|
| Template (outbound) | Price alerts, listing confirmations | ₹0.50-1.50/msg |
| Session (within 24h) | Conversational replies | Free |
| Voice notes | Farmer sends voice → STT → agent | Free in session |
| Images | Produce photos for grading | Free in session |

---

## 🏗️ Implementation Spec

### Webhook Handler
```python
# src/api/routers/whatsapp.py

@router.get("/webhooks/whatsapp")
async def verify_webhook(request: Request):
    """Meta webhook verification challenge."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if mode == "subscribe" and token == settings.WA_VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")
    return Response(status_code=403)

@router.post("/webhooks/whatsapp")
async def handle_message(request: Request):
    """Process incoming WhatsApp message."""
    body = await request.json()
    
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            messages = change["value"].get("messages", [])
            for msg in messages:
                await whatsapp_bot.process_message(msg)
    
    return {"status": "ok"}
```

### Bot Agent
```python
class WhatsAppBotAgent:
    """
    WhatsApp bot with multi-modal message handling.
    
    Supports:
    - Text messages → Route to appropriate agent
    - Voice notes → STT → Agent → Text response
    - Images → Quality assessment agent
    - Location → Set farmer GPS
    """
    
    async def process_message(self, message: dict):
        msg_type = message.get("type")
        sender = message["from"]
        
        if msg_type == "text":
            text = message["text"]["body"]
            response = await self._handle_text(sender, text)
        elif msg_type == "audio":
            audio_url = message["audio"]["url"]
            response = await self._handle_voice_note(sender, audio_url)
        elif msg_type == "image":
            image_url = message["image"]["url"]
            caption = message["image"].get("caption", "")
            response = await self._handle_image(sender, image_url, caption)
        elif msg_type == "location":
            lat = message["location"]["latitude"]
            lon = message["location"]["longitude"]
            response = await self._handle_location(sender, lat, lon)
        else:
            response = "Sorry, I can't process this type of message yet."
        
        await self._send_reply(sender, response)
    
    async def _send_reply(self, to: str, message: str):
        """Send reply via Twilio WhatsApp API."""
        from twilio.rest import Client
        client = Client(settings.TWILIO_SID, settings.TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=f"whatsapp:{settings.TWILIO_WA_NUMBER}",
            to=f"whatsapp:{to}",
        )
```

### Template Messages (Outbound Notifications)
```python
TEMPLATES = {
    'price_alert': "🌾 Price Alert: {commodity} is now ₹{price}/kg in {market}. {trend}.",
    'listing_matched': "✅ Your {qty}kg {commodity} listing has a buyer! {buyer_district}. Reply YES to confirm.",
    'order_update': "📦 Order #{order_id}: Status updated to {status}.",
    'weekly_demand': "📊 This week's hot crops: {crops}. Send DETAILS for more info.",
}
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Webhook verification + message receipt works | 20% |
| 2 | Text messages route to correct agent | 20% |
| 3 | Voice notes → STT → agent response | 20% |
| 4 | Image → quality assessment trigger | 15% |
| 5 | Template messages for price alerts and order updates | 15% |
| 6 | Rate limiting to prevent API cost overrun | 10% |
