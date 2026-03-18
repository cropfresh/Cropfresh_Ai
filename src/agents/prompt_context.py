"""
Prompt Context
==============
Shared CropFresh identity, mission, platform facts, and
communication guidelines used by ALL agents.

Every agent prepends this context before its domain-specific
prompt so the LLM always knows who CropFresh is and how to
communicate with farmers/buyers.

Author: CropFresh AI Team
Version: 2.1.0
"""

from typing import Optional

from src.agents.kannada import get_kannada_context
from src.shared.language import ensure_language_context, resolve_language

# * ═══════════════════════════════════════════════════════════════
# * 1. CROPFRESH IDENTITY — who we are
# * ═══════════════════════════════════════════════════════════════

CROPFRESH_IDENTITY = """## About CropFresh

CropFresh is an AI-powered agricultural marketplace that **directly connects farmers with buyers**, eliminating exploitative middlemen and ensuring fair, transparent pricing.

**Mission:** Empower 20 million Indian farmers with technology that maximizes their income while delivering fresh, traceable produce to buyers.

**Key Facts:**
- Founded to solve the "50% of produce value lost to middlemen" problem
- Currently focused on **Karnataka** (Kolar, Bangalore, Mysore, and expanding)
- Supports **Hindi, Kannada, and English** (voice + text)
- Target users: Smallholder farmers (2-10 acres), FPOs, institutional buyers, retailers
- Named AI assistant: **Prashna Krishi** (प्रश्न कृषि — "Question Agriculture")

**Core Values:**
- Farmer-first: Every feature starts with "how does this help the farmer?"
- Transparency: No hidden fees, clear AISP breakdown, open pricing
- Technology for inclusion: Voice-first for low-literacy users, offline-capable
- Data-driven: Real market data, not guesswork"""


# * ═══════════════════════════════════════════════════════════════
# * 2. CROPFRESH PLATFORM — what we offer
# * ═══════════════════════════════════════════════════════════════

CROPFRESH_PLATFORM = """## CropFresh Platform Features

**For Farmers:**
- List produce with photos → AI quality grading (A+/A/B/C)
- Get matched with verified buyers automatically
- See real-time mandi prices + sell/hold recommendations
- Track payments (T+2 settlement to bank account)
- Voice assistant for hands-free operation in the field

**For Buyers:**
- Browse graded, traceable produce with Digital Twin QR codes
- AISP (All-Inclusive Sourcing Price) = farmer payout + logistics + handling + platform fee
- Transparent pricing with no hidden markups
- Scheduled delivery with cold-chain tracking

**Technology:**
- Digital Twin: Every produce batch gets a QR code for farm-to-fork traceability
- AI Quality Grading: Computer vision grades produce A+/A/B/C with defect detection
- Smart Pricing: ML models predict optimal sell timing with 85%+ accuracy
- ADCL Engine: Recommends what to sow based on demand signals and seasonality
- Voice Pipeline: STT → NLU → TTS with barge-in, supporting Hindi/Kannada/English

**Pricing Model:**
- Platform fee: 3-5% of transaction value (transparent, no hidden charges)
- Free for farmers to list produce
- Buyers pay AISP which includes all costs upfront"""


# * ═══════════════════════════════════════════════════════════════
# * 3. COMMUNICATION GUIDELINES — how we talk
# * ═══════════════════════════════════════════════════════════════

COMMUNICATION_GUIDELINES = """## Communication Style

**Tone:** Warm, professional, farmer-friendly. You're a trusted agricultural advisor, not a corporate chatbot.

**Language Rules:**
- Use simple, clear language — many users have limited formal education
- Use ₹ for all prices, show per-kg AND per-quintal when relevant
- Use emojis sparingly for warmth (🌱💰📊) but not excessively
- Provide actionable advice: specific quantities, timings, costs
- When uncertain, say so honestly rather than guessing

**Formatting:**
- Use **bold** for key terms, prices, and recommendations
- Use bullet points for lists of steps or options
- Keep responses under 300 words unless the query is complex
- End with 1-2 suggested follow-up actions

**Regional Awareness:**
- Primary region: Karnataka (Kolar, Bangalore, Mysore, Tumkur, Hassan districts)
- Climate: Tropical, two main seasons (Kharif June-Oct, Rabi Nov-Mar)
- Major crops: Tomato, onion, potato, ragi, maize, coconut, arecanut, coffee
- Currency: Indian Rupees (₹), weights in kg and quintals (1 quintal = 100 kg)
- Markets: Government-regulated mandis, private procurement, CropFresh direct

**Farmer Personas (adapt tone to context):**
- Smallholder (2-5 acres): Needs simple, actionable advice; may be new to technology
- Progressive farmer (5-15 acres): Wants data, comparisons, optimization tips
- FPO leader: Needs bulk pricing, logistics planning, market strategy"""


# * ═══════════════════════════════════════════════════════════════
# * 4. PROMPT BUILDER — utility for all agents
# * ═══════════════════════════════════════════════════════════════


def build_system_prompt(
    role_description: str,
    domain_prompt: str,
    context: Optional[dict] = None,
    include_platform: bool = False,
    agent_domain: str = "general",
) -> str:
    """
    Assemble a complete system prompt with shared CropFresh context.

    Args:
        role_description: One-line role, e.g. "You are the Agronomy Expert"
        domain_prompt:    Agent-specific instructions and guidelines
        context:          Optional user context (profile, location, etc.)
        include_platform: Whether to include full platform features
        agent_domain:     The specific domain of the agent (general, agronomy, commerce, platform)

    Returns:
        Complete system prompt string
    """
    normalized_context = ensure_language_context(context) if context else None

    parts = [
        role_description,
        "",
        CROPFRESH_IDENTITY,
    ]

    if include_platform:
        parts.append(CROPFRESH_PLATFORM)

    parts.extend(
        [
            "",
            domain_prompt,
            "",
            COMMUNICATION_GUIDELINES,
        ]
    )

    # * Append user context if available
    is_kannada_user = False
    if normalized_context:
        user_section = _build_user_context(normalized_context)
        if user_section:
            parts.append(user_section)

        if resolve_language(context=normalized_context, default="en") == "kn":
            is_kannada_user = True

    # * Inject Kannada Guidelines if preferred language is Kannada
    if is_kannada_user:
        parts.append(get_kannada_context(agent_domain, normalized_context))

    return "\n\n".join(parts)


def get_identity_preamble() -> str:
    """
    Minimal CropFresh identity for routing prompts
    where token budget is tight.
    """
    return (
        "CropFresh is an AI-powered agricultural marketplace "
        "connecting Indian farmers directly with buyers, eliminating "
        "middlemen. It offers AI quality grading, real-time mandi "
        "prices, voice support in Hindi/Kannada/English, and "
        "transparent AISP pricing. Currently focused on Karnataka."
    )


def _build_user_context(context: dict) -> str:
    """Build a user context section from session data."""
    lines = ["## Current User Context"]

    profile = context.get("user_profile", {})
    if profile:
        user_type = profile.get("type", "unknown")
        lines.append(f"- User type: {user_type}")

        if profile.get("district"):
            lines.append(f"- District: {profile['district']}")
        if profile.get("crops"):
            lines.append(f"- Crops: {', '.join(profile['crops'])}")
        if profile.get("farm_size_acres"):
            lines.append(f"- Farm size: {profile['farm_size_acres']} acres")
        preferred_language = (
            profile.get("language")
            or profile.get("language_pref")
            or context.get("response_language")
            or context.get("language")
        )
        if preferred_language:
            lines.append(f"- Preferred language: {preferred_language}")

    entities = context.get("entities", {})
    if entities:
        if entities.get("commodity"):
            lines.append(f"- Active commodity: {entities['commodity']}")
        if entities.get("location"):
            lines.append(f"- Location: {entities['location']}")

    summary = context.get("conversation_summary")
    if summary:
        lines.append(f"- Conversation so far: {summary}")

    return "\n".join(lines) if len(lines) > 1 else ""
