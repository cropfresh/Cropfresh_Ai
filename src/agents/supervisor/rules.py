"""
Rule-based logic for the Supervisor Agent.
"""

from src.agents.supervisor.models import RoutingDecision
from src.agents.supervisor.multilingual_rules import route_multilingual_intent


def route_rule_based(query: str) -> RoutingDecision:
    """
    Simple rule-based routing as fallback.

    Uses keyword matching when LLM is unavailable.
    """
    query_lower = query.lower()

    # Agronomy keywords
    agronomy_kw = [
        "grow",
        "plant",
        "cultivat",
        "harvest",
        "pest",
        "disease",
        "fertilizer",
        "soil",
        "seed",
        "irrigation",
        "organic",
        "variety",
        "crop",
        "farming",
        "agriculture",
    ]

    # Commerce keywords
    commerce_kw = [
        "price",
        "sell",
        "buy",
        "mandi",
        "market",
        "rate",
        "cost",
        "profit",
        "aisp",
        "logistics",
        "₹",
        "rupee",
        "quintal",
    ]

    # Platform keywords
    platform_kw = [
        "register",
        "login",
        "app",
        "feature",
        "account",
        "order",
        "payment",
        "cropfresh",
        "digital twin",
    ]

    # Buyer matching keywords
    matching_kw = [
        "find buyer",
        "match buyer",
        "buyer matching",
        "who wants to buy",
        "find farmer",
        "find supplier",
        "supplier match",
        "sell my produce",
        "need tomatoes",
        "need onion",
        "procurement",
    ]

    # Quality assessment keywords
    quality_kw = [
        "quality check",
        "quality assessment",
        "grade produce",
        "produce grade",
        "defect",
        "bruise",
        "worm hole",
        "fungal",
        "shelf life",
        "inspect quality",
        "a+ grade",
        "quality grading",
    ]

    # Prefer explicit matching intents before weighted keyword scoring.
    if any(keyword in query_lower for keyword in matching_kw):
        return RoutingDecision(
            agent_name="buyer_matching_agent",
            confidence=0.85,
            reasoning="Rule-based routing",
        )

    # Prefer explicit quality assessment intents before weighted scoring.
    if any(keyword in query_lower for keyword in quality_kw):
        return RoutingDecision(
            agent_name="quality_assessment_agent",
            confidence=0.84,
            reasoning="Rule-based routing",
        )

    # Web scraping keywords (live data)
    scraping_kw = [
        "live",
        "current",
        "today",
        "real-time",
        "realtime",
        "fetch",
        "scrape",
        "website",
        "portal",
        "enam",
        "agmarknet",
        "weather",
        "latest",
        "now",
        "today's",
    ]

    # Browser agent keywords (interactive)
    browser_kw = [
        "login to",
        "submit",
        "navigate",
        "download",
        "form",
        "interactive",
        "authenticated",
        "dashboard",
        "check my",
    ]

    # Research agent keywords (deep research)
    research_kw = [
        "research",
        "investigate",
        "comprehensive",
        "detailed",
        "compare",
        "analysis",
        "report",
        "study",
        "in-depth",
    ]

    # * NEW: ADCL crop recommendation keywords
    adcl_kw = [
        "recommend",
        "sow",
        "what to grow",
        "demand",
        "crop suggestion",
        "weekly report",
        "which crop",
        "what should i grow",
    ]

    # * NEW: Price prediction keywords
    prediction_kw = [
        "predict",
        "forecast",
        "trend",
        "future price",
        "will price",
        "hold or sell",
        "price tomorrow",
        "price next week",
    ]

    # * NEW: Crop listing keywords
    listing_kw = [
        "list my crop",
        "sell my produce",
        "create listing",
        "my listings",
        "cancel listing",
        "update listing",
    ]

    # * NEW: Logistics keywords
    logistics_kw = [
        "delivery",
        "transport",
        "route",
        "vehicle",
        "logistics cost",
        "shipping",
        "pickup",
        "truck",
        "tempo",
    ]

    # * NEW: Knowledge agent keywords
    knowledge_kw = [
        "explain",
        "tell me about",
        "information",
        "knowledge",
        "learn",
        "what is",
        "how does",
    ]

    # Direct/general keywords
    general_kw = [
        "hello",
        "hi",
        "thanks",
        "thank you",
        "bye",
        "help",
        "who are you",
        "what are you",
    ]

    # * Check explicit phrase matches first (before weighted scoring)
    for kw in adcl_kw:
        if kw in query_lower:
            return RoutingDecision(
                agent_name="adcl_agent",
                confidence=0.83,
                reasoning="Rule-based: crop recommendation",
            )
    for kw in listing_kw:
        if kw in query_lower:
            return RoutingDecision(
                agent_name="crop_listing_agent",
                confidence=0.83,
                reasoning="Rule-based: listing",
            )
    for kw in logistics_kw:
        if kw in query_lower:
            return RoutingDecision(
                agent_name="logistics_agent",
                confidence=0.82,
                reasoning="Rule-based: logistics",
            )

    multilingual_route = route_multilingual_intent(query)
    if multilingual_route:
        return multilingual_route

    # Score each category
    scores = {
        "agronomy_agent": sum(1 for kw in agronomy_kw if kw in query_lower),
        "commerce_agent": sum(1 for kw in commerce_kw if kw in query_lower),
        "platform_agent": sum(1 for kw in platform_kw if kw in query_lower),
        "buyer_matching_agent": sum(1 for kw in matching_kw if kw in query_lower),
        "quality_assessment_agent": sum(1 for kw in quality_kw if kw in query_lower),
        "web_scraping_agent": sum(1 for kw in scraping_kw if kw in query_lower),
        "browser_agent": sum(1 for kw in browser_kw if kw in query_lower),
        "research_agent": sum(1 for kw in research_kw if kw in query_lower),
        "price_prediction_agent": sum(1 for kw in prediction_kw if kw in query_lower),
        "knowledge_agent": sum(1 for kw in knowledge_kw if kw in query_lower),
        "general_agent": sum(1 for kw in general_kw if kw in query_lower),
    }

    # Find best match
    best_agent = max(scores, key=scores.get)
    best_score = scores[best_agent]

    # Default to general if no keywords match
    if best_score == 0:
        best_agent = "general_agent"

    confidence = min(best_score * 0.2 + 0.3, 0.9)  # Scale to 0.3-0.9

    return RoutingDecision(
        agent_name=best_agent,
        confidence=confidence,
        reasoning="Rule-based routing",
    )
