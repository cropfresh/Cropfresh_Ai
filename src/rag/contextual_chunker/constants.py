"""
Contextual Chunker Constants
============================
LLM prompts and regex patterns for entity extraction.
"""

# Context generation prompt
CONTEXT_PROMPT = """You are generating context for a document chunk to help a search system understand it better.

Given the document information and a specific chunk from it, write a brief context statement (2-3 sentences) that explains:
1. What document this is from and its topic
2. What specific subject this chunk covers
3. Any key entities or concepts mentioned

Document Title: {title}
Document Source: {source}
Document Summary: {summary}

Chunk:
{chunk}

Brief Context (2-3 sentences):"""

# Entity extraction patterns
ENTITY_PATTERNS = {
    "CROP": [
        r'\b(tomato|potato|onion|rice|wheat|maize|cotton|sugarcane|mango|banana|chilli|turmeric)\b',
        r'\b(टमाटर|आलू|प्याज|चावल|गेहूं|मक्का|कपास)\b',  # Hindi
    ],
    "PEST": [
        r'\b(aphid|whitefly|thrips|mite|borer|caterpillar|beetle|worm|nematode)\b',
        r'\b(pest|insect|bug|larvae)\b',
    ],
    "DISEASE": [
        r'\b(blight|wilt|rust|rot|mosaic|mildew|canker|scab|leaf spot)\b',
        r'\b(fungal|bacterial|viral|disease|infection)\b',
    ],
    "CHEMICAL": [
        r'\b(neem|pesticide|fungicide|insecticide|herbicide|fertilizer|urea|DAP|potash)\b',
    ],
    "LOCATION": [
        r'\b(Karnataka|Maharashtra|Andhra Pradesh|Tamil Nadu|Kerala|Gujarat|Punjab|Haryana)\b',
        r'\b(Kolar|Nashik|Pune|Bangalore|Hyderabad|Chennai)\b',
    ],
    "PRICE": [
        r'₹\s*[\d,]+',
        r'\b\d+\s*(?:per|/)\s*(?:kg|quintal|tonne)\b',
    ],
}
