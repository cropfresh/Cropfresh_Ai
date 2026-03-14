"""
Deep Research Constants
=======================
Configuration values for the deep research tool.
"""

JINA_BASE_URL = "https://r.jina.ai/"
MAX_PAGES = 15
MAX_CONTENT_CHARS = 12_000   # per page, to stay within LLM context
FETCH_TIMEOUT_SEC = 15
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Fast extraction model — cheap and quick
GROQ_FAST_MODEL = "llama-3.1-8b-instant"

# Synthesis model — higher quality for the final answer
GROQ_SYNTH_MODEL = "llama-3.3-70b-versatile"
