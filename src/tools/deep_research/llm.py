"""
Deep Research LLM Helpers
=========================
Helper functions for interacting with Groq for Deep Research.
"""

import httpx
from loguru import logger

from .constants import GROQ_API_URL


async def _groq_complete(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Call the Groq Chat Completions API asynchronously.
    """
    try:
        resp = await client.post(
            GROQ_API_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning(f"Groq call failed ({model}): {exc}")
        return ""
