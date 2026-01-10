"""
Supabase Client (REST API)
==========================
PostgreSQL database client using Supabase REST API.

Uses httpx instead of full supabase-py SDK to avoid build issues.
"""

from functools import lru_cache
from typing import Any, Optional

import httpx
from loguru import logger
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # 'user' or 'assistant'
    content: str
    agent_name: Optional[str] = None


class SupabaseClient:
    """
    Supabase/PostgreSQL client using REST API.
    
    Uses direct REST API calls instead of supabase-py SDK.
    This avoids the pyroaring build issues on Windows.
    
    Usage:
        client = SupabaseClient(url, key)
        await client.save_chat_message(user_id, session_id, message)
    """
    
    def __init__(self, url: str, key: str):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL
            key: Supabase API key (anon/service role)
        """
        self.url = url.rstrip("/")
        self.key = key
        self._client: Optional[httpx.Client] = None
        
        logger.info(f"Initializing Supabase REST client for {url}")
    
    @property
    def client(self) -> httpx.Client:
        """Get HTTP client with auth headers."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=f"{self.url}/rest/v1",
                headers={
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                },
                timeout=30.0,
            )
        return self._client
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """Make a REST API request."""
        response = self.client.request(
            method=method,
            url=endpoint,
            json=data,
            params=params,
        )
        response.raise_for_status()
        return response.json() if response.text else []
    
    # ═══════════════════════════════════════════════════════════════
    # Chat History
    # ═══════════════════════════════════════════════════════════════
    
    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> dict[str, Any]:
        """Save a chat message to history."""
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "role": message.role,
            "content": message.content,
            "agent_name": message.agent_name,
        }
        
        result = self._request("POST", "/chat_history", data=data)
        logger.debug(f"Saved chat message for session {session_id}")
        return result[0] if result else {}
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ChatMessage]:
        """Get chat history for a session."""
        params = {
            "session_id": f"eq.{session_id}",
            "order": "created_at.asc",
            "limit": str(limit),
        }
        
        result = self._request("GET", "/chat_history", params=params)
        
        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                agent_name=row.get("agent_name"),
            )
            for row in result
        ]
    
    # ═══════════════════════════════════════════════════════════════
    # Users
    # ═══════════════════════════════════════════════════════════════
    
    async def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get user by ID."""
        params = {"id": f"eq.{user_id}"}
        result = self._request("GET", "/users", params=params)
        return result[0] if result else None
    
    async def create_user(
        self,
        phone: str,
        name: str,
        user_type: str = "farmer",
        location: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a new user."""
        data = {
            "phone": phone,
            "name": name,
            "user_type": user_type,
            "location": location or {},
        }
        
        result = self._request("POST", "/users", data=data)
        logger.info(f"Created user: {name} ({user_type})")
        return result[0] if result else {}
    
    # ═══════════════════════════════════════════════════════════════
    # Produce Listings
    # ═══════════════════════════════════════════════════════════════
    
    async def list_produce(
        self,
        crop_name: Optional[str] = None,
        quality_grade: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List available produce with optional filters."""
        params = {
            "status": "eq.available",
            "limit": str(limit),
        }
        
        if crop_name:
            params["crop_name"] = f"ilike.*{crop_name}*"
        if quality_grade:
            params["quality_grade"] = f"eq.{quality_grade}"
        
        return self._request("GET", "/produce", params=params)
    
    async def create_produce_listing(
        self,
        farmer_id: str,
        crop_name: str,
        quantity_kg: float,
        price_per_kg: float,
        quality_grade: str = "B",
        location: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a new produce listing."""
        data = {
            "farmer_id": farmer_id,
            "crop_name": crop_name,
            "quantity_kg": quantity_kg,
            "price_per_kg": price_per_kg,
            "quality_grade": quality_grade,
            "location": location or {},
            "status": "available",
        }
        
        result = self._request("POST", "/produce", data=data)
        logger.info(f"Created listing: {quantity_kg}kg {crop_name}")
        return result[0] if result else {}
    
    # ═══════════════════════════════════════════════════════════════
    # Health Check
    # ═══════════════════════════════════════════════════════════════
    
    def health_check(self) -> bool:
        """Check if Supabase connection is healthy."""
        try:
            # Simple request to test connection
            response = self.client.get("/")
            return response.status_code in (200, 404)
        except Exception as e:
            logger.error(f"Supabase health check failed: {e}")
            return False
    
    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


@lru_cache(maxsize=1)
def get_supabase(url: str = None, key: str = None) -> SupabaseClient:
    """Get cached Supabase client instance."""
    if url is None or key is None:
        from src.config import get_settings
        settings = get_settings()
        url = settings.supabase_url
        key = settings.supabase_key
    
    return SupabaseClient(url=url, key=key)
