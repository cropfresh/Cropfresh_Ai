"""
Session Manager
================
Manage persistent browser sessions with authentication.

Features:
- Save/load session state (cookies, localStorage)
- Authenticated session management
- Session expiry and refresh
- Multi-site session handling

Author: CropFresh AI Team
Version: 1.0.0
"""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel, Field
from playwright.async_api import BrowserContext


class SessionState(BaseModel):
    """Stored session state."""
    session_id: str
    site: str
    cookies: list[dict] = Field(default_factory=list)
    local_storage: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_authenticated: bool = False
    metadata: dict = Field(default_factory=dict)


class SessionManager:
    """
    Manage persistent browser sessions.
    
    Usage:
        manager = SessionManager()
        
        # Save session after login
        await manager.save_session("enam_session", context, site="enam.gov.in")
        
        # Load session in new browser
        await manager.load_session("enam_session", context)
        
        # Get or create authenticated session
        session = await manager.get_authenticated_session(
            site="enam.gov.in",
            credentials={"username": "user", "password": "pass"},
            login_workflow=my_login_function
        )
    """
    
    SESSIONS_DIR = Path("data/browser_sessions")
    DEFAULT_EXPIRY_HOURS = 24
    
    def __init__(self, sessions_dir: Optional[Path] = None):
        """
        Initialize SessionManager.
        
        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions_dir = sessions_dir or self.SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_cache: dict[str, SessionState] = {}
        
        logger.info("SessionManager initialized (dir: {})", self.sessions_dir)
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for session."""
        return self.sessions_dir / f"{session_id}.json"
    
    async def save_session(
        self,
        session_id: str,
        context: BrowserContext,
        site: str,
        is_authenticated: bool = False,
        expiry_hours: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> SessionState:
        """
        Save browser session state to disk.
        
        Args:
            session_id: Unique identifier for the session
            context: Playwright BrowserContext
            site: Website domain
            is_authenticated: Whether session is authenticated
            expiry_hours: Hours until session expires
            metadata: Additional session metadata
            
        Returns:
            SessionState object
        """
        # Get cookies from context
        cookies = await context.cookies()
        
        # Get storage state (includes localStorage)
        storage_state = await context.storage_state()
        local_storage = {}
        for origin in storage_state.get("origins", []):
            local_storage[origin.get("origin", "")] = origin.get("localStorage", [])
        
        # Calculate expiry
        expiry = None
        if expiry_hours:
            expiry = datetime.now() + timedelta(hours=expiry_hours)
        elif self.DEFAULT_EXPIRY_HOURS:
            expiry = datetime.now() + timedelta(hours=self.DEFAULT_EXPIRY_HOURS)
        
        # Create session state
        session = SessionState(
            session_id=session_id,
            site=site,
            cookies=cookies,
            local_storage=local_storage,
            expires_at=expiry,
            is_authenticated=is_authenticated,
            metadata=metadata or {},
        )
        
        # Save to disk
        session_path = self._get_session_path(session_id)
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session.model_dump(mode='json'), f, indent=2, default=str)
        
        # Cache in memory
        self._sessions_cache[session_id] = session
        
        logger.info("Session saved: {} (site={}, auth={})", session_id, site, is_authenticated)
        return session
    
    async def load_session(
        self,
        session_id: str,
        context: BrowserContext,
    ) -> Optional[SessionState]:
        """
        Load session state into browser context.
        
        Args:
            session_id: Session identifier
            context: Playwright BrowserContext to load into
            
        Returns:
            SessionState if loaded, None if not found or expired
        """
        session = await self.get_session(session_id)
        
        if not session:
            logger.warning("Session not found: {}", session_id)
            return None
        
        # Check expiry
        if session.expires_at and datetime.now() > session.expires_at:
            logger.warning("Session expired: {}", session_id)
            await self.delete_session(session_id)
            return None
        
        # Load cookies into context
        if session.cookies:
            await context.add_cookies(session.cookies)
        
        logger.info("Session loaded: {} (site={})", session_id, session.site)
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session state without loading into browser.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionState if found, None otherwise
        """
        # Check memory cache first
        if session_id in self._sessions_cache:
            return self._sessions_cache[session_id]
        
        # Load from disk
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            return None
        
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse datetime fields
            if data.get('created_at'):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if data.get('expires_at'):
                data['expires_at'] = datetime.fromisoformat(data['expires_at'])
            
            session = SessionState(**data)
            self._sessions_cache[session_id] = session
            return session
            
        except Exception as e:
            logger.error("Failed to load session {}: {}", session_id, str(e))
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        self._sessions_cache.pop(session_id, None)
        
        # Remove from disk
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            logger.info("Session deleted: {}", session_id)
            return True
        
        return False
    
    async def list_sessions(self, site: Optional[str] = None) -> list[SessionState]:
        """
        List all saved sessions.
        
        Args:
            site: Optional filter by site
            
        Returns:
            List of SessionState objects
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem
            session = await self.get_session(session_id)
            if session:
                if site is None or session.site == site:
                    sessions.append(session)
        
        return sessions
    
    async def cleanup_expired(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions deleted
        """
        deleted = 0
        now = datetime.now()
        
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem
            session = await self.get_session(session_id)
            
            if session and session.expires_at and now > session.expires_at:
                await self.delete_session(session_id)
                deleted += 1
        
        if deleted:
            logger.info("Cleaned up {} expired sessions", deleted)
        
        return deleted
    
    async def get_authenticated_session(
        self,
        site: str,
        credentials: dict,
        login_workflow: callable,
        session_id: Optional[str] = None,
        force_reauth: bool = False,
    ) -> Optional[SessionState]:
        """
        Get or create an authenticated session.
        
        Args:
            site: Website domain
            credentials: Login credentials dict
            login_workflow: Async function to perform login
            session_id: Optional session ID (auto-generated if not provided)
            force_reauth: Force re-authentication even if session exists
            
        Returns:
            SessionState if successful, None otherwise
            
        Example:
            async def enam_login(browser, creds):
                await browser.goto("https://enam.gov.in/login")
                await browser.fill("#username", creds["username"])
                await browser.fill("#password", creds["password"])
                await browser.click("#submit")
                return await browser.page.wait_for_selector(".dashboard")
            
            session = await manager.get_authenticated_session(
                site="enam.gov.in",
                credentials={"username": "user", "password": "pass"},
                login_workflow=enam_login
            )
        """
        import hashlib
        
        # Generate session ID if not provided
        if not session_id:
            cred_hash = hashlib.md5(
                json.dumps(credentials, sort_keys=True).encode()
            ).hexdigest()[:8]
            session_id = f"{site.replace('.', '_')}_{cred_hash}"
        
        # Check for existing valid session
        if not force_reauth:
            existing = await self.get_session(session_id)
            if existing and existing.is_authenticated:
                if not existing.expires_at or datetime.now() < existing.expires_at:
                    logger.info("Using existing authenticated session: {}", session_id)
                    return existing
        
        # Need to create new authenticated session
        logger.info("Creating new authenticated session for {}", site)
        
        try:
            from src.agents.browser_agent import BrowserAgent
            
            browser = BrowserAgent(headless=True, stealth=True)
            await browser.start_session()
            
            # Execute login workflow
            success = await login_workflow(browser, credentials)
            
            if success:
                # Save authenticated session
                session = await self.save_session(
                    session_id=session_id,
                    context=browser._context,
                    site=site,
                    is_authenticated=True,
                    metadata={"username": credentials.get("username", "unknown")},
                )
                
                await browser.close_session()
                return session
            else:
                logger.error("Login workflow failed for {}", site)
                await browser.close_session()
                return None
                
        except Exception as e:
            logger.error("Authentication failed for {}: {}", site, str(e))
            return None


# Predefined login workflows for known sites
class LoginWorkflows:
    """Pre-built login workflows for agricultural portals."""
    
    @staticmethod
    async def enam_login(browser, credentials: dict) -> bool:
        """
        Login workflow for eNAM portal.
        
        Args:
            browser: BrowserAgent instance
            credentials: Dict with 'username' and 'password'
            
        Returns:
            True if login successful
        """
        from src.agents.browser_agent import BrowserAction, ActionType
        
        try:
            # Navigate to login page
            await browser.execute_action(BrowserAction(
                action=ActionType.GOTO,
                value="https://enam.gov.in/web/login"
            ))
            
            # Fill credentials
            await browser.execute_action(BrowserAction(
                action=ActionType.FILL,
                selector="#username",
                value=credentials.get("username", "")
            ))
            
            await browser.execute_action(BrowserAction(
                action=ActionType.FILL,
                selector="#password",
                value=credentials.get("password", "")
            ))
            
            # Submit
            await browser.execute_action(BrowserAction(
                action=ActionType.CLICK,
                selector="button[type='submit']"
            ))
            
            # Wait for dashboard
            result = await browser.execute_action(BrowserAction(
                action=ActionType.WAIT_FOR_SELECTOR,
                selector=".dashboard, .user-menu, #logout",
                timeout=10000
            ))
            
            return result.success
            
        except Exception as e:
            logger.error("eNAM login failed: {}", str(e))
            return False
    
    @staticmethod
    async def agmarknet_login(browser, credentials: dict) -> bool:
        """Login workflow for Agmarknet portal."""
        # Agmarknet doesn't require login for most data
        # This is a placeholder for authenticated features
        return True
