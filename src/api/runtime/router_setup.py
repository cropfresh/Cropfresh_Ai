"""Central router registration for the FastAPI app."""

from __future__ import annotations

from fastapi import FastAPI


def include_api_routers(app: FastAPI) -> None:
    """Register all API, REST, and WebSocket routers."""
    from src.api import websocket as voice_ws
    from src.api.rest import voice as voice_rest
    from src.api.routers import auth as auth_router
    from src.api.routers import listings as listings_router
    from src.api.routers import orders as orders_router
    from src.api.routers import vision as vision_router
    from src.api.routes import adcl, chat, prices, rag
    from src.api.routes import data as data_routes

    app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(prices.router, prefix="/api/v1", tags=["prices"])
    app.include_router(data_routes.router, prefix="/api/v1", tags=["data"])
    app.include_router(adcl.router, prefix="/api/v1", tags=["adcl"])
    app.include_router(listings_router.router, prefix="/api/v1", tags=["listings"])
    app.include_router(orders_router.router, prefix="/api/v1", tags=["orders"])
    app.include_router(vision_router.router, prefix="/api/v1", tags=["vision"])
    app.include_router(auth_router.router, prefix="/api/v1", tags=["auth"])
    app.include_router(voice_rest.router, tags=["voice"])
    app.include_router(voice_ws.router, tags=["websocket"])
