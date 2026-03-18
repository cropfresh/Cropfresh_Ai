"""Local entrypoint that runs both the HTTP and gRPC surfaces."""

from __future__ import annotations

import asyncio

import uvicorn

from .api import create_app
from .config import VadServiceSettings
from .grpc_server import create_grpc_server
from .runtime import VadServiceRuntime


async def main() -> None:
    """Run the FastAPI app and the gRPC server in one local process."""
    settings = VadServiceSettings()
    runtime = VadServiceRuntime(settings)
    await runtime.bootstrap()

    grpc_server = None
    if settings.enable_grpc:
        grpc_server = await create_grpc_server(settings, runtime)
        await grpc_server.start()

    app = create_app(runtime)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )
    )

    try:
        await server.serve()
    finally:
        if grpc_server is not None:
            await grpc_server.stop(grace=5)


if __name__ == "__main__":
    asyncio.run(main())
