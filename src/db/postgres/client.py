"""
Aurora PostgreSQL Client — Core connection management.

Composes all domain-specific operation mixins into a unified client.
See ADR docs for architecture decisions.

Authentication:
- IAM token: boto3.rds.generate_db_auth_token() (production)
- Password: direct connection string (development)
"""

from typing import Optional

from loguru import logger

from src.db.postgres.chat import ChatOperationsMixin
from src.db.postgres.digital_twins import DigitalTwinOperationsMixin
from src.db.postgres.listings import ListingOperationsMixin
from src.db.postgres.orders import OrderOperationsMixin
from src.db.postgres.prices import PriceOperationsMixin
from src.db.postgres.users import UserOperationsMixin
from src.db.postgres.vectors import VectorOperationsMixin


class AuroraPostgresClient(
    VectorOperationsMixin,
    ChatOperationsMixin,
    UserOperationsMixin,
    ListingOperationsMixin,
    OrderOperationsMixin,
    DigitalTwinOperationsMixin,
    PriceOperationsMixin,
):
    """
    Amazon Aurora PostgreSQL client with pgvector support.

    Handles both relational operations (users, chat, produce) and
    vector search (agri_knowledge with pgvector).

    Usage:
        client = AuroraPostgresClient(host, database, ...)
        await client.connect()
        results = await client.vector_search(embedding, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        database: str = "cropfresh",
        port: int = 5432,
        user: str = "cropfresh_app",
        password: str = "",
        region: str = "ap-south-1",
        use_iam_auth: bool = False,
        pool_min: int = 2,
        pool_max: int = 10,
    ):
        self.host = host
        self.database = database
        self.port = port
        self.user = user
        self.password = password
        self.region = region
        self.use_iam_auth = use_iam_auth
        self.pool_min = pool_min
        self.pool_max = pool_max
        self._pool = None

        logger.info(f"Initializing Aurora PostgreSQL client for {host}/{database}")

    def _get_iam_token(self) -> str:
        """Generate IAM authentication token for RDS."""
        import boto3

        client = boto3.client("rds", region_name=self.region)
        token = client.generate_db_auth_token(
            DBHostname=self.host,
            Port=self.port,
            DBUsername=self.user,
            Region=self.region,
        )
        return token

    async def connect(self):
        """Establish connection pool."""
        import asyncpg

        password = self._get_iam_token() if self.use_iam_auth else self.password
        ssl_context = "require" if self.use_iam_auth else None

        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=password,
                min_size=self.pool_min,
                max_size=self.pool_max,
                ssl=ssl_context,
                command_timeout=30,
            )
            logger.info(f"Connected to Aurora PostgreSQL at {self.host}")
        except Exception as e:
            logger.error(f"Failed to connect to Aurora PostgreSQL: {e}")
            raise

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Aurora PostgreSQL connection pool closed")

    @property
    def pool(self):
        """Get connection pool (must call connect() first)."""
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not initialized. Call await client.connect() first."
            )
        return self._pool

    async def initialize_schema(self):
        """
        Create tables and pgvector extension if they don't exist.

        Safe to call multiple times (uses IF NOT EXISTS).
        """
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "schema.sql"

        if schema_path.exists():
            schema_sql = schema_path.read_text(encoding="utf-8")
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("Database schema initialized from schema.sql")
        else:
            # Inline minimal schema if file not found
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS agri_knowledge (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        text TEXT NOT NULL,
                        source TEXT DEFAULT '',
                        category TEXT DEFAULT '',
                        metadata JSONB DEFAULT '{}',
                        embedding vector(1024),
                        created_at TIMESTAMPTZ DEFAULT now()
                    );
                """)
            logger.info("Database schema initialized (inline)")

    async def run_migrations(self) -> list[str]:
        """
        Apply all pending SQL migrations using the MigrationRunner.

        Returns:
            List of migration filenames that were applied.
        """
        from src.db.migrations.migration_runner import MigrationRunner

        runner = MigrationRunner(self.pool)
        applied = await runner.run_pending()
        return applied

    async def health_check(self) -> bool:
        """Check if PostgreSQL connection is healthy."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# Singleton Factory
# ═══════════════════════════════════════════════════════════════

_client: Optional[AuroraPostgresClient] = None


async def get_postgres(
    host: str | None = None,
    database: str | None = None,
    **kwargs,
) -> AuroraPostgresClient:
    """
    Get or create shared AuroraPostgresClient instance.

    On first call, initializes connection pool and schema.
    """
    global _client

    if _client is None:
        if host is None:
            from src.config import get_settings
            settings = get_settings()
            host = settings.pg_host
            database = settings.pg_database
            kwargs.setdefault("port", settings.pg_port)
            kwargs.setdefault("user", settings.pg_user)
            kwargs.setdefault("password", settings.pg_password)
            kwargs.setdefault("region", settings.aws_region)
            kwargs.setdefault("use_iam_auth", settings.pg_use_iam_auth)

        _client = AuroraPostgresClient(
            host=host,
            database=database or "cropfresh",
            **kwargs,
        )
        await _client.connect()

    return _client
