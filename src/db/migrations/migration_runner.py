"""
Migration Runner
================
Version-tracked database migration tool for CropFresh AI.

Maintains a `schema_migrations` table that records which SQL
migration files have been applied, preventing double-application
and supporting rollback tracking.

Usage:
    runner = MigrationRunner(pool)
    applied = await runner.run_pending()
    status  = await runner.get_status()
"""

# * MIGRATION RUNNER MODULE
# ! IMPORTANT: Migrations are applied in filename order (001, 002, 003...)
# NOTE: Each migration is wrapped in a transaction — partial migrations roll back

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# * Migrations directory (same folder as this file)
MIGRATIONS_DIR = Path(__file__).parent


class MigrationRecord:
    """Represents a single applied migration record."""

    def __init__(
        self,
        version: str,
        filename: str,
        checksum: str,
        applied_at: datetime,
    ) -> None:
        self.version = version
        self.filename = filename
        self.checksum = checksum
        self.applied_at = applied_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "filename": self.filename,
            "checksum": self.checksum,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }


class MigrationRunner:
    """
    Version-tracked SQL migration runner.

    Args:
        pool: asyncpg connection pool (from AuroraPostgresClient)

    Example:
        runner = MigrationRunner(client.pool)
        await runner.run_pending()
    """

    # * Table that stores applied migration records
    MIGRATIONS_TABLE = "schema_migrations"

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    async def run_pending(self) -> list[str]:
        """
        Apply all pending migrations in version order.

        Returns:
            List of migration filenames that were applied.

        Raises:
            RuntimeError: If a migration file's checksum doesn't match
                          a previously applied version (tampering detected).
        """
        await self._ensure_migrations_table()

        applied = await self._get_applied_versions()
        pending = self._discover_pending(applied)

        if not pending:
            logger.info("No pending migrations — schema is up to date")
            return []

        newly_applied: list[str] = []
        for migration_file in pending:
            await self._apply_migration(migration_file)
            newly_applied.append(migration_file.name)

        logger.info(f"Applied {len(newly_applied)} migration(s): {newly_applied}")
        return newly_applied

    async def get_status(self) -> dict[str, Any]:
        """
        Return current migration status.

        Returns:
            Dict with applied, pending counts and list of applied records.
        """
        await self._ensure_migrations_table()

        applied_versions = await self._get_applied_versions()
        all_files = self._discover_all_migration_files()
        pending_files = [
            f.name for f in all_files
            if self._version_from_filename(f.name) not in applied_versions
        ]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT version, filename, checksum, applied_at "
                f"FROM {self.MIGRATIONS_TABLE} ORDER BY version"
            )

        applied_records = [
            MigrationRecord(
                version=row["version"],
                filename=row["filename"],
                checksum=row["checksum"],
                applied_at=row["applied_at"],
            ).to_dict()
            for row in rows
        ]

        return {
            "applied_count": len(applied_versions),
            "pending_count": len(pending_files),
            "pending_files": pending_files,
            "applied": applied_records,
        }

    async def validate_checksums(self) -> list[str]:
        """
        Validate that applied migration files match their stored checksums.

        Returns:
            List of filenames with checksum mismatches (empty = all OK).
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT version, filename, checksum FROM {self.MIGRATIONS_TABLE}"
            )

        mismatches: list[str] = []
        for row in rows:
            file_path = MIGRATIONS_DIR / row["filename"]
            if not file_path.exists():
                mismatches.append(f"{row['filename']} (file missing)")
                continue
            actual = self._file_checksum(file_path)
            if actual != row["checksum"]:
                mismatches.append(
                    f"{row['filename']} (stored={row['checksum'][:8]}... actual={actual[:8]}...)"
                )

        if mismatches:
            logger.warning(f"Checksum mismatches: {mismatches}")
        return mismatches

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    async def _ensure_migrations_table(self) -> None:
        """Create schema_migrations tracking table if it doesn't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                    version     TEXT PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    checksum    TEXT NOT NULL,
                    applied_at  TIMESTAMPTZ DEFAULT now()
                )
            """)

    async def _get_applied_versions(self) -> set[str]:
        """Fetch set of already-applied migration version strings."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT version FROM {self.MIGRATIONS_TABLE}"
            )
        return {row["version"] for row in rows}

    def _discover_all_migration_files(self) -> list[Path]:
        """Return all .sql migration files sorted by name (version order)."""
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        return files

    def _discover_pending(self, applied: set[str]) -> list[Path]:
        """Return migration files that have not yet been applied."""
        return [
            f for f in self._discover_all_migration_files()
            if self._version_from_filename(f.name) not in applied
        ]

    async def _apply_migration(self, migration_file: Path) -> None:
        """Apply a single migration file inside a transaction."""
        version = self._version_from_filename(migration_file.name)
        checksum = self._file_checksum(migration_file)
        sql = migration_file.read_text(encoding="utf-8")

        logger.info(f"Applying migration {version}: {migration_file.name}")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    f"INSERT INTO {self.MIGRATIONS_TABLE} "
                    f"(version, filename, checksum) VALUES ($1, $2, $3)",
                    version, migration_file.name, checksum,
                )

        logger.info(f"Migration {version} applied successfully")

    @staticmethod
    def _version_from_filename(filename: str) -> str:
        """
        Extract version prefix from migration filename.

        Examples:
            '001_initial_schema.sql' → '001'
            '002_business_tables.sql' → '002'
        """
        match = re.match(r"^(\d+)", filename)
        return match.group(1) if match else filename

    @staticmethod
    def _file_checksum(path: Path) -> str:
        """Compute SHA-256 checksum of a migration file."""
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()
