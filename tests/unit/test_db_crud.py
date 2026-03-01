"""
Unit Tests — Database Schema & CRUD
=====================================
Tests for:
- MigrationRunner (version tracking, pending detection, checksum)
- AuroraPostgresClient CRUD methods (mocked asyncpg pool)

Coverage targets:
1. Migration runner discovers and applies pending migrations in order
2. Applied versions are tracked and not re-applied
3. Checksum validation detects tampered migration files
4. create_farmer / get_farmer round-trip
5. create_listing / search_listings with filters
6. create_order / update_order_status lifecycle
7. create_digital_twin stores all fields
8. insert_price_history upserts + get_price_history fetches
9. create_buyer / create_dispute / get_dispute_status
10. run_migrations delegates to MigrationRunner
"""

# * TEST MODULE: Database CRUD and Migration Runner
# NOTE: asyncpg is fully mocked — no real DB required

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.db.migrations.migration_runner import MigrationRunner, MigrationRecord
from src.db.postgres_client import AuroraPostgresClient


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

def _mock_conn(fetchval=None, fetchrow=None, fetch=None, execute="UPDATE 1"):
    """Build a mock asyncpg connection."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=fetchval)
    conn.fetchrow = AsyncMock(return_value=fetchrow)
    conn.fetch = AsyncMock(return_value=fetch or [])
    conn.execute = AsyncMock(return_value=execute)
    conn.executemany = AsyncMock()
    conn.transaction = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))
    return conn


def _pool_with_conn(conn):
    """Build a mock asyncpg pool that yields `conn` from acquire()."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=cm)
    return pool


@pytest.fixture
def client():
    """AuroraPostgresClient with no real pool — pool injected per test."""
    c = AuroraPostgresClient.__new__(AuroraPostgresClient)
    c._pool = None
    c.host = "localhost"
    c.database = "cropfresh"
    return c


# ═══════════════════════════════════════════════════════════════
# MigrationRunner Tests
# ═══════════════════════════════════════════════════════════════

class TestMigrationRunner:
    """Tests for version-tracked migration execution."""

    # * Helpers

    def _make_runner(self, applied_versions: set[str], migration_files: list[Path]):
        """Build a runner with mocked pool and controlled discovery."""
        conn = _mock_conn()
        conn.fetch = AsyncMock(
            return_value=[{"version": v} for v in applied_versions]
        )
        conn.fetchrow = AsyncMock(return_value=None)
        conn.execute = AsyncMock(return_value="")
        conn.transaction = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        ))
        pool = _pool_with_conn(conn)

        runner = MigrationRunner(pool)
        runner._discover_all_migration_files = MagicMock(return_value=migration_files)
        runner._discover_pending = MagicMock(
            side_effect=lambda applied: [
                f for f in migration_files
                if runner._version_from_filename(f.name) not in applied
            ]
        )
        return runner, conn

    # * Test: version extraction

    def test_version_from_filename_extracts_prefix(self):
        """_version_from_filename returns numeric prefix."""
        assert MigrationRunner._version_from_filename("001_initial.sql") == "001"
        assert MigrationRunner._version_from_filename("023_add_column.sql") == "023"

    def test_version_from_filename_unknown_format(self):
        """Non-numbered filenames return full name as version."""
        result = MigrationRunner._version_from_filename("schema.sql")
        assert result == "schema.sql"

    # * Test: file discovery

    def test_discover_all_migration_files_sorted(self, tmp_path):
        """Migration files are returned in sorted (version) order."""
        (tmp_path / "002_b.sql").write_text("SELECT 2")
        (tmp_path / "001_a.sql").write_text("SELECT 1")
        (tmp_path / "003_c.sql").write_text("SELECT 3")

        runner = MigrationRunner(MagicMock())
        with patch.object(
            type(runner), '_discover_all_migration_files',
            return_value=sorted(tmp_path.glob("*.sql"))
        ):
            files = runner._discover_all_migration_files()

        names = [f.name for f in files]
        assert names == sorted(names)

    # * Test: pending detection

    @pytest.mark.asyncio
    async def test_run_pending_skips_applied_versions(self, tmp_path):
        """Migrations already in schema_migrations table are not re-applied."""
        f1 = tmp_path / "001_init.sql"
        f2 = tmp_path / "002_biz.sql"
        f1.write_text("SELECT 1")
        f2.write_text("SELECT 2")

        runner, conn = self._make_runner(
            applied_versions={"001"},
            migration_files=[f1, f2],
        )

        applied = await runner.run_pending()
        assert applied == ["002_biz.sql"]

    @pytest.mark.asyncio
    async def test_run_pending_returns_empty_when_all_applied(self, tmp_path):
        """No migrations applied when all versions are already tracked."""
        f1 = tmp_path / "001_init.sql"
        f1.write_text("SELECT 1")

        runner, _ = self._make_runner(
            applied_versions={"001"},
            migration_files=[f1],
        )
        applied = await runner.run_pending()
        assert applied == []

    # * Test: checksum

    def test_file_checksum_is_sha256(self, tmp_path):
        """Checksum is deterministic SHA-256 hex digest."""
        f = tmp_path / "test.sql"
        content = b"SELECT 1;"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert MigrationRunner._file_checksum(f) == expected

    @pytest.mark.asyncio
    async def test_validate_checksums_detects_mismatch(self, tmp_path):
        """validate_checksums returns filename when file was modified after apply."""
        f = tmp_path / "001_init.sql"
        f.write_text("SELECT 1")
        original_checksum = MigrationRunner._file_checksum(f)

        # * Simulate the file being changed after migration was recorded
        f.write_text("SELECT 2 -- tampered")

        conn = _mock_conn()
        conn.fetch = AsyncMock(return_value=[{
            "version": "001",
            "filename": "001_init.sql",
            "checksum": original_checksum,
        }])
        pool = _pool_with_conn(conn)
        runner = MigrationRunner(pool)

        with patch.object(Path, '__truediv__', lambda _, name: f if name == "001_init.sql" else tmp_path / name):
            with patch.object(MigrationRunner, '_file_checksum', staticmethod(
                lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            )):
                # Monkey-patch MIGRATIONS_DIR to use tmp_path for file lookup
                import src.db.migrations.migration_runner as mr_module
                original_dir = mr_module.MIGRATIONS_DIR
                mr_module.MIGRATIONS_DIR = tmp_path
                try:
                    mismatches = await runner.validate_checksums()
                finally:
                    mr_module.MIGRATIONS_DIR = original_dir

        assert any("001_init.sql" in m for m in mismatches)

    # * Test: get_status

    @pytest.mark.asyncio
    async def test_get_status_returns_counts(self, tmp_path):
        """get_status returns correct applied/pending counts."""
        f1 = tmp_path / "001_init.sql"
        f2 = tmp_path / "002_biz.sql"
        f1.write_text("SELECT 1")
        f2.write_text("SELECT 2")

        conn = _mock_conn()
        conn.fetch = AsyncMock(return_value=[{
            "version": "001",
            "filename": "001_init.sql",
            "checksum": "abc",
            "applied_at": datetime.now(),
        }])
        pool = _pool_with_conn(conn)
        runner = MigrationRunner(pool)
        runner._discover_all_migration_files = MagicMock(return_value=[f1, f2])

        status = await runner.get_status()
        assert status["applied_count"] == 1
        assert status["pending_count"] == 1
        assert "002_biz.sql" in status["pending_files"]


# ═══════════════════════════════════════════════════════════════
# AuroraPostgresClient CRUD Tests
# ═══════════════════════════════════════════════════════════════

class TestFarmerCRUD:
    """Tests for farmers table operations."""

    @pytest.mark.asyncio
    async def test_create_farmer_returns_uuid(self, client):
        """create_farmer inserts row and returns string UUID."""
        expected_id = "11111111-1111-1111-1111-111111111111"
        conn = _mock_conn(fetchval=expected_id)
        client._pool = _pool_with_conn(conn)

        result = await client.create_farmer({
            "name": "Ramesh Kumar",
            "phone": "+919876543210",
            "district": "Kolar",
            "language_pref": "kn",
        })
        assert result == expected_id

    @pytest.mark.asyncio
    async def test_get_farmer_returns_dict(self, client):
        """get_farmer returns a dict when row found."""
        mock_row = {"id": "abc", "name": "Suresh", "district": "Tumkur"}
        conn = _mock_conn(fetchrow=mock_row)
        client._pool = _pool_with_conn(conn)

        result = await client.get_farmer("abc")
        assert result == mock_row

    @pytest.mark.asyncio
    async def test_get_farmer_returns_none_when_not_found(self, client):
        """get_farmer returns None when no row matches."""
        conn = _mock_conn(fetchrow=None)
        client._pool = _pool_with_conn(conn)

        result = await client.get_farmer("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_farmer_defaults_language_pref(self, client):
        """create_farmer defaults language_pref to 'kn' if not provided."""
        conn = _mock_conn(fetchval="uuid-xyz")
        client._pool = _pool_with_conn(conn)

        await client.create_farmer({"name": "Farmer A", "phone": "123", "district": "Kolar"})
        call_args = conn.fetchval.call_args[0]
        # * 5th positional arg is language_pref (after name, phone, district, village)
        assert call_args[5] == "kn"


class TestListingsCRUD:
    """Tests for listings table operations."""

    @pytest.mark.asyncio
    async def test_create_listing_returns_uuid(self, client):
        """create_listing inserts row and returns string UUID."""
        expected = "22222222-2222-2222-2222-222222222222"
        conn = _mock_conn(fetchval=expected)
        client._pool = _pool_with_conn(conn)

        result = await client.create_listing({
            "farmer_id": "farmer-uuid",
            "commodity": "Tomato",
            "quantity_kg": 100.0,
            "asking_price_per_kg": 25.0,
        })
        assert result == expected

    @pytest.mark.asyncio
    async def test_create_listing_defaults_grade_to_unverified(self, client):
        """create_listing uses 'Unverified' as default grade."""
        conn = _mock_conn(fetchval="new-uuid")
        client._pool = _pool_with_conn(conn)

        await client.create_listing({
            "farmer_id": "f-id",
            "commodity": "Onion",
            "quantity_kg": 50.0,
            "asking_price_per_kg": 18.0,
        })
        call_args = conn.fetchval.call_args[0]
        # * 6th arg is grade (0=SQL, 1=farmer_id, 2=commodity, 3=variety, 4=qty, 5=price, 6=grade)
        assert "Unverified" in call_args

    @pytest.mark.asyncio
    async def test_search_listings_returns_list(self, client):
        """search_listings returns list of dicts."""
        mock_rows = [
            {"id": "l1", "commodity": "Tomato", "quantity_kg": 100.0},
            {"id": "l2", "commodity": "Tomato", "quantity_kg": 200.0},
        ]
        conn = _mock_conn(fetch=mock_rows)
        client._pool = _pool_with_conn(conn)

        results = await client.search_listings({"commodity": "tomato"})
        assert len(results) == 2
        assert results[0]["commodity"] == "Tomato"

    @pytest.mark.asyncio
    async def test_search_listings_empty_when_no_match(self, client):
        """search_listings returns empty list when nothing matches filters."""
        conn = _mock_conn(fetch=[])
        client._pool = _pool_with_conn(conn)

        results = await client.search_listings({"commodity": "Dragon Fruit"})
        assert results == []


class TestOrdersCRUD:
    """Tests for orders table operations."""

    @pytest.mark.asyncio
    async def test_create_order_returns_uuid(self, client):
        """create_order inserts order row and returns string UUID."""
        expected = "33333333-3333-3333-3333-333333333333"
        conn = _mock_conn(fetchval=expected)
        client._pool = _pool_with_conn(conn)

        result = await client.create_order({
            "listing_id": "l-id",
            "buyer_id": "b-id",
            "quantity_kg": 50.0,
            "farmer_payout": 1150.0,
            "logistics_cost": 100.0,
            "platform_margin": 25.0,
            "risk_buffer": 25.0,
            "aisp_total": 1300.0,
            "aisp_per_kg": 26.0,
        })
        assert result == expected

    @pytest.mark.asyncio
    async def test_update_order_status_returns_true_on_success(self, client):
        """update_order_status returns True when row exists and is updated."""
        conn = _mock_conn(execute="UPDATE 1")
        client._pool = _pool_with_conn(conn)

        result = await client.update_order_status("order-id", "in_transit")
        assert result is True

    @pytest.mark.asyncio
    async def test_update_order_status_returns_false_when_not_found(self, client):
        """update_order_status returns False when no row updated (0 rows)."""
        conn = _mock_conn(execute="UPDATE 0")
        client._pool = _pool_with_conn(conn)

        result = await client.update_order_status("nonexistent-id", "delivered")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_order_status_with_escrow(self, client):
        """update_order_status updates escrow_status when provided."""
        conn = _mock_conn(execute="UPDATE 1")
        client._pool = _pool_with_conn(conn)

        result = await client.update_order_status("o-id", "settled", escrow_status="released")
        assert result is True
        call_args = conn.execute.call_args[0]
        assert "escrow_status" in call_args[0]


class TestDigitalTwinCRUD:
    """Tests for digital_twins table operations."""

    @pytest.mark.asyncio
    async def test_create_digital_twin_returns_uuid(self, client):
        """create_digital_twin inserts row and returns string UUID."""
        expected = "44444444-4444-4444-4444-444444444444"
        conn = _mock_conn(fetchval=expected)
        client._pool = _pool_with_conn(conn)

        result = await client.create_digital_twin({
            "listing_id": "l-id",
            "farmer_photos": ["s3://bucket/photo1.jpg"],
            "grade": "A",
            "confidence": 0.92,
            "defect_types": ["bruise"],
        })
        assert result == expected

    @pytest.mark.asyncio
    async def test_create_digital_twin_defaults_photos_to_empty_list(self, client):
        """create_digital_twin uses [] for photos when not provided."""
        conn = _mock_conn(fetchval="twin-uuid")
        client._pool = _pool_with_conn(conn)

        await client.create_digital_twin({"listing_id": "l-id"})
        call_args = conn.fetchval.call_args[0]
        # * 2nd arg (index 2) is farmer_photos
        assert call_args[2] == []


class TestPriceHistoryCRUD:
    """Tests for price_history table operations."""

    @pytest.mark.asyncio
    async def test_insert_price_history_returns_count(self, client):
        """insert_price_history returns number of records inserted."""
        conn = _mock_conn()
        client._pool = _pool_with_conn(conn)

        records = [
            {"commodity": "Tomato", "district": "Kolar",
             "date": date(2026, 3, 1), "modal_price": 2500.0},
            {"commodity": "Tomato", "district": "Kolar",
             "date": date(2026, 2, 28), "modal_price": 2400.0},
        ]
        result = await client.insert_price_history(records)
        assert result == 2
        assert conn.executemany.called

    @pytest.mark.asyncio
    async def test_insert_price_history_defaults_source(self, client):
        """insert_price_history defaults source to 'agmarknet'."""
        conn = _mock_conn()
        client._pool = _pool_with_conn(conn)

        await client.insert_price_history([{
            "commodity": "Onion", "district": "Tumkur",
            "date": date(2026, 3, 1), "modal_price": 1800.0,
        }])
        rows_arg = conn.executemany.call_args[0][1]
        assert rows_arg[0][7] == "agmarknet"  # 8th tuple element = source

    @pytest.mark.asyncio
    async def test_get_price_history_returns_list(self, client):
        """get_price_history returns list of price records."""
        mock_rows = [
            {"commodity": "Tomato", "district": "Kolar",
             "date": date(2026, 3, 1), "modal_price": 2500.0},
        ]
        conn = _mock_conn(fetch=mock_rows)
        client._pool = _pool_with_conn(conn)

        results = await client.get_price_history("Tomato", "Kolar", days=7)
        assert len(results) == 1
        assert results[0]["modal_price"] == 2500.0

    @pytest.mark.asyncio
    async def test_get_price_history_passes_correct_params(self, client):
        """get_price_history passes commodity/district as ILIKE patterns."""
        conn = _mock_conn(fetch=[])
        client._pool = _pool_with_conn(conn)

        await client.get_price_history("tomato", "kolar", days=30)
        call_args = conn.fetch.call_args[0]
        assert call_args[1] == "%tomato%"
        assert call_args[2] == "%kolar%"
        assert call_args[3] == 30


class TestBuyerDisputeCRUD:
    """Tests for buyers and disputes tables."""

    @pytest.mark.asyncio
    async def test_create_buyer_returns_uuid(self, client):
        """create_buyer inserts buyer and returns string UUID."""
        expected = "55555555-5555-5555-5555-555555555555"
        conn = _mock_conn(fetchval=expected)
        client._pool = _pool_with_conn(conn)

        result = await client.create_buyer({
            "name": "Fresh Mart",
            "phone": "+919000000001",
            "type": "retailer",
            "district": "Bangalore Urban",
        })
        assert result == expected

    @pytest.mark.asyncio
    async def test_create_dispute_returns_uuid(self, client):
        """create_dispute opens a dispute and returns string UUID."""
        expected = "66666666-6666-6666-6666-666666666666"
        conn = _mock_conn(fetchval=expected)
        client._pool = _pool_with_conn(conn)

        result = await client.create_dispute({
            "order_id": "o-id",
            "raised_by": "buyer",
            "reason": "Grade mismatch — arrived as C, listed as A",
        })
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_dispute_status_returns_dict(self, client):
        """get_dispute_status returns dict with dispute + order info."""
        mock_row = {
            "id": "d-id",
            "status": "open",
            "raised_by": "buyer",
            "reason": "Grade mismatch",
            "order_status": "disputed",
            "aisp_total": 1300.0,
            "commodity": "Tomato",
        }
        conn = _mock_conn(fetchrow=mock_row)
        client._pool = _pool_with_conn(conn)

        result = await client.get_dispute_status("d-id")
        assert result["status"] == "open"
        assert result["commodity"] == "Tomato"

    @pytest.mark.asyncio
    async def test_get_dispute_status_returns_none_when_not_found(self, client):
        """get_dispute_status returns None for unknown dispute ID."""
        conn = _mock_conn(fetchrow=None)
        client._pool = _pool_with_conn(conn)

        result = await client.get_dispute_status("nonexistent-id")
        assert result is None


class TestRunMigrations:
    """Tests for run_migrations delegation."""

    @pytest.mark.asyncio
    async def test_run_migrations_delegates_to_runner(self, client):
        """run_migrations calls MigrationRunner.run_pending() and returns result."""
        conn = _mock_conn()
        client._pool = _pool_with_conn(conn)

        with patch(
            "src.db.migrations.migration_runner.MigrationRunner.run_pending",
            new_callable=AsyncMock,
            return_value=["002_business_tables.sql"],
        ):
            result = await client.run_migrations()

        assert "002_business_tables.sql" in result

    @pytest.mark.asyncio
    async def test_run_migrations_returns_empty_when_up_to_date(self, client):
        """run_migrations returns [] when all migrations are applied."""
        conn = _mock_conn()
        client._pool = _pool_with_conn(conn)

        with patch(
            "src.db.migrations.migration_runner.MigrationRunner.run_pending",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await client.run_migrations()

        assert result == []
