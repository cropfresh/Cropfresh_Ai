"""
Normalizes raw scraped records into canonical Price records.
"""
from datetime import date, datetime
from typing import Any, Optional

from loguru import logger

from src.db.models.price_records import NormalizedPriceRecord, RawPriceRecord


class PriceNormalizer:
    """
    Cleans, standardizes, and normalizes raw scraped data into canonical formats.
    """

    # Simple mapping dictionaries could be expanded into DB tables or NLP models
    COMMODITY_MAP = {
        "bhindi(ladies finger)": "Bhindi",
        "ladies finger": "Bhindi",
        "tomato": "Tomato",
        "onion": "Onion",
        "potato": "Potato",
        # ... add more mappings as needed
    }

    UNIT_MAP = {
        "rs/quintal": "INR/Quintal",
        "rs./quintal": "INR/Quintal",
        "rs/ton": "INR/Ton",
        "rs/kg": "INR/Kg"
    }

    @staticmethod
    def _normalize_commodity(raw_name: str) -> str:
        """Map raw commodity to standard ID/name."""
        if not raw_name: return "Unknown"
        clean_name = raw_name.strip().lower()
        return PriceNormalizer.COMMODITY_MAP.get(clean_name, raw_name.strip().title())

    @staticmethod
    def _normalize_unit(raw_unit: str) -> str:
        """Standardize weight and currency units."""
        if not raw_unit: return "Unknown"
        clean_unit = raw_unit.strip().lower()
        return PriceNormalizer.UNIT_MAP.get(clean_unit, raw_unit.strip())

    @staticmethod
    def _parse_date(date_intake: Any) -> date:
        """Ensure date is a valid python date object."""
        if isinstance(date_intake, date) and not isinstance(date_intake, datetime):
            return date_intake
        if isinstance(date_intake, datetime):
            return date_intake.date()
        if isinstance(date_intake, str):
            try:
                return datetime.fromisoformat(date_intake).date()
            except ValueError:
                # Fallback to today if completely unparseable
                logger.warning(f"Failed to parse date string '{date_intake}', using today.")
                return datetime.today().date()
        return datetime.today().date()

    @staticmethod
    def normalize(raw_record: RawPriceRecord) -> Optional[NormalizedPriceRecord]:
        """
        Convert a RawPriceRecord into a NormalizedPriceRecord.
        Returns None if record lacks required fields.
        """
        data = raw_record.raw_data

        # Required fields check
        market = data.get("market")
        raw_commodity = data.get("commodity")

        if not market or not raw_commodity:
            logger.warning("Record missing required 'market' or 'commodity' fields. Skipping normalization.")
            return None

        commodity = PriceNormalizer._normalize_commodity(raw_commodity)
        unit = PriceNormalizer._normalize_unit(data.get("unit", ""))
        price_date = PriceNormalizer._parse_date(data.get("price_date"))

        # Build normalized record
        return NormalizedPriceRecord(
            commodity=commodity,
            market=market.strip().title(),
            price_date=price_date,
            source=raw_record.source,
            variety=data.get("variety", "").strip().title() or None,
            state=data.get("state", "").strip().title() or None,
            min_price=data.get("min_price"),
            max_price=data.get("max_price"),
            modal_price=data.get("modal_price"),
            unit=unit,
            raw_record_id=raw_record.id
        )
