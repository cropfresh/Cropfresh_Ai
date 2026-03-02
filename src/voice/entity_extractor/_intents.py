"""
VoiceIntent enum and ExtractionResult dataclass.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VoiceIntent(Enum):
    """Supported voice intents"""
    CREATE_LISTING = "create_listing"
    CHECK_PRICE = "check_price"
    TRACK_ORDER = "track_order"
    MY_LISTINGS = "my_listings"
    FIND_BUYER = "find_buyer"
    CHECK_WEATHER = "check_weather"
    GET_ADVISORY = "get_advisory"
    REGISTER = "register"
    DISPUTE_STATUS = "dispute_status"
    QUALITY_CHECK = "quality_check"
    WEEKLY_DEMAND = "weekly_demand"
    HELP = "help"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class ExtractionResult:
    """Result from entity extraction"""
    intent: VoiceIntent
    entities: dict[str, Any]
    confidence: float
    original_text: str
    language: str

    @property
    def is_actionable(self) -> bool:
        return self.intent != VoiceIntent.UNKNOWN and self.confidence > 0.5
