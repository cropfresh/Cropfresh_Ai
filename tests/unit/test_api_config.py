"""Tests for tolerant API configuration parsing."""

from __future__ import annotations

import pytest

from src.api.config import Settings


def test_settings_treat_release_debug_value_as_false() -> None:
    settings = Settings(debug="release", environment="production")

    assert settings.debug is False
    assert settings.environment == "production"


def test_settings_treat_development_debug_value_as_true() -> None:
    settings = Settings(debug="development", environment="development")

    assert settings.debug is True


def test_settings_reject_unknown_debug_value() -> None:
    with pytest.raises(ValueError):
        Settings(debug="maybe", environment="development")
