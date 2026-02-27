"""Shared pytest fixtures for CropFresh AI tests."""
import pytest

@pytest.fixture
def sample_crop():
    return {"name_en": "Tomato", "name_kn": "ಟೊಮ್ಯಾಟೊ", "category": "vegetable"}

@pytest.fixture
def sample_farmer():
    return {"phone": "+919876543210", "name": "Raju", "role": "farmer", "district": "Haveri"}
