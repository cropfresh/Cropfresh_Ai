"""
Listing Constants
=================
Constants for listing lifecycle management.
"""

# Commodity shelf-life calendar (days)
SHELF_LIFE_DAYS: dict[str, int] = {
    "tomato": 7,
    "onion": 60,
    "potato": 90,
    "beans": 5,
    "okra": 4,
    "carrot": 21,
    "cauliflower": 7,
    "cucumber": 6,
    "chilli": 14,
    "leafy greens": 3,
    "spinach": 3,
    "coriander": 5,
    "default": 14,
}

# Grade ordering for min_grade filter comparisons
GRADE_ORDER: dict[str, int] = {"A+": 4, "A": 3, "B": 2, "C": 1, "Unverified": 0}
