"""HTML parsing helpers shared by table-based rate connectors."""

from __future__ import annotations

import re
from datetime import date, datetime

from bs4 import BeautifulSoup

HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "commodity": ("commodity", "crop", "commodity name", "vegetable", "product"),
    "variety": ("variety", "grade", "type"),
    "district": ("district", "city"),
    "market": ("market", "mandi", "apmc", "market name", "location"),
    "min_price": ("min", "minimum", "min price", "low"),
    "max_price": ("max", "maximum", "max price", "high"),
    "modal_price": ("modal", "model", "average", "price", "modal price"),
    "price": ("price", "support price", "gold rate", "diesel", "petrol"),
    "date": ("date", "updated", "arrival date", "price date"),
    "unit": ("unit", "uom"),
}

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
)


def clean_text(value: str | None) -> str:
    """Normalize whitespace for parser matching."""
    return re.sub(r"\s+", " ", (value or "").strip())


def slugify_header(value: str) -> str:
    """Turn a table header into a simplified slug."""
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def parse_price(value: str | None) -> float | None:
    """Parse an INR amount from free-form text."""
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"([\d,]+(?:\.\d+)?)", text.replace("?", ""))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_date(value: str | None, fallback: date | None = None) -> date:
    """Parse a date using common India-facing formats."""
    text = clean_text(value)
    if not text:
        return fallback or date.today()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    numeric = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})", text)
    if numeric:
        return parse_date(numeric.group(1), fallback=fallback)
    return fallback or date.today()


def extract_table_rows(html: str) -> list[dict[str, str]]:
    """Extract rows from all HTML tables using normalized headers."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        headers = [slugify_header(cell.get_text(" ", strip=True)) for cell in header_cells]
        if not headers:
            first_row = table.find("tr")
            if not first_row:
                continue
            headers = [slugify_header(cell.get_text(" ", strip=True)) for cell in first_row.find_all(["td", "th"])]
        if not headers:
            continue

        for row in table.find_all("tr"):
            cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
            if len(cells) != len(headers) or not cells:
                continue
            rows.append(dict(zip(headers, cells)))
    return rows


def match_field(row: dict[str, str], logical_field: str) -> str:
    """Get the best matching raw value for a logical field."""
    aliases = HEADER_ALIASES.get(logical_field, (logical_field,))
    for key, value in row.items():
        for alias in aliases:
            if alias in key:
                return value
    return ""


def extract_rate_from_text(html: str, keywords: tuple[str, ...]) -> float | None:
    """Extract a nearby price for one of the provided keywords."""
    text = clean_text(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
    for keyword in keywords:
        match = re.search(rf"{re.escape(keyword)}[^\d]{{0,30}}([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            return parse_price(match.group(1))
    return None
