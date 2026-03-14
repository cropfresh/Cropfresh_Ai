"""
HTML/JSON Parser for Agmarknet Mandi data.
"""
from datetime import datetime
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from loguru import logger


class AgmarknetParser:
    """Parses raw Agmarknet JSON responses into structured records."""

    @staticmethod
    def parse_json_response(json_data: dict, url: str) -> List[Dict[str, Any]]:
        """
        Parses the JSON response from the Agmarknet API.
        Expected JSON structure: { "data": { "records": [ { "data": [ ... rows ... ] } ] } }
        """
        if not json_data or not json_data.get("status"):
            return []

        data_block = json_data.get("data", {})
        if not data_block or "records" not in data_block or not data_block["records"]:
            return []

        inner_data = data_block["records"][0].get("data", [])
        if not inner_data:
            return []

        records = []
        for row in inner_data:
            try:
                # Row format example:
                # {
                #   "cmdt_name": "Onion", "max_price": "1,200.00", "min_price": "300.00",
                #   "grade_name": "Local", "state_name": "Maharashtra", "market_name": "Newasa(Ghodegaon) APMC",
                #   "model_price": "800.00", "arrival_date": "09-03-2026", "variety_name": "Red",
                #   "cmdt_grp_name": "Vegetables", "district_name": "Ahmednagar", "unit_name_price": "Rs./Quintal"
                # }

                raw_date = row.get("arrival_date", "")
                parsed_date = raw_date
                try:
                    dt = datetime.strptime(raw_date, "%d-%m-%Y").date()
                    parsed_date = dt.isoformat()
                except ValueError:
                    pass

                def parse_price(val: Any) -> float | None:
                    if not val:
                        return None
                    try:
                        return float(str(val).replace(",", ""))
                    except (ValueError, TypeError):
                        return None

                record = {
                    "state": row.get("state_name", "Unknown"),
                    "district": row.get("district_name", "Unknown"),
                    "market": row.get("market_name", "Unknown"),
                    "commodity": row.get("cmdt_name", "Unknown"),
                    "variety": row.get("variety_name", "Unknown"),
                    "grade": row.get("grade_name", "Unknown"),
                    "min_price": parse_price(row.get("min_price")),
                    "max_price": parse_price(row.get("max_price")),
                    "modal_price": parse_price(row.get("model_price")),
                    "price_date": parsed_date,
                    "unit": row.get("unit_name_price", "Rs/Quintal")
                }
                records.append(record)

            except Exception as e:
                logger.debug(f"Row parsing skipped due to error: {e}")
                continue

        return records

    @staticmethod
    def parse_html_table(html_content: str, url: str) -> List[Dict[str, Any]]:
        """
        Parses the standard Agmarknet arrivals/prices HTML table.
        Retained for fallback / future scraping of similar React lists.
        """
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")
        if not table:
            logger.warning("No data table found in HTML")
            return []

        records = []
        rows = table.find_all("tr")
        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
        else:
            rows = rows[2:] if len(rows) > 2 else []

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 12:
                continue
            try:
                data = [c.get_text(strip=True) for c in cols]
                raw_date = data[11]
                try:
                    parsed_date = datetime.strptime(raw_date, "%d-%m-%Y").date()
                except ValueError:
                    try:
                        parsed_date = datetime.strptime(raw_date, "%d/%m/%Y").date()
                    except ValueError:
                        parsed_date = raw_date

                def parse_price(val: str) -> float | None:
                    try:
                        return float(val.replace(",", ""))
                    except (ValueError, TypeError):
                        return None

                record = {
                    "state": data[0],
                    "district": data[1],
                    "market": data[2],
                    "commodity": data[4],
                    "variety": data[5],
                    "grade": data[6],
                    "min_price": parse_price(data[7]),
                    "max_price": parse_price(data[8]),
                    "modal_price": parse_price(data[9]),
                    "price_date": parsed_date.isoformat() if isinstance(parsed_date, datetime.date.__class__) else parsed_date,
                    "unit": data[10] or "Rs/Quintal"
                }
                records.append(record)

            except Exception as e:
                logger.debug(f"Row parsing skipped due to error: {e}")
                continue

        return records
