import pytest
from src.scrapers.agmarknet.parser import AgmarknetParser

def test_agmarknet_json_parser():
    mock_json = {
        "status": True,
        "data": {
            "records": [
                {
                    "data": [
                        {
                            "cmdt_name": "Tomato",
                            "max_price": "1,500.00",
                            "min_price": "1,000.00",
                            "grade_name": "Local",
                            "state_name": "Karnataka",
                            "market_name": "Bangarpet APMC",
                            "model_price": "1,200.00",
                            "arrival_date": "09-03-2026",
                            "variety_name": "Tomato (Variety)",
                            "cmdt_grp_name": "Vegetables",
                            "district_name": "Kolar",
                            "unit_name_price": "Rs./Quintal"
                        }
                    ]
                }
            ]
        }
    }
    
    records = AgmarknetParser.parse_json_response(mock_json, "http://mock.url")
    assert len(records) == 1
    
    rec = records[0]
    assert rec["state"] == "Karnataka"
    assert rec["district"] == "Kolar"
    assert rec["market"] == "Bangarpet APMC"
    assert rec["commodity"] == "Tomato"
    assert rec["min_price"] == 1000.0
    assert rec["max_price"] == 1500.0
    assert rec["modal_price"] == 1200.0
    assert str(rec["price_date"]) == "2026-03-09"
