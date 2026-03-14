"""
AI Kosha Agricultural Catalog
=============================
Curated catalog of agricultural datasets from AI Kosha.
"""

from .models import AIKoshaDataset, AIKoshaCategory


def get_agri_catalog() -> list[AIKoshaDataset]:
    """
    Curated catalog of agricultural datasets from AI Kosha.

    These are real datasets available on the platform. When API access
    is available, this serves as a discovery index; without API access,
    it provides metadata about what's available.
    """
    return [
        AIKoshaDataset(
            id="aikosha-agri-001",
            title="Daily Average Price of Commodities Across India",
            description=(
                "Daily commodity price data from APMC mandis across India. "
                "Includes 150+ commodities with min, max, and modal prices."
            ),
            category=AIKoshaCategory.AGRICULTURE.value,
            source_organization="Department of Consumer Affairs",
            format="CSV",
            record_count=500000,
            tags=["prices", "commodities", "mandi", "APMC", "agriculture"],
            ai_readiness_score=85.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-002",
            title="Kishan Call Center - Farmer Query Data",
            description=(
                "Call recordings and transcripts of farmer queries from "
                "Kishan Call Center. Covers crop info, pest management, "
                "weather queries, and government scheme inquiries across "
                "multiple Indian languages."
            ),
            category=AIKoshaCategory.AGRICULTURE.value,
            source_organization="Ministry of Agriculture & Farmers Welfare",
            format="JSON",
            record_count=100000,
            tags=["farmer", "queries", "kisan", "agriculture", "voice", "NLP"],
            ai_readiness_score=78.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-003",
            title="Soil Health Card Data",
            description=(
                "Soil nutrient data from Soil Health Card scheme covering "
                "pH, organic carbon, nitrogen, phosphorus, and potassium "
                "levels across Indian districts."
            ),
            category=AIKoshaCategory.AGRICULTURE.value,
            source_organization="Ministry of Agriculture",
            format="CSV",
            record_count=250000,
            tags=["soil", "nutrients", "health", "agriculture", "farming"],
            ai_readiness_score=80.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-004",
            title="Crop Production Statistics of India",
            description=(
                "State-wise, district-wise crop production data including "
                "area sown, production quantity, and yield for major crops "
                "across India from 2010-present."
            ),
            category=AIKoshaCategory.AGRICULTURE.value,
            source_organization="Directorate of Economics and Statistics",
            format="CSV",
            record_count=150000,
            tags=["crop", "production", "yield", "statistics", "agriculture"],
            ai_readiness_score=88.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-005",
            title="Indian Meteorological Department Weather Data",
            description=(
                "Historical and real-time weather data including temperature, "
                "rainfall, humidity, and wind speed from IMD stations across India."
            ),
            category=AIKoshaCategory.METEOROLOGY.value,
            source_organization="IMD - Ministry of Earth Sciences",
            format="CSV",
            record_count=1000000,
            tags=["weather", "temperature", "rainfall", "IMD", "meteorology"],
            ai_readiness_score=90.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-006",
            title="Satellite Imagery - NDVI Crop Health Index",
            description=(
                "Normalized Difference Vegetation Index (NDVI) data derived "
                "from satellite imagery for monitoring crop health, drought, "
                "and vegetation patterns across Indian agricultural regions."
            ),
            category=AIKoshaCategory.SATELLITE.value,
            source_organization="ISRO / NRSC",
            format="GeoTIFF",
            record_count=50000,
            tags=["satellite", "NDVI", "crop", "health", "remote sensing"],
            ai_readiness_score=75.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-007",
            title="PM-KISAN Beneficiary Statistics",
            description=(
                "District-wise PM-KISAN beneficiary data including enrollment "
                "counts, payment installment status, and farmer demographics."
            ),
            category=AIKoshaCategory.AGRICULTURE.value,
            source_organization="Ministry of Agriculture",
            format="JSON",
            record_count=200000,
            tags=["PM-KISAN", "government scheme", "farmers", "subsidies"],
            ai_readiness_score=82.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-008",
            title="Agricultural Census - Census 2011 Integration",
            description=(
                "Agricultural holdings data from India Census 2011 with "
                "land use patterns, irrigation sources, and crop categories "
                "at village level."
            ),
            category=AIKoshaCategory.AGRICULTURE.value,
            source_organization="Ministry of Statistics",
            format="CSV",
            record_count=600000,
            tags=["census", "land", "irrigation", "agriculture", "demographics"],
            ai_readiness_score=85.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-009",
            title="eNAM Trade Data — National Agriculture Market",
            description=(
                "Live and historical trade data from eNAM platform covering "
                "1000+ mandis with bid prices, trade volumes, and commodity "
                "arrivals data."
            ),
            category=AIKoshaCategory.AGRICULTURE.value,
            source_organization="Small Farmers Agribusiness Consortium",
            format="JSON",
            record_count=800000,
            tags=["eNAM", "trade", "mandi", "prices", "agriculture"],
            ai_readiness_score=87.0,
        ),
        AIKoshaDataset(
            id="aikosha-agri-010",
            title="Fisheries and Aquaculture Production Data",
            description=(
                "State-wise marine and inland fisheries production data "
                "including species-wise catch, aquaculture area, and "
                "export statistics."
            ),
            category=AIKoshaCategory.AQUACULTURE.value,
            source_organization="Department of Fisheries",
            format="CSV",
            record_count=50000,
            tags=["fisheries", "aquaculture", "production", "marine"],
            ai_readiness_score=72.0,
        ),
    ]
