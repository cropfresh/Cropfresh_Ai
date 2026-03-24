"""
Agricultural Source Registry for Browser RAG.
"""

from src.rag.browser_rag_pkg.models import ScrapeIntent, TargetSource


class AgriSourceSelector:
    """
    Maps query intents to priority-ranked agricultural web sources.

    Sources are chosen for:
    - Reliability (government-primary, then established agri news)
    - Scrape-friendliness (public HTML preferred over JS-heavy SPA)
    - Indian agricultural relevance
    """

    # Source registry: intent → list of TargetSource (priority order)
    SOURCES: dict[ScrapeIntent, list[TargetSource]] = {
        ScrapeIntent.SCHEME_UPDATE: [
            TargetSource(
                url="https://dbtbharat.gov.in/page/frontcontentview/?id=MjM=",
                intent=ScrapeIntent.SCHEME_UPDATE,
                source_name="DBT Bharat Agriculture",
                ttl_hours=24.0,
                fetcher_type="basic",
            ),
            TargetSource(
                url="https://farmer.gov.in/HelpDocument.aspx",
                intent=ScrapeIntent.SCHEME_UPDATE,
                source_name="Farmer.gov.in",
                ttl_hours=24.0,
                fetcher_type="basic",
                css_selector=".content-area",
            ),
            TargetSource(
                url="https://www.india.gov.in/topics/agriculture",
                intent=ScrapeIntent.SCHEME_UPDATE,
                source_name="India.gov.in Agriculture",
                ttl_hours=48.0,
                fetcher_type="basic",
            ),
        ],

        ScrapeIntent.DISEASE_ALERT: [
            TargetSource(
                url="https://www.agrifarming.in/crop-diseases",
                intent=ScrapeIntent.DISEASE_ALERT,
                source_name="AgrifarMing Crop Diseases",
                ttl_hours=12.0,
                fetcher_type="basic",
                css_selector="article, .post-content",
            ),
            TargetSource(
                url="https://icar.org.in/news",
                intent=ScrapeIntent.DISEASE_ALERT,
                source_name="ICAR News",
                ttl_hours=12.0,
                fetcher_type="basic",
                css_selector=".views-row",
            ),
            TargetSource(
                url="https://nhb.gov.in/horticultural-crops",
                intent=ScrapeIntent.DISEASE_ALERT,
                source_name="National Horticulture Board",
                ttl_hours=24.0,
                fetcher_type="basic",
            ),
        ],

        ScrapeIntent.PESTICIDE_BAN: [
            TargetSource(
                url="https://cibrc.nic.in/prt.php",
                intent=ScrapeIntent.PESTICIDE_BAN,
                source_name="CIB&RC Pesticide Registrations",
                ttl_hours=72.0,
                fetcher_type="basic",
                css_selector="table",
            ),
            TargetSource(
                url="https://ppqs.gov.in/divisions/insecticides-act-division",
                intent=ScrapeIntent.PESTICIDE_BAN,
                source_name="PPQS Insecticides Act",
                ttl_hours=72.0,
                fetcher_type="basic",
            ),
        ],

        ScrapeIntent.PRICE_NEWS: [
            TargetSource(
                url="https://www.commodityindia.com/agriculture-news",
                intent=ScrapeIntent.PRICE_NEWS,
                source_name="CommodityIndia Market News",
                ttl_hours=2.0,
                fetcher_type="basic",
                css_selector=".news-list, .article-body",
            ),
            TargetSource(
                url="https://agriwatch.com/news-2/",
                intent=ScrapeIntent.PRICE_NEWS,
                source_name="Agriwatch News",
                ttl_hours=3.0,
                fetcher_type="basic",
                css_selector=".entry-content",
            ),
        ],

        ScrapeIntent.EXPORT_POLICY: [
            TargetSource(
                url="https://apeda.gov.in/apedawebsite/news_letter/news_letter.htm",
                intent=ScrapeIntent.EXPORT_POLICY,
                source_name="APEDA News",
                ttl_hours=24.0,
                fetcher_type="basic",
                css_selector=".news",
            ),
            TargetSource(
                url="https://agriexchange.apeda.gov.in/",
                intent=ScrapeIntent.EXPORT_POLICY,
                source_name="APEDA AgriXchange",
                ttl_hours=6.0,
                fetcher_type="stealth",
            ),
        ],

        ScrapeIntent.WEATHER_ADVISORY: [
            TargetSource(
                url="https://mausam.imd.gov.in/responsive/agricultureweather.php",
                intent=ScrapeIntent.WEATHER_ADVISORY,
                source_name="IMD Agro-meteorological Advisory",
                ttl_hours=6.0,
                fetcher_type="basic",
                css_selector=".agromet",
            ),
        ],

        ScrapeIntent.MARKET_NEWS: [
            TargetSource(
                url="https://krishijagran.com/news/",
                intent=ScrapeIntent.MARKET_NEWS,
                source_name="Krishi Jagran",
                ttl_hours=4.0,
                fetcher_type="basic",
                css_selector="article",
            ),
            TargetSource(
                url="https://www.thehindubusinessline.com/markets/commodities/",
                intent=ScrapeIntent.MARKET_NEWS,
                source_name="Hindu BusinessLine Commodities",
                ttl_hours=4.0,
                fetcher_type="stealth",
                css_selector="article, .article-content",
            ),
        ],
    }

    # Intent classification rules (keyword → intent)
    INTENT_RULES: list[tuple[list[str], ScrapeIntent]] = [
        (["pest", "disease", "blight", "outbreak", "infection", "symptom"], ScrapeIntent.DISEASE_ALERT),
        (["pesticide", "chemical", "banned", "restricted", "herbicide"], ScrapeIntent.PESTICIDE_BAN),
        (["export", "import", "ban", "restriction", "mep", "minimum export"], ScrapeIntent.EXPORT_POLICY),
        (["weather", "monsoon", "rainfall", "forecast", "advisory"], ScrapeIntent.WEATHER_ADVISORY),
        (["price", "mandi", "market rate", "commodity", "bhaav"], ScrapeIntent.PRICE_NEWS),
        (["scheme", "subsidy", "government scheme", "yojana", "policy", "benefit"], ScrapeIntent.SCHEME_UPDATE),
    ]

    def classify_intent(self, query: str) -> ScrapeIntent:
        """
        Classify a query into a scraping intent.

        Args:
            query: User query text

        Returns:
            ScrapeIntent for the matching category (default: MARKET_NEWS)
        """
        q = query.lower()
        for keywords, intent in self.INTENT_RULES:
            if any(kw in q for kw in keywords):
                return intent
        return ScrapeIntent.MARKET_NEWS

    def get_sources(
        self,
        intent: ScrapeIntent,
        max_sources: int = 3,
    ) -> list[TargetSource]:
        """
        Get priority-ranked sources for a scraping intent.

        Args:
            intent: The scraping intent
            max_sources: Maximum number of sources to return

        Returns:
            List of TargetSource, most reliable first
        """
        sources = self.SOURCES.get(intent, self.SOURCES[ScrapeIntent.MARKET_NEWS])
        return sources[:max_sources]
