"""Base scraper class for all CropFresh data collectors."""

class BaseScraper:
    def __init__(self, name: str):
        self.name = name
    
    async def scrape(self) -> list[dict]:
        raise NotImplementedError
    
    async def save(self, data: list[dict]) -> None:
        raise NotImplementedError
