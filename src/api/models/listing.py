from pydantic import BaseModel

class ListingBase(BaseModel):
    crop_id: str
    quantity_kg: float
    price_per_kg: float
    grade: str | None = None
