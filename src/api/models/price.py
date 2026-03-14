from pydantic import BaseModel


class PriceBase(BaseModel):
    crop_id: str
    mandi: str
    price_min: float
    price_max: float
