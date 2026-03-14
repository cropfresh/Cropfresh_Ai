from pydantic import BaseModel


class OrderBase(BaseModel):
    listing_id: str
    quantity_kg: float
