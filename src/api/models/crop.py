from pydantic import BaseModel


class CropBase(BaseModel):
    name_en: str
    name_kn: str | None = None
    category: str
