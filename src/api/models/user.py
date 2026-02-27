from pydantic import BaseModel

class UserBase(BaseModel):
    phone: str
    name: str
    role: str = 'farmer'
