import os

from pydantic import BaseModel


class AddingRequest(BaseModel):
    user_id: str
    product_id: int
