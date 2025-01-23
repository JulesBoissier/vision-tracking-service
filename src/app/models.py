from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ProfileResponse(BaseModel):
    id: int
    profile_name: str
    updated_at: datetime


class ProfileListResponse(BaseModel):
    profiles: List[ProfileResponse]


class GazePredictionResponse(BaseModel):
    prediction: List[Optional[float]]
