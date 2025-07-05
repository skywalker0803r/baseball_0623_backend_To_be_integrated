from typing import Optional
from pydantic import BaseModel

class PitchAnalysisUpdate(BaseModel):
    pitcher_name: Optional[str] = None
    max_speed_kmh: Optional[float] = None
    pitch_score: Optional[int] = None
    ball_score: Optional[float] = None
    release_frame_url: Optional[str] = None
    landing_frame_url: Optional[str] = None
    shoulder_frame_url: Optional[str] = None