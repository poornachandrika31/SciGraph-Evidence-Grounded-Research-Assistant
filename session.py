from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class SessionCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=200, example="Microplastics and Plant Growth")
    research_question: str = Field(..., min_length=5, example="What is the impact of microplastics on soil microbiome and plant growth?")

class SessionUpdate(BaseModel):
    title: Optional[str] = None
    research_question: Optional[str] = None

class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    research_question: str
    created_at: datetime
    updated_at: datetime
    hypotheses_count: int = 0
    papers_count: int = 0
    experiments_count: int = 0

class SessionDetailResponse(SessionResponse):
    pass
