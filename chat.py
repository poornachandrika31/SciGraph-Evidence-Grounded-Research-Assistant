from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.schemas.literature import PaperResponse
from app.schemas.hypothesis import HypothesisResponse
from app.schemas.experiment import ExperimentResponse

class ChatMessageCreate(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, example="Generate three hypotheses on how microplastics affect nitrogen fixation in legumes.")

class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    user_id: str
    role: str  # user, assistant, system
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime

class ChatAgentResponse(BaseModel):
    message: ChatMessageResponse
    intent: str
    retrieved_papers: List[PaperResponse] = []
    hypotheses: List[HypothesisResponse] = []
    experiment: Optional[ExperimentResponse] = None
