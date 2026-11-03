from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VariablesSchema(BaseModel):
    independent: List[str] = Field(default_factory=list, description="Variables manipulated by the experimenter")
    dependent: List[str] = Field(default_factory=list, description="Variables measured/observed")
    control: List[str] = Field(default_factory=list, description="Variables held constant")
    confounders: List[str] = Field(default_factory=list, description="Potential confounding factors")

class HypothesisBase(BaseModel):
    hypothesis: str
    rationale: str
    novelty_score: float = Field(..., ge=1.0, le=10.0)
    testability_score: float = Field(..., ge=1.0, le=10.0)
    plausibility_score: float = Field(..., ge=1.0, le=10.0)
    evidence_score: float = Field(..., ge=1.0, le=10.0)
    falsifiability_score: float = Field(..., ge=1.0, le=10.0)
    overall_score: Optional[float] = None
    variables: VariablesSchema
    supporting_evidence: Optional[str] = None
    critique: Optional[str] = None
    status: str = "generated"  # generated, saved, refined, archived

class HypothesisCreate(HypothesisBase):
    session_id: str

class HypothesisRefineRequest(BaseModel):
    refinement_prompt: str = Field(..., min_length=3, example="Make hypothesis 2 focus specifically on root elongation")

class HypothesisResponse(HypothesisBase):
    id: str
    user_id: str
    session_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
