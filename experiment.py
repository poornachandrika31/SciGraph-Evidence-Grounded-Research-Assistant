from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ExperimentBase(BaseModel):
    objective: str
    independent_variables: List[str] = Field(default_factory=list)
    dependent_variables: List[str] = Field(default_factory=list)
    control_variables: List[str] = Field(default_factory=list)
    control_group: Optional[str] = None
    experimental_groups: List[str] = Field(default_factory=list)
    methodology: str
    measurements: List[str] = Field(default_factory=list)
    expected_results: str
    confounders: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    statistical_analysis: str
    disclaimer: str = "AI-generated experimental design requires rigorous human scientific review and protocol validation prior to execution."

class ExperimentCreate(ExperimentBase):
    hypothesis_id: str
    session_id: str

class ExperimentResponse(ExperimentBase):
    id: str
    user_id: str
    hypothesis_id: str
    session_id: str
    created_at: datetime
