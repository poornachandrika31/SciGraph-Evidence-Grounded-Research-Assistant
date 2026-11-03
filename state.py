from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    research_question: str
    user_message: str
    history: List[Dict[str, str]]
    intent: str
    retrieved_papers: List[Dict[str, Any]]
    candidate_hypotheses: List[Dict[str, Any]]
    critiques: List[Dict[str, Any]]
    experiment_design: Optional[Dict[str, Any]]
    final_response: str
    error: Optional[str]
    no_papers_error: bool  # Set True when hypothesis generation is blocked due to no uploaded papers
