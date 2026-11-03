from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timezone
from typing import List
from bson import ObjectId
from app.schemas.experiment import ExperimentResponse
from app.services.auth import get_current_user
from app.services.llm import LLMService
from app.agent.prompts import EXPERIMENT_DESIGN_SYSTEM_PROMPT
from app.config.database import get_db

router = APIRouter(prefix="/api", tags=["Experiments"])

@router.post("/hypotheses/{hypothesis_id}/experiment", response_model=ExperimentResponse)
async def generate_hypothesis_experiment(hypothesis_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    user_id = current_user["id"]
    try:
        oid = ObjectId(hypothesis_id)
    except Exception:
        oid = hypothesis_id

    h = await db.hypotheses.find_one({"_id": oid})
    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    session_id = str(h["session_id"])
    
    prompt = (
        f"Hypothesis: {h.get('hypothesis')}\n"
        f"Rationale: {h.get('rationale')}\n"
        f"Variables: {h.get('variables')}\n\n"
        "Design a rigorous laboratory experimental protocol to test this specific scientific hypothesis."
    )

    try:
        data = await LLMService.generate_structured_json(EXPERIMENT_DESIGN_SYSTEM_PROMPT, prompt)
    except Exception as e:
        import json
        data = json.loads(LLMService._domain_fallback_completion("", "experiment"))

    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "hypothesis_id": str(h["_id"]),
        "objective": data.get("objective", "Empirically test hypothesis"),
        "independent_variables": data.get("independent_variables", []),
        "dependent_variables": data.get("dependent_variables", []),
        "control_variables": data.get("control_variables", []),
        "control_group": data.get("control_group", "Baseline control"),
        "experimental_groups": data.get("experimental_groups", []),
        "methodology": data.get("methodology", ""),
        "measurements": data.get("measurements", []),
        "expected_results": data.get("expected_results", ""),
        "confounders": data.get("confounders", []),
        "limitations": data.get("limitations", []),
        "statistical_analysis": data.get("statistical_analysis", "ANOVA with post-hoc Tukey HSD"),
        "disclaimer": "AI-generated experimental design requires rigorous human scientific review and protocol validation prior to execution.",
        "created_at": now
    }

    res = await db.experiments.insert_one(doc)
    doc["id"] = str(res.inserted_id)

    return ExperimentResponse(**doc)

@router.get("/hypotheses/{hypothesis_id}/experiments", response_model=List[ExperimentResponse])
async def get_experiments_for_hypothesis(hypothesis_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db.experiments.find({"hypothesis_id": hypothesis_id}).sort("created_at", -1)
    
    results = []
    async for e in cursor:
        results.append(ExperimentResponse(
            id=str(e["_id"]),
            user_id=str(e["user_id"]),
            hypothesis_id=str(e["hypothesis_id"]),
            session_id=str(e["session_id"]),
            objective=e.get("objective", ""),
            independent_variables=e.get("independent_variables", []),
            dependent_variables=e.get("dependent_variables", []),
            control_variables=e.get("control_variables", []),
            control_group=e.get("control_group"),
            experimental_groups=e.get("experimental_groups", []),
            methodology=e.get("methodology", ""),
            measurements=e.get("measurements", []),
            expected_results=e.get("expected_results", ""),
            confounders=e.get("confounders", []),
            limitations=e.get("limitations", []),
            statistical_analysis=e.get("statistical_analysis", ""),
            disclaimer=e.get("disclaimer", ""),
            created_at=e.get("created_at")
        ))
    return results

@router.get("/experiments", response_model=List[ExperimentResponse])
async def get_all_user_experiments(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db.experiments.find({"user_id": current_user["id"]}).sort("created_at", -1)
    
    results = []
    async for e in cursor:
        results.append(ExperimentResponse(
            id=str(e["_id"]),
            user_id=str(e["user_id"]),
            hypothesis_id=str(e["hypothesis_id"]),
            session_id=str(e["session_id"]),
            objective=e.get("objective", ""),
            independent_variables=e.get("independent_variables", []),
            dependent_variables=e.get("dependent_variables", []),
            control_variables=e.get("control_variables", []),
            control_group=e.get("control_group"),
            experimental_groups=e.get("experimental_groups", []),
            methodology=e.get("methodology", ""),
            measurements=e.get("measurements", []),
            expected_results=e.get("expected_results", ""),
            confounders=e.get("confounders", []),
            limitations=e.get("limitations", []),
            statistical_analysis=e.get("statistical_analysis", ""),
            disclaimer=e.get("disclaimer", ""),
            created_at=e.get("created_at")
        ))
    return results
