from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timezone
from typing import List, Optional
from bson import ObjectId
from app.schemas.hypothesis import HypothesisResponse, HypothesisRefineRequest
from app.services.auth import get_current_user
from app.services.llm import LLMService
from app.config.database import get_db

router = APIRouter(prefix="/api", tags=["Hypotheses"])

@router.get("/sessions/{session_id}/hypotheses", response_model=List[HypothesisResponse])
async def get_session_hypotheses(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db.hypotheses.find({"session_id": session_id}).sort("created_at", -1)
    
    results = []
    async for h in cursor:
        results.append(HypothesisResponse(
            id=str(h["_id"]),
            user_id=str(h.get("user_id", current_user["id"])),
            session_id=str(h["session_id"]),
            hypothesis=h.get("hypothesis", ""),
            rationale=h.get("rationale", ""),
            novelty_score=h.get("novelty_score", 7.0),
            testability_score=h.get("testability_score", 7.0),
            plausibility_score=h.get("plausibility_score", 7.0),
            evidence_score=h.get("evidence_score", 7.0),
            falsifiability_score=h.get("falsifiability_score", 7.0),
            overall_score=h.get("overall_score", 7.0),
            variables=h.get("variables", {}),
            supporting_evidence=h.get("supporting_evidence"),
            critique=h.get("critique"),
            status=h.get("status", "generated"),
            created_at=h.get("created_at", datetime.now(timezone.utc)),
            updated_at=h.get("updated_at")
        ))
    return results

@router.get("/hypotheses", response_model=List[HypothesisResponse])
async def get_all_user_hypotheses(status_filter: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    db = get_db()
    query = {"user_id": current_user["id"]}
    if status_filter:
        query["status"] = status_filter

    cursor = db.hypotheses.find(query).sort("created_at", -1)
    results = []
    async for h in cursor:
        results.append(HypothesisResponse(
            id=str(h["_id"]),
            user_id=str(h.get("user_id", current_user["id"])),
            session_id=str(h["session_id"]),
            hypothesis=h.get("hypothesis", ""),
            rationale=h.get("rationale", ""),
            novelty_score=h.get("novelty_score", 7.0),
            testability_score=h.get("testability_score", 7.0),
            plausibility_score=h.get("plausibility_score", 7.0),
            evidence_score=h.get("evidence_score", 7.0),
            falsifiability_score=h.get("falsifiability_score", 7.0),
            overall_score=h.get("overall_score", 7.0),
            variables=h.get("variables", {}),
            supporting_evidence=h.get("supporting_evidence"),
            critique=h.get("critique"),
            status=h.get("status", "generated"),
            created_at=h.get("created_at", datetime.now(timezone.utc)),
            updated_at=h.get("updated_at")
        ))
    return results

@router.get("/hypotheses/{hypothesis_id}", response_model=HypothesisResponse)
async def get_hypothesis(hypothesis_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    try:
        h = await db.hypotheses.find_one({"_id": ObjectId(hypothesis_id)})
    except Exception:
        h = await db.hypotheses.find_one({"_id": hypothesis_id})

    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    return HypothesisResponse(
        id=str(h["_id"]),
        user_id=str(h.get("user_id", current_user["id"])),
        session_id=str(h["session_id"]),
        hypothesis=h.get("hypothesis", ""),
        rationale=h.get("rationale", ""),
        novelty_score=h.get("novelty_score", 7.0),
        testability_score=h.get("testability_score", 7.0),
        plausibility_score=h.get("plausibility_score", 7.0),
        evidence_score=h.get("evidence_score", 7.0),
        falsifiability_score=h.get("falsifiability_score", 7.0),
        overall_score=h.get("overall_score", 7.0),
        variables=h.get("variables", {}),
        supporting_evidence=h.get("supporting_evidence"),
        critique=h.get("critique"),
        status=h.get("status", "generated"),
        created_at=h.get("created_at", datetime.now(timezone.utc)),
        updated_at=h.get("updated_at")
    )

@router.post("/hypotheses/{hypothesis_id}/save", response_model=HypothesisResponse)
async def save_hypothesis(hypothesis_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    try:
        oid = ObjectId(hypothesis_id)
    except Exception:
        oid = hypothesis_id

    h = await db.hypotheses.find_one({"_id": oid})
    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    new_status = "saved" if h.get("status") != "saved" else "generated"
    await db.hypotheses.update_one(
        {"_id": oid},
        {"$set": {"status": new_status, "updated_at": datetime.now(timezone.utc)}}
    )
    h["status"] = new_status
    h["id"] = str(h["_id"])
    return HypothesisResponse(**h)

@router.post("/hypotheses/{hypothesis_id}/refine", response_model=HypothesisResponse)
async def refine_hypothesis(
    hypothesis_id: str,
    refine_in: HypothesisRefineRequest,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    try:
        oid = ObjectId(hypothesis_id)
    except Exception:
        oid = hypothesis_id

    h = await db.hypotheses.find_one({"_id": oid})
    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    prompt = (
        f"Original Hypothesis: {h.get('hypothesis')}\n"
        f"Rationale: {h.get('rationale')}\n"
        f"Refinement Request: {refine_in.refinement_prompt}\n\n"
        "Revise and enhance this hypothesis, ensuring high testability, novelty, and clear variables. "
        "Return JSON with: hypothesis, rationale, novelty_score, testability_score, plausibility_score, "
        "evidence_score, falsifiability_score, variables (independent, dependent, control, confounders), "
        "supporting_evidence, critique."
    )

    try:
        updated = await LLMService.generate_structured_json(
            "You are a rigorous Scientific Editor refining a research hypothesis.",
            prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refine hypothesis: {e}")

    nov = float(updated.get("novelty_score", h.get("novelty_score", 8.0)))
    test = float(updated.get("testability_score", h.get("testability_score", 8.0)))
    plaus = float(updated.get("plausibility_score", h.get("plausibility_score", 8.0)))
    evid = float(updated.get("evidence_score", h.get("evidence_score", 8.0)))
    fals = float(updated.get("falsifiability_score", h.get("falsifiability_score", 8.0)))
    overall = round((nov + test + plaus + evid + fals) / 5.0, 2)

    update_doc = {
        "hypothesis": updated.get("hypothesis", h.get("hypothesis")),
        "rationale": updated.get("rationale", h.get("rationale")),
        "novelty_score": nov,
        "testability_score": test,
        "plausibility_score": plaus,
        "evidence_score": evid,
        "falsifiability_score": fals,
        "overall_score": overall,
        "variables": updated.get("variables", h.get("variables")),
        "supporting_evidence": updated.get("supporting_evidence", h.get("supporting_evidence")),
        "critique": updated.get("critique", h.get("critique")),
        "status": "refined",
        "updated_at": datetime.now(timezone.utc)
    }

    await db.hypotheses.update_one({"_id": oid}, {"$set": update_doc})
    h.update(update_doc)
    h["id"] = str(h["_id"])
    return HypothesisResponse(**h)
